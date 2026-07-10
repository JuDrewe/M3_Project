# =============================================================================
# 02_train_model.py
# Scaffold-stratified training and evaluation of the M3 antagonist classifier.
#
# Steps: load curated data -> compute Bemis-Murcko scaffolds -> scaffold-
# stratified dev/held-out split -> compare FP-only vs FP+PhysChem by scaffold
# CV -> fit final model -> collect OOF predictions -> select threshold by MCC
# -> save misclassifications, per-scaffold summary, difficult scaffolds,
# domain-shift table -> evaluate on held-out test set.
#
# Outputs: models/m3_fp_physchem_scaffoldcv.joblib,
#          results/tables/{scaffold_cv_misclassifications,
#          scaffold_performance_summary, hard_scaffolds,
#          hard_scaffolds_domain_shift_per_compound,
#          hard_scaffolds_domain_shift_summary, heldout_test_predictions}.csv
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray

# --- Configuration -----------------------------------------------------------
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT, "data", "raw",
                         "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
MODELS_DIR = os.path.join(_PROJECT, "models")
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
WEIGHTS = {"active": 1.0, "inactive": 1.0, "active_single": 0.5, "inactive_single": 0.7}

FP_RADIUS, FP_NBITS, FP_CHIRALITY = 2, 2048, True
N_SPLITS, OUTER_N_SPLITS = 10, 5
CV_SEED, OUTER_SEED = 42, 0
HARD_MIN_N, HARD_MCC_MAX = 10, 0.5

# --- Featurisation -------------------------------------------------------
def smiles_to_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    scaf = MurckoScaffold.GetScaffoldForMol(mol) if mol else None
    return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else ""

def _morgan_gen():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=FP_CHIRALITY)

def morgan_fp_matrix(smiles_list):
    gen = _morgan_gen()
    X = np.zeros((len(smiles_list), FP_NBITS), dtype=np.float32)
    tmp = np.zeros(FP_NBITS, dtype=np.int8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        tmp[:] = 0
        ConvertToNumpyArray(gen.GetFingerprint(mol), tmp)
        X[i, :] = tmp
    return X

def physchem_matrix(smiles_list):
    X = np.zeros((len(smiles_list), 6), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            X[i, :] = np.nan
            continue
        X[i] = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)]
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
    return X

def build_X(smiles_list, use_physchem):
    X_fp = morgan_fp_matrix(smiles_list)
    return np.hstack([X_fp, physchem_matrix(smiles_list)]).astype(np.float32) if use_physchem else X_fp

def compute_fp_objects(smiles_list):
    gen = _morgan_gen()
    return [gen.GetFingerprint(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
            for s in smiles_list]

def make_pipe():
    return Pipeline([("clf", LogisticRegression(max_iter=5000, solver="liblinear"))])

# --- Data loading --------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""].copy()
    df["y"] = df["consensus_label"].isin(POS_LABELS).astype(int)
    df["w"] = df["consensus_label"].map(WEIGHTS).astype(float)
    return df

def eval_mode(name, use_physchem, cv, X_smiles, y, w, groups):
    metrics = {"roc_auc": [], "pr_auc": [], "bal_acc": [], "mcc": []}
    for train_idx, test_idx in cv.split(X_smiles, y, groups=groups):
        X_tr = build_X(X_smiles[train_idx], use_physchem)
        X_te = build_X(X_smiles[test_idx], use_physchem)
        pipe = make_pipe()
        pipe.fit(X_tr, y[train_idx], clf__sample_weight=w[train_idx])
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics["roc_auc"].append(roc_auc_score(y[test_idx], proba))
        metrics["pr_auc"].append(average_precision_score(y[test_idx], proba))
        metrics["bal_acc"].append(balanced_accuracy_score(y[test_idx], pred))
        metrics["mcc"].append(matthews_corrcoef(y[test_idx], pred))
    print(f"{name}: " + " | ".join(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" for k, v in metrics.items()))
    return metrics

# --- Main ------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    df = df[df["scaffold"] != ""].copy()
    print(f"[DATA] n={len(df)}")

    X_smiles, y, w, groups = df["smiles"].values, df["y"].values.astype(int), df["w"].values.astype(float), df["scaffold"].values

    outer_cv = StratifiedGroupKFold(n_splits=OUTER_N_SPLITS, shuffle=True, random_state=OUTER_SEED)
    dev_idx, test_idx = next(outer_cv.split(X_smiles, y, groups=groups))
    X_dev, X_test = X_smiles[dev_idx], X_smiles[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]
    w_dev, groups_dev = w[dev_idx], groups[dev_idx]
    print(f"[SPLIT] dev={len(dev_idx)} test={len(test_idx)}  prevalence dev={y_dev.mean():.3f} test={y_test.mean():.3f}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    eval_mode("FP-only", False, cv, X_dev, y_dev, w_dev, groups_dev)
    eval_mode("FP+PhysChem", True, cv, X_dev, y_dev, w_dev, groups_dev)

    # Final model on full dev set
    X_dev_full = build_X(X_dev, True)
    final_pipe = make_pipe()
    final_pipe.fit(X_dev_full, y_dev, clf__sample_weight=w_dev)

    # Out-of-fold predictions for threshold selection + error analysis
    oof = []
    for train_idx, te_idx in cv.split(X_dev, y_dev, groups=groups_dev):
        pipe = make_pipe()
        pipe.fit(build_X(X_dev[train_idx], True), y_dev[train_idx], clf__sample_weight=w_dev[train_idx])
        proba = pipe.predict_proba(build_X(X_dev[te_idx], True))[:, 1]
        for i, idx in enumerate(te_idx):
            oof.append((int(dev_idx[idx]), float(proba[i]), int(y_dev[idx])))

    oof_idx, oof_proba, oof_true = map(np.array, zip(*oof))
    thresholds = np.linspace(0.05, 0.95, 181)
    mccs = [matthews_corrcoef(oof_true, (oof_proba >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[np.argmax(mccs)])
    print(f"[THRESHOLD] {best_t:.2f} -> OOF MCC={max(mccs):.3f}")

    joblib.dump({"pipe": final_pipe, "use_physchem": True, "fp_radius": FP_RADIUS,
                 "fp_nbits": FP_NBITS, "fp_use_chirality": FP_CHIRALITY, "physchem_dim": 6,
                 "threshold": best_t}, os.path.join(MODELS_DIR, "m3_fp_physchem_scaffoldcv.joblib"))

    # Misclassification table
    pred = (oof_proba >= best_t).astype(int)
    err = np.where((pred == 1) & (oof_true == 0), "FP", np.where((pred == 0) & (oof_true == 1), "FN", "OK"))
    err_df = pd.DataFrame({
        "row_index": oof_idx,
        "molecule_chembl_id": df.iloc[oof_idx]["molecule_chembl_id"].values,
        "consensus_label": df.iloc[oof_idx]["consensus_label"].values,
        "smiles": df.iloc[oof_idx]["smiles"].values,
        "scaffold": df.iloc[oof_idx]["scaffold"].values,
        "true_label": oof_true, "pred_label": pred, "p_active": oof_proba, "error_type": err,
    })
    err_df.to_csv(os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv"), index=False)

    # Per-scaffold performance summary + difficult scaffolds
    rows = []
    for scaf, g in err_df.groupby("scaffold"):
        if g["true_label"].nunique() < 2:
            continue
        rows.append({"scaffold": scaf, "n": len(g),
                     "mcc": matthews_corrcoef(g["true_label"], g["pred_label"]),
                     "bal_acc": balanced_accuracy_score(g["true_label"], g["pred_label"]),
                     "n_errors": (g["error_type"] != "OK").sum(), "pos_frac": g["true_label"].mean()})
    scaf_df = pd.DataFrame(rows).sort_values(["mcc", "n"], ascending=[True, False])
    scaf_df.to_csv(os.path.join(TABLES_DIR, "scaffold_performance_summary.csv"), index=False)

    hard = scaf_df.query(f"n >= {HARD_MIN_N} and mcc < {HARD_MCC_MAX}").copy()
    hard.to_csv(os.path.join(TABLES_DIR, "hard_scaffolds.csv"), index=False)
    print(f"[DIFFICULT SCAFFOLDS] n={len(hard)}")

    # Domain-shift table for difficult scaffolds
    if len(hard):
        fps = compute_fp_objects(df["smiles"].values)
        valid = np.array([i for i, fp in enumerate(fps) if fp is not None])
        domain_rows = []
        for scaf in hard["scaffold"]:
            mask = df["scaffold"].values == scaf
            scaf_idx, train_idx = valid[mask[valid]], valid[~mask[valid]]
            train_fps = [fps[i] for i in train_idx]
            for idx in scaf_idx:
                sims = DataStructs.BulkTanimotoSimilarity(fps[idx], train_fps)
                domain_rows.append({
                    "scaffold": scaf, "row_index": int(idx),
                    "molecule_chembl_id": df.iloc[idx]["molecule_chembl_id"],
                    "consensus_label": df.iloc[idx]["consensus_label"],
                    "true_label": int(df.iloc[idx]["y"]),
                    "max_sim_to_outside_scaffold": float(np.max(sims)),
                    "mean_sim_to_outside_scaffold": float(np.mean(sims)),
                    "p95_sim_to_outside_scaffold": float(np.percentile(sims, 95)),
                })
        domain_df = pd.DataFrame(domain_rows)
        domain_df.to_csv(os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_per_compound.csv"), index=False)
        domain_df.groupby("scaffold").agg(
            n=("row_index", "count"), maxSim_mean=("max_sim_to_outside_scaffold", "mean"),
            maxSim_min=("max_sim_to_outside_scaffold", "min"),
            p95Sim_mean=("p95_sim_to_outside_scaffold", "mean"),
            meanSim_mean=("mean_sim_to_outside_scaffold", "mean"),
        ).reset_index().sort_values("maxSim_mean").to_csv(
            os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_summary.csv"), index=False)

    # Held-out test evaluation
    proba_test = final_pipe.predict_proba(build_X(X_test, True))[:, 1]
    pred_test = (proba_test >= best_t).astype(int)
    print(f"[HELD-OUT] ROC-AUC={roc_auc_score(y_test, proba_test):.3f} "
          f"PR-AUC={average_precision_score(y_test, proba_test):.3f} "
          f"MCC={matthews_corrcoef(y_test, pred_test):.3f} "
          f"BalAcc={balanced_accuracy_score(y_test, pred_test):.3f}")
    print(confusion_matrix(y_test, pred_test))

    pd.DataFrame({
        "row_index": test_idx,
        "molecule_chembl_id": df.iloc[test_idx]["molecule_chembl_id"].values,
        "consensus_label": df.iloc[test_idx]["consensus_label"].values,
        "smiles": df.iloc[test_idx]["smiles"].values,
        "scaffold": df.iloc[test_idx]["scaffold"].values,
        "true_label": y_test, "pred_label": pred_test, "p_active": proba_test,
    }).to_csv(os.path.join(TABLES_DIR, "heldout_test_predictions.csv"), index=False)
    print("[DONE]")


if __name__ == "__main__":
    main()
