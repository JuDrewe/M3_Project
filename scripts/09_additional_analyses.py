# =============================================================================
# 06_additional_analyses.py
# Additional validation analyses requested by reviewers, using the identical
# feature pipeline (Morgan FP 2048 bit, radius=2, chirality=True, 6 PhysChem
# descriptors, confidence-based sample weights) as 02_train_model.py:
#
#   1) Random-split baseline vs. scaffold split                  -> Table 6
#   2) Tanimoto similarity of test to development set, both splits -> Table 7
#   3) Applicability-domain-restricted held-out performance      -> Table 8
#   4) Sensitivity ablation excluding single-measurement compounds -> Table 9
#
# Outputs: results/tables/{similarity_scaffold_split_test_to_dev,
#          similarity_random_split_test_to_dev, ad_restricted_heldout_predictions,
#          singles_ablation_fold_composition, additional_analyses_summary}.csv/json
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              matthews_corrcoef, balanced_accuracy_score)
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT, "data", "raw", "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
OUT_DIR = os.path.join(_PROJECT, "results", "tables")

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
WEIGHTS = {"active": 1.0, "inactive": 1.0, "active_single": 0.5, "inactive_single": 0.7}
FP_RADIUS, FP_NBITS = 2, 2048
N_SPLITS_INNER, CV_SEED, OUTER_SEED, TEST_FRACTION = 10, 42, 0, 0.20

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=True)


def smiles_to_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    scaf = MurckoScaffold.GetScaffoldForMol(mol) if mol else None
    return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else ""


def fp_objects(smiles_list):
    return [GEN.GetFingerprint(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
            for s in smiles_list]


def build_X(smiles_list):
    X_fp = np.zeros((len(smiles_list), FP_NBITS), dtype=np.float32)
    tmp = np.zeros(FP_NBITS, dtype=np.int8)
    X_pc = np.zeros((len(smiles_list), 6), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            X_pc[i] = np.nan
            continue
        tmp[:] = 0
        ConvertToNumpyArray(GEN.GetFingerprint(mol), tmp)
        X_fp[i] = tmp
        X_pc[i] = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
                   Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)]
    if np.isnan(X_pc).any():
        med = np.nanmedian(X_pc, axis=0)
        r, c = np.where(np.isnan(X_pc))
        X_pc[r, c] = med[c]
    return np.hstack([X_fp, X_pc]).astype(np.float32)


def make_pipe():
    return Pipeline([("clf", LogisticRegression(max_iter=5000, solver="liblinear"))])


def inner_cv(X_smiles, y, w, cv, groups=None, label=""):
    metrics = {"roc_auc": [], "pr_auc": [], "bal_acc": [], "mcc": []}
    for tr, te in cv.split(X_smiles, y, groups=groups):
        pipe = make_pipe()
        pipe.fit(build_X(X_smiles[tr]), y[tr], clf__sample_weight=w[tr])
        proba = pipe.predict_proba(build_X(X_smiles[te]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics["roc_auc"].append(roc_auc_score(y[te], proba))
        metrics["pr_auc"].append(average_precision_score(y[te], proba))
        metrics["bal_acc"].append(balanced_accuracy_score(y[te], pred))
        metrics["mcc"].append(matthews_corrcoef(y[te], pred))
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
    print(f"{label}: " + " | ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in summary.items()))
    return summary


def eval_test(X_smiles_dev, y_dev, w_dev, X_smiles_test, y_test, label=""):
    pipe = make_pipe()
    pipe.fit(build_X(X_smiles_dev), y_dev, clf__sample_weight=w_dev)
    proba = pipe.predict_proba(build_X(X_smiles_test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {"label": label, "n_dev": len(y_dev), "n_test": len(y_test), "prevalence_test": float(y_test.mean()),
           "roc_auc": float(roc_auc_score(y_test, proba)), "pr_auc": float(average_precision_score(y_test, proba)),
           "bal_acc": float(balanced_accuracy_score(y_test, pred)), "mcc": float(matthews_corrcoef(y_test, pred))}
    print(f"[TEST] {label}: ROC-AUC={out['roc_auc']:.3f} MCC={out['mcc']:.3f}")
    return out, proba


def max_sim_to_dev(fps, dev_idx, test_idx):
    dev_fps = [fps[i] for i in dev_idx if fps[i] is not None]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], dev_fps)) if fps[i] is not None else np.nan
                     for i in test_idx])


def main():
    summary = {}
    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["y"] = df["consensus_label"].isin(POS_LABELS).astype(int)
    df["w"] = df["consensus_label"].map(WEIGHTS).astype(float)
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    df = df[df["scaffold"] != ""].reset_index(drop=True)
    print(f"[DATA] n={len(df)}")

    X_all, y_all, w_all, scaf_all = df["smiles"].values, df["y"].values, df["w"].values, df["scaffold"].values

    # --- 1) Scaffold split vs. random split (Table 6) -------------------------
    dev_s, test_s = next(StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED)
                          .split(X_all, y_all, groups=scaf_all))
    scaf_cv = inner_cv(X_all[dev_s], y_all[dev_s], w_all[dev_s],
                       StratifiedGroupKFold(N_SPLITS_INNER, shuffle=True, random_state=CV_SEED),
                       groups=scaf_all[dev_s], label="Scaffold-grouped inner CV")
    scaf_test, scaf_proba = eval_test(X_all[dev_s], y_all[dev_s], w_all[dev_s],
                                       X_all[test_s], y_all[test_s], "Scaffold split")

    dev_r, test_r = train_test_split(np.arange(len(y_all)), test_size=TEST_FRACTION,
                                      stratify=y_all, random_state=OUTER_SEED)
    rand_cv = inner_cv(X_all[dev_r], y_all[dev_r], w_all[dev_r],
                       StratifiedKFold(N_SPLITS_INNER, shuffle=True, random_state=CV_SEED),
                       label="Random (stratified) inner CV")
    rand_test, _ = eval_test(X_all[dev_r], y_all[dev_r], w_all[dev_r],
                              X_all[test_r], y_all[test_r], "Random split")

    summary["scaffold_split"] = {"inner_cv": scaf_cv, "held_out_test": scaf_test, "n_dev": len(dev_s), "n_test": len(test_s)}
    summary["random_split"] = {"inner_cv": rand_cv, "held_out_test": rand_test, "n_dev": len(dev_r), "n_test": len(test_r)}

    # --- 2) Tanimoto similarity test->dev, both splits (Table 7) --------------
    fps = fp_objects(X_all)
    sim_s = max_sim_to_dev(fps, dev_s, test_s)
    sim_r = max_sim_to_dev(fps, dev_r, test_r)
    print(f"[SIMILARITY] scaffold median={np.nanmedian(sim_s):.3f} (n={len(sim_s)})  "
          f"random median={np.nanmedian(sim_r):.3f} (n={len(sim_r)})")
    pd.DataFrame({"scaffold_split_max_sim": sim_s}).to_csv(
        os.path.join(OUT_DIR, "similarity_scaffold_split_test_to_dev.csv"), index=False)
    pd.DataFrame({"random_split_max_sim": sim_r}).to_csv(
        os.path.join(OUT_DIR, "similarity_random_split_test_to_dev.csv"), index=False)
    summary["similarity_test_to_dev"] = {
        "scaffold_split": {"mean": float(np.nanmean(sim_s)), "median": float(np.nanmedian(sim_s)), "n": len(sim_s)},
        "random_split": {"mean": float(np.nanmean(sim_r)), "median": float(np.nanmedian(sim_r)), "n": len(sim_r)}}

    # --- 3) AD-restricted held-out performance (Table 8) ----------------------
    ad_df = pd.DataFrame({"row_index": test_s, "true_label": y_all[test_s],
                          "p_active": scaf_proba, "max_sim_to_dev": sim_s})
    median_sim = float(np.nanmedian(sim_s))
    for thresh_name, thresh in [("median_split", median_sim), ("fixed_0.4", 0.4)]:
        for name, mask in [("in-AD", ad_df["max_sim_to_dev"] >= thresh),
                           ("out-of-AD", ad_df["max_sim_to_dev"] < thresh)]:
            sub = ad_df[mask]
            if sub["true_label"].nunique() < 2 or len(sub) < 5:
                continue
            roc = roc_auc_score(sub["true_label"], sub["p_active"])
            mcc = matthews_corrcoef(sub["true_label"], (sub["p_active"] >= 0.5).astype(int))
            print(f"[AD:{thresh_name}] {name} n={len(sub)} ROC-AUC={roc:.3f} MCC={mcc:.3f}")
    ad_df.to_csv(os.path.join(OUT_DIR, "ad_restricted_heldout_predictions.csv"), index=False)

    # --- 4) Sensitivity ablation excluding single-measurement compounds (Table 9) --
    strict = df[df["consensus_label"].isin(["active", "inactive"])].reset_index(drop=True)
    print(f"[STRICT] n={len(strict)}  prevalence={strict['y'].mean():.3f}")
    X_st, y_st, w_st, scaf_st = strict["smiles"].values, strict["y"].values, strict["w"].values, strict["scaffold"].values
    dev_st, test_st = next(StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED)
                           .split(X_st, y_st, groups=scaf_st))
    cv_strict = StratifiedGroupKFold(N_SPLITS_INNER, shuffle=True, random_state=CV_SEED)

    fold_rows = []
    for fold, (tr, te) in enumerate(cv_strict.split(X_st[dev_st], y_st[dev_st], groups=scaf_st[dev_st]), start=1):
        y_te = y_st[dev_st][te]
        fold_rows.append({"fold": fold, "n_active": int((y_te == 1).sum()), "n_inactive": int((y_te == 0).sum())})
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, "singles_ablation_fold_composition.csv"), index=False)
    zero_inactive_folds = [r["fold"] for r in fold_rows if r["n_inactive"] == 0]
    print(f"[STRICT] folds with 0 inactive test compounds: {zero_inactive_folds}")

    strict_cv = inner_cv(X_st[dev_st], y_st[dev_st], w_st[dev_st], cv_strict,
                        groups=scaf_st[dev_st], label="Strict-label inner CV")
    strict_test, _ = eval_test(X_st[dev_st], y_st[dev_st], w_st[dev_st],
                                X_st[test_st], y_st[test_st], "Strict-label held-out test")
    summary["strict_labels_no_singles"] = {"n_total": len(strict), "n_dev": len(dev_st), "n_test": len(test_st),
                                            "inner_cv": strict_cv, "held_out_test": strict_test,
                                            "zero_inactive_folds": zero_inactive_folds}

    with open(os.path.join(OUT_DIR, "additional_analyses_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[DONE]")


if __name__ == "__main__":
    main()
