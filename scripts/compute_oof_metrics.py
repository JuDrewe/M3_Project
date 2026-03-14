"""
compute_oof_metrics.py
Compute OOF metrics for FP-only and FP+PhysChem models, plus Brier score.
Used to fill in manuscript Table 2 (CV comparison) and calibration stats.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
)

# ---------------------------------------------------------------------------
# Configuration (must match train_model.py)
# ---------------------------------------------------------------------------
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    _PROJECT, "data", "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
WEIGHTS = {
    "active":          1.0,
    "inactive":        1.0,
    "active_single":   0.5,
    "inactive_single": 0.7,
}

FP_RADIUS        = 2
FP_NBITS         = 2048
FP_USE_CHIRALITY = True

N_SPLITS           = 10
OUTER_N_SPLITS     = 5
CV_RANDOM_STATE    = 42
OUTER_RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=FP_USE_CHIRALITY
)


def smiles_to_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf else ""


def fp_matrix(smiles_list) -> np.ndarray:
    smiles_list = np.asarray(smiles_list, dtype=object)
    X = np.zeros((len(smiles_list), FP_NBITS), dtype=np.float32)
    tmp = np.zeros(FP_NBITS, dtype=np.int8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        bv = _gen.GetFingerprint(mol)
        tmp[:] = 0
        ConvertToNumpyArray(bv, tmp)
        X[i] = tmp
    return X


def pc_matrix(smiles_list) -> np.ndarray:
    smiles_list = np.asarray(smiles_list, dtype=object)
    X = np.zeros((len(smiles_list), 6), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            X[i] = np.nan
            continue
        X[i] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
    return X


def build_X(smiles_list, use_physchem: bool) -> np.ndarray:
    X_fp = fp_matrix(smiles_list)
    if not use_physchem:
        return X_fp
    return np.hstack([X_fp, pc_matrix(smiles_list)]).astype(np.float32)


def make_pipe() -> Pipeline:
    return Pipeline([
        ("clf", LogisticRegression(
            max_iter=5000, solver="liblinear", class_weight=None,
        )),
    ])


def run_cv(Xs_dev, y_dev, w_dev, g_dev, cv, use_physchem: bool, label: str):
    oof_p, oof_t = [], []
    for tr, te in cv.split(Xs_dev, y_dev, groups=g_dev):
        X_tr = build_X(Xs_dev[tr], use_physchem)
        X_te = build_X(Xs_dev[te], use_physchem)
        pipe = make_pipe()
        pipe.fit(X_tr, y_dev[tr], clf__sample_weight=w_dev[tr])
        p = pipe.predict_proba(X_te)[:, 1]
        oof_p.extend(p.tolist())
        oof_t.extend(y_dev[te].tolist())

    oof_p = np.array(oof_p)
    oof_t = np.array(oof_t)

    t_arr = np.linspace(0.05, 0.95, 181)
    mccs  = [matthews_corrcoef(oof_t, (oof_p >= t).astype(int)) for t in t_arr]
    best_t = float(t_arr[np.argmax(mccs)])
    pred   = (oof_p >= best_t).astype(int)

    print(f"\n{'='*50}")
    print(f"  {label}  (OOF best threshold = {best_t:.2f})")
    print(f"{'='*50}")
    print(f"  ROC-AUC  : {roc_auc_score(oof_t, oof_p):.4f}")
    print(f"  PR-AUC   : {average_precision_score(oof_t, oof_p):.4f}")
    print(f"  MCC      : {matthews_corrcoef(oof_t, pred):.4f}")
    print(f"  BalAcc   : {balanced_accuracy_score(oof_t, pred):.4f}")
    print(f"  Brier    : {brier_score_loss(oof_t, oof_p):.4f}")
    print(f"  OOF n    : {len(oof_t)}")

    return {
        "label":    label,
        "threshold": best_t,
        "roc_auc":  roc_auc_score(oof_t, oof_p),
        "pr_auc":   average_precision_score(oof_t, oof_p),
        "mcc":      matthews_corrcoef(oof_t, pred),
        "bal_acc":  balanced_accuracy_score(oof_t, pred),
        "brier":    brier_score_loss(oof_t, oof_p),
        "n_oof":    len(oof_t),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[DATA] Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""].copy()
    df["y"] = df["consensus_label"].isin(POS_LABELS).astype(int)
    df["w"] = df["consensus_label"].map(WEIGHTS).astype(float)
    print(f"[DATA] {len(df)} compounds | active frac: {df['y'].mean():.3f}")

    print("[SCAFFOLDS] Computing Murcko scaffolds ...")
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    bad = df["scaffold"].eq("")
    if bad.any():
        print(f"  Dropping {bad.sum()} rows with empty scaffold.")
        df = df[~bad].copy()

    X_smiles = df["smiles"].values
    y        = df["y"].values.astype(int)
    w        = df["w"].values.astype(float)
    groups   = df["scaffold"].values

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_N_SPLITS, shuffle=True, random_state=OUTER_RANDOM_STATE
    )
    _dev_idx, _test_idx = next(outer_cv.split(X_smiles, y, groups=groups))
    Xs_dev  = X_smiles[_dev_idx]
    y_dev   = y[_dev_idx]
    w_dev   = w[_dev_idx]
    g_dev   = groups[_dev_idx]
    print(f"[SPLIT] dev={len(_dev_idx)}  held-out test={len(_test_idx)}")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    r1 = run_cv(Xs_dev, y_dev, w_dev, g_dev, cv, use_physchem=False, label="FP-only")
    r2 = run_cv(Xs_dev, y_dev, w_dev, g_dev, cv, use_physchem=True,  label="FP + PhysChem")

    print("\n[SUMMARY TABLE]")
    results_df = pd.DataFrame([r1, r2])
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
