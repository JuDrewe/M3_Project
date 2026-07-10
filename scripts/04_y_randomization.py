# =============================================================================
# 04_y_randomization.py
# Y-randomization (label permutation) test, using the identical scaffold-
# stratified CV protocol, features, and sample weights as the main model
# (02_train_model.py), so the "real" reference line matches the reported
# OOF ROC-AUC exactly.
#
# Outputs: results/tables/y_randomization_{runs,summary}.csv,
#          results/figures/y_randomization_histogram.png
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT, "data", "raw",
                         "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
TABLE_DIR = os.path.join(_PROJECT, "results", "tables")
FIG_DIR = os.path.join(_PROJECT, "results", "figures")

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
WEIGHTS = {"active": 1.0, "inactive": 1.0, "active_single": 0.5, "inactive_single": 0.7}

N_BITS, RADIUS = 2048, 2
N_SPLITS, CV_SEED = 10, 42          # identical to 02_train_model.py OOF CV
OUTER_N_SPLITS, OUTER_SEED = 5, 0   # identical dev/held-out split
N_PERMUTATIONS, PERM_SEED = 100, 42

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)


def featurise(df):
    X = np.zeros((len(df), N_BITS), dtype=np.float32)
    scaffolds = []
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        fp = FPGEN.GetFingerprint(mol) if mol is not None else None
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, X[i])
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else "")
    return X, np.array(scaffolds)


def mean_oof_auc(X, y, w, groups, cv):
    """Out-of-fold ROC-AUC pooled across folds (matches 02_train_model.py)."""
    oof_true, oof_proba = [], []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        pipe = Pipeline([("clf", LogisticRegression(max_iter=5000, solver="liblinear"))])
        pipe.fit(X[train_idx], y[train_idx], clf__sample_weight=w[train_idx])
        oof_proba.extend(pipe.predict_proba(X[test_idx])[:, 1])
        oof_true.extend(y[test_idx])
    return roc_auc_score(oof_true, oof_proba)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    X, scaffolds = featurise(df)
    valid = scaffolds != ""
    X, df = X[valid], df[valid].reset_index(drop=True)
    scaffolds = scaffolds[valid]
    y = df["consensus_label"].isin(POS_LABELS).astype(int).values
    w = df["consensus_label"].map(WEIGHTS).astype(float).values
    print(f"[DATA] X={X.shape} n_scaffolds={len(set(scaffolds))}")

    # Restrict to the development set (identical outer split as 02_train_model.py)
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_N_SPLITS, shuffle=True, random_state=OUTER_SEED)
    dev_idx, _ = next(outer_cv.split(X, y, groups=scaffolds))
    X_dev, y_dev, w_dev, groups_dev = X[dev_idx], y[dev_idx], w[dev_idx], scaffolds[dev_idx]

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    real_auc = mean_oof_auc(X_dev, y_dev, w_dev, groups_dev, cv)
    print(f"[REAL] OOF ROC-AUC={real_auc:.4f}")

    rng = np.random.RandomState(PERM_SEED)
    perm_aucs = np.array([
        mean_oof_auc(X_dev, rng.permutation(y_dev), w_dev, groups_dev, cv)
        for _ in range(N_PERMUTATIONS)
    ])
    p_value = (np.sum(perm_aucs >= real_auc) + 1) / (len(perm_aucs) + 1)
    print(f"[PERMUTED] mean={perm_aucs.mean():.4f} std={perm_aucs.std():.4f} p={p_value:.3f}")

    pd.DataFrame({"perm_auc": perm_aucs}).to_csv(os.path.join(TABLE_DIR, "y_randomization_runs.csv"), index=False)
    pd.DataFrame([{"real_auc": real_auc, "perm_mean_auc": perm_aucs.mean(), "perm_std_auc": perm_aucs.std(),
                   "p_value": p_value, "n_permutations": N_PERMUTATIONS}]).to_csv(
        os.path.join(TABLE_DIR, "y_randomization_summary.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(perm_aucs, bins=25, alpha=0.75, label="Permuted labels")
    plt.axvline(real_auc, color="red", linestyle="--", linewidth=2, label=f"Real model (AUC={real_auc:.3f})")
    plt.xlabel("ROC-AUC"); plt.ylabel("Frequency"); plt.title("Y-randomization test"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "y_randomization_histogram.png"), dpi=300)
    plt.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
