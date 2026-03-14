# =============================================================================
# 04_y_randomization.py
# Y-randomization test using scaffold-grouped cross-validation
#
# Inputs
#   data/raw/ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv
#
# Outputs
#   results/tables/y_randomization_runs.csv
#   results/tables/y_randomization_summary.csv
#   results/figures/y_randomization_histogram.png
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold



# =============================================================================
# Paths
# =============================================================================

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    _PROJECT,
    "data",
    "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)

TABLE_DIR = os.path.join(_PROJECT, "results", "tables")
FIG_DIR = os.path.join(_PROJECT, "results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RUNS_PATH = os.path.join(TABLE_DIR, "y_randomization_runs.csv")
SUMMARY_PATH = os.path.join(TABLE_DIR, "y_randomization_summary.csv")
FIG_PATH = os.path.join(FIG_DIR, "y_randomization_histogram.png")


# =============================================================================
# Parameters
# =============================================================================

N_BITS = 2048
RADIUS = 2
N_SPLITS = 5
N_PERMUTATIONS = 100
RANDOM_STATE = 42


# =============================================================================
# Helper functions
# =============================================================================

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=N_BITS,
    includeChirality=True,
)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FPGEN.GetFingerprint(mol)


def fps_to_numpy(fps):

    arr = np.zeros((len(fps), N_BITS), dtype=int)

    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])

    return arr


def smiles_to_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


# =============================================================================
# Load data
# =============================================================================

print("[LOAD]", DATA_PATH)

df = pd.read_csv(DATA_PATH)

df = df[df["consensus_label"].isin(
    ["active", "active_single", "inactive", "inactive_single"]
)].copy()

df["fp"] = df["smiles"].apply(smiles_to_fp)
df = df[df["fp"].notna()].copy()

df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)

X = fps_to_numpy(df["fp"].tolist())

y = df["consensus_label"].isin(
    ["active", "active_single"]
).astype(int).values

scaffolds = df["scaffold"].values

print("X shape:", X.shape)
print("y:", len(y))
print("scaffolds:", len(scaffolds))


# =============================================================================
# Model
# =============================================================================

pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(
        solver="liblinear",
        penalty="l2",
        class_weight="balanced",
        max_iter=5000,
    ))
])


# =============================================================================
# Real ROC-AUC
# =============================================================================

gkf = GroupKFold(n_splits=N_SPLITS)

fold_aucs = []

for train_idx, test_idx in gkf.split(X, y, groups=scaffolds):

    pipeline.fit(X[train_idx], y[train_idx])

    prob = pipeline.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y[test_idx], prob)

    fold_aucs.append(auc)

real_auc = np.mean(fold_aucs)

print("\nReal ROC-AUC:", round(real_auc, 4))


# =============================================================================
# Y randomization
# =============================================================================

rng = np.random.RandomState(RANDOM_STATE)

perm_aucs = []

for i in range(N_PERMUTATIONS):

    y_perm = rng.permutation(y)

    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y_perm, groups=scaffolds):

        pipeline.fit(X[train_idx], y_perm[train_idx])

        prob = pipeline.predict_proba(X[test_idx])[:, 1]

        auc = roc_auc_score(y_perm[test_idx], prob)

        fold_aucs.append(auc)

    perm_auc = np.mean(fold_aucs)

    perm_aucs.append(perm_auc)

    if (i + 1) % 10 == 0:
        print("Permutation", i + 1, "/", N_PERMUTATIONS)


perm_aucs = np.array(perm_aucs)

p_value = (np.sum(perm_aucs >= real_auc) + 1) / (len(perm_aucs) + 1)


# =============================================================================
# Save results
# =============================================================================

runs_df = pd.DataFrame({
    "perm_auc": perm_aucs
})

runs_df.to_csv(RUNS_PATH, index=False)

summary_df = pd.DataFrame([{
    "real_auc": real_auc,
    "perm_mean_auc": perm_aucs.mean(),
    "perm_std_auc": perm_aucs.std(),
    "p_value": p_value,
    "n_permutations": N_PERMUTATIONS
}])

summary_df.to_csv(SUMMARY_PATH, index=False)

print("\nSaved:")
print(RUNS_PATH)
print(SUMMARY_PATH)


# =============================================================================
# Plot
# =============================================================================

plt.figure(figsize=(6,4))

plt.hist(
    perm_aucs,
    bins=25,
    alpha=0.75,
    label="Permuted labels"
)

plt.axvline(
    real_auc,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Real model (AUC={real_auc:.3f})"
)

plt.xlabel("ROC-AUC")
plt.ylabel("Frequency")
plt.title("Y-randomization test")

plt.legend()

plt.tight_layout()

plt.savefig(FIG_PATH, dpi=300)

print("Saved:", FIG_PATH)

plt.close()

print("\nDONE.")