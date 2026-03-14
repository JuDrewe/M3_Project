# =============================================================================
# 06_umap_projection.py
# UMAP projection of M3 training chemical space + COCONUT background + top hits
#
# Strategy:
#   1. Fit UMAP on all training compounds only
#   2. Load pre-scored COCONUT file
#   3. Keep all top-1% hits
#   4. Sample 50,000 background COCONUT compounds from the remainder
#   5. Transform COCONUT compounds into the training-defined UMAP space
#
# Inputs:
#   data/raw/ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv
#   data/raw/coconut_screen_ranked.csv
#
# Outputs:
#   results/figures/figure_11_umap_chemical_space.png
#   results/tables/umap_train_embedding.csv
#   results/tables/umap_coconut_projection.csv
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import umap


# =============================================================================
# Paths
# =============================================================================

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    _PROJECT,
    "data",
    "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)

COCONUT_PATH = os.path.join(
    _PROJECT,
    "data",
    "raw",
    "coconut_screen_ranked.csv",
)

FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

FIG_PATH = os.path.join(FIGURES_DIR, "figure_11_umap_chemical_space.png")
TRAIN_EMB_PATH = os.path.join(TABLES_DIR, "umap_train_embedding.csv")
COCONUT_EMB_PATH = os.path.join(TABLES_DIR, "umap_coconut_projection.csv")


# =============================================================================
# Parameters
# =============================================================================

N_BITS = 2048
RADIUS = 2

TOP_PERCENTILE = 0.01
BACKGROUND_SAMPLE_N = 50000
RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "jaccard"


# =============================================================================
# Fingerprint generator
# =============================================================================

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=N_BITS,
    includeChirality=True,
)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return FPGEN.GetFingerprint(mol)


def fps_to_numpy(fps, n_bits=N_BITS):
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


# =============================================================================
# Load and prepare training data
# =============================================================================

print("[LOAD] training data")
train = pd.read_csv(TRAIN_PATH)

train = train[train["consensus_label"].isin(
    ["active", "active_single", "inactive", "inactive_single"]
)].copy()

train = train.dropna(subset=["smiles"]).copy()
train["smiles"] = train["smiles"].astype(str).str.strip()
train = train[train["smiles"] != ""].copy()

train["fp"] = train["smiles"].apply(smiles_to_fp)
train = train[train["fp"].notna()].copy()
train = train.reset_index(drop=True)

train["active"] = train["consensus_label"].isin(
    ["active", "active_single"]
).astype(int)

X_train = fps_to_numpy(train["fp"].tolist())

print("Training compounds:", len(train))


# =============================================================================
# Load and prepare COCONUT data
# =============================================================================

print("[LOAD] COCONUT predictions")
coconut = pd.read_csv(COCONUT_PATH)

required_cols = {"SMILES", "p_antagonist"}
missing = required_cols - set(coconut.columns)
if missing:
    raise ValueError(f"Missing required COCONUT columns: {missing}")

coconut = coconut.dropna(subset=["SMILES", "p_antagonist"]).copy()
coconut["SMILES"] = coconut["SMILES"].astype(str).str.strip()
coconut = coconut[coconut["SMILES"] != ""].copy()

coconut["fp"] = coconut["SMILES"].apply(smiles_to_fp)
coconut = coconut[coconut["fp"].notna()].copy()
coconut = coconut.reset_index(drop=True)

print("COCONUT compounds after SMILES parsing:", len(coconut))


# =============================================================================
# Define top hits and background sample
# =============================================================================

threshold = coconut["p_antagonist"].quantile(1 - TOP_PERCENTILE)
coconut["is_top_hit"] = coconut["p_antagonist"] >= threshold

hits_df = coconut[coconut["is_top_hit"]].copy()
background_pool = coconut[~coconut["is_top_hit"]].copy()

background_n = min(BACKGROUND_SAMPLE_N, len(background_pool))
background_df = background_pool.sample(
    n=background_n,
    random_state=RANDOM_STATE,
).copy()

coconut_plot = pd.concat([background_df, hits_df], ignore_index=True)
coconut_plot = coconut_plot.reset_index(drop=True)

print("Top hits:", len(hits_df))
print("Background sample:", len(background_df))
print("Total COCONUT points projected:", len(coconut_plot))
print("Hit threshold (top 1%):", round(threshold, 6))

X_coconut_plot = fps_to_numpy(coconut_plot["fp"].tolist())


# =============================================================================
# UMAP fit on training data only
# =============================================================================

print("[UMAP] fitting on training compounds only")
reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric=UMAP_METRIC,
    random_state=RANDOM_STATE,
)

emb_train = reducer.fit_transform(X_train)

print("[UMAP] transforming COCONUT projection set")
emb_coconut = reducer.transform(X_coconut_plot)


# =============================================================================
# Save embeddings
# =============================================================================

train_out = train[[
    "molecule_chembl_id", "smiles", "consensus_label", "active"
]].copy()
train_out["UMAP1"] = emb_train[:, 0]
train_out["UMAP2"] = emb_train[:, 1]
train_out.to_csv(TRAIN_EMB_PATH, index=False)
print("[SAVED]", TRAIN_EMB_PATH)

keep_cols = []
for col in [
    "NP-ID", "molecular_name", "molecular_formula", "MW", "XLogP",
    "SMILES", "InChIKey", "Murcko_scaffold", "p_antagonist",
    "penalty_physchem", "score_penalized"
]:
    if col in coconut_plot.columns:
        keep_cols.append(col)

coconut_out = coconut_plot[keep_cols].copy()
coconut_out["is_top_hit"] = coconut_plot["is_top_hit"].values
coconut_out["UMAP1"] = emb_coconut[:, 0]
coconut_out["UMAP2"] = emb_coconut[:, 1]
coconut_out.to_csv(COCONUT_EMB_PATH, index=False)
print("[SAVED]", COCONUT_EMB_PATH)


# =============================================================================
# Plot
# =============================================================================

plt.figure(figsize=(9, 7))

# Training compounds: inactive first, then active
inactive_mask = train["active"].values == 0
active_mask = train["active"].values == 1

plt.scatter(
    emb_train[inactive_mask, 0],
    emb_train[inactive_mask, 1],
    s=14,
    alpha=0.55,
    label="Training inactive",
)

plt.scatter(
    emb_train[active_mask, 0],
    emb_train[active_mask, 1],
    s=14,
    alpha=0.75,
    label="Training active",
)

# Density contours of training chemical space
x = emb_train[:, 0]
y = emb_train[:, 1]

xy = np.vstack([x, y])
kde = gaussian_kde(xy)

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
grid_coords = np.vstack([xx.ravel(), yy.ravel()])
zz = kde(grid_coords).reshape(xx.shape)

plt.contour(
    xx,
    yy,
    zz,
    levels=6,
    colors="black",
    linewidths=1.0,
    alpha=0.8,
)

# COCONUT background
bg_mask = ~coconut_plot["is_top_hit"].values
hit_mask = coconut_plot["is_top_hit"].values

plt.scatter(
    emb_coconut[bg_mask, 0],
    emb_coconut[bg_mask, 1],
    s=5,
    alpha=0.18,
    label="COCONUT background sample",
)

# Top hits
plt.scatter(
    emb_coconut[hit_mask, 0],
    emb_coconut[hit_mask, 1],
    s=18,
    alpha=0.85,
    label="Top 1% predicted hits",
)

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("Figure 11. UMAP representation of M3 chemical space")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()

print("[SAVED]", FIG_PATH)
print("\n[DONE] Figure 11 generated successfully.")