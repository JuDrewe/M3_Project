# =============================================================================
# generate_figures.py
# M3 Muscarinic Receptor Antagonist — Figure Generation
#
# Reads outputs produced by train_model.py and generates publication-quality
# figures saved to results/figures/.
#
# Figures produced:
#   fig_01_oof_calibration.png         — OOF calibration curve + Brier score
#   fig_02_oof_prob_tp_tn.png          — OOF probability distributions: TP vs TN
#   fig_03_oof_prob_fp_fn.png          — OOF probability distributions: FP vs FN
#   fig_04_scaffold_mcc_dist.png       — Per-scaffold MCC histogram
#   fig_05_scaffold_mcc_vs_size.png    — Per-scaffold MCC vs scaffold size scatter
#   fig_06_domain_shift_similarity.png — Tanimoto similarity for hard scaffolds
#   fig_07_umap_hard_scaffolds.png     — UMAP: hard scaffolds highlighted  [optional]
#   fig_08_umap_misclassifications.png — UMAP: OOF FP/FN highlighted       [optional]
#
# Requirements:
#   - numpy, pandas, matplotlib, scikit-learn, rdkit
#   - umap-learn  (optional — figs 07/08 only; skipped if not installed)
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

TABLES_DIR  = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")
MODELS_DIR  = os.path.join(_PROJECT, "models")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Input tables (written by train_model.py)
MISCLASS_PATH     = os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv")
SCAF_SUM_PATH     = os.path.join(TABLES_DIR, "scaffold_performance_summary.csv")
HARD_SCAF_PATH    = os.path.join(TABLES_DIR, "hard_scaffolds.csv")
DOMAIN_SHIFT_PATH = os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_per_compound.csv")

# Model bundle (used to read the OOF-selected threshold)
MODEL_PATH = os.path.join(MODELS_DIR, "m3_fp_physchem_scaffoldcv.joblib")

# Original data CSV — required only for UMAP figures (figs 07/08).
# Set to None to skip UMAP figures.
DATA_PATH = os.path.join(
    _PROJECT, "data", "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)

# Hard-scaffold threshold line drawn on figs 04/05 — must match train_model.py
HARD_MCC_MAX = 0.5

# Calibration bins
CAL_N_BINS   = 10
CAL_STRATEGY = "quantile"

# Figure output settings
DPI    = 200
STYLE  = "seaborn-v0_8-whitegrid"
FIGEXT = ".png"

# UMAP settings (only used if umap-learn is available and DATA_PATH is set)
UMAP_NEIGHBORS    = 15
UMAP_MIN_DIST     = 0.10
UMAP_RANDOM_STATE = 42

# Fingerprint settings for UMAP (must match train_model.py)
FP_RADIUS        = 2
FP_NBITS         = 2048
FP_USE_CHIRALITY = True


# =============================================================================
# 2. UTILITIES
# =============================================================================

def savefig(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name + FIGEXT)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def load_threshold(fallback: float = 0.5) -> float:
    """Read the OOF-selected decision threshold from the saved model bundle."""
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        t = float(bundle.get("threshold", fallback))
        print(f"[THRESHOLD] Loaded from model bundle: {t:.3f}")
        return t
    print(f"[THRESHOLD] Model bundle not found — using fallback {fallback}")
    return fallback


# =============================================================================
# 3. FIGURE FUNCTIONS
# =============================================================================

def fig_oof_calibration(oof: pd.DataFrame) -> None:
    """Fig 01 — OOF calibration curve with Brier score annotation."""
    y_true = oof["true_label"].values
    y_prob = oof["p_active"].values

    brier    = brier_score_loss(y_true, y_prob)
    baseline = float(y_true.mean() * (1.0 - y_true.mean()))

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=CAL_N_BINS, strategy=CAL_STRATEGY
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label="OOF predictions")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("OOF Calibration Curve")
    ax.text(
        0.04, 0.93,
        f"Brier = {brier:.4f}  (baseline = {baseline:.4f})",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig(fig, "fig_01_oof_calibration")


def fig_oof_prob_tp_tn(oof: pd.DataFrame) -> None:
    """Fig 02 — OOF probability distributions for TP and TN."""
    y_true = oof["true_label"].values
    y_pred = oof["pred_label"].values
    y_prob = oof["p_active"].values

    tp = y_prob[(y_true == 1) & (y_pred == 1)]
    tn = y_prob[(y_true == 0) & (y_pred == 0)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(tp, bins=30, alpha=0.65, label=f"TP  (n={len(tp)})")
    ax.hist(tn, bins=30, alpha=0.65, label=f"TN  (n={len(tn)})")
    ax.set_xlabel("OOF predicted probability (p_active)")
    ax.set_ylabel("Count")
    ax.set_title("OOF Probability Distribution: TP vs TN")
    ax.legend(fontsize=9)
    savefig(fig, "fig_02_oof_prob_tp_tn")


def fig_oof_prob_fp_fn(oof: pd.DataFrame, threshold: float) -> None:
    """Fig 03 — OOF probability distributions for FP and FN errors."""
    fp = oof.loc[oof["error_type"] == "FP", "p_active"].values
    fn = oof.loc[oof["error_type"] == "FN", "p_active"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if len(fp):
        ax.hist(fp, bins=20, alpha=0.7, label=f"FP  (n={len(fp)})")
    if len(fn):
        ax.hist(fn, bins=20, alpha=0.7, label=f"FN  (n={len(fn)})")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("OOF predicted probability (p_active)")
    ax.set_ylabel("Count")
    ax.set_title("OOF Probability Distribution: FP vs FN")
    ax.legend(fontsize=9)
    savefig(fig, "fig_03_oof_prob_fp_fn")


def fig_scaffold_mcc_dist(scaf: pd.DataFrame) -> None:
    """Fig 04 — Histogram of per-scaffold MCC values."""
    mcc_vals = scaf["mcc"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(mcc_vals, bins=20, edgecolor="white", lw=0.5)
    ax.axvline(HARD_MCC_MAX, color="crimson", linestyle="--", lw=1.5,
               label=f"Hard scaffold threshold (MCC < {HARD_MCC_MAX})")
    ax.set_xlabel("Per-scaffold MCC (OOF)")
    ax.set_ylabel("Number of scaffolds")
    ax.set_title("Distribution of Per-Scaffold MCC")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    savefig(fig, "fig_04_scaffold_mcc_dist")


def fig_scaffold_mcc_vs_size(scaf: pd.DataFrame, hard: pd.DataFrame) -> None:
    """Fig 05 — Scaffold size vs OOF MCC, hard scaffolds highlighted."""
    hard_set  = set(hard["scaffold"].tolist())
    mask_hard = scaf["scaffold"].isin(hard_set)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(
        scaf.loc[~mask_hard, "n"], scaf.loc[~mask_hard, "mcc"],
        s=60, alpha=0.7, label="Scaffold",
    )
    ax.scatter(
        scaf.loc[mask_hard, "n"], scaf.loc[mask_hard, "mcc"],
        s=80, alpha=0.9, marker="D", label="Hard scaffold",
    )
    ax.axhline(HARD_MCC_MAX, color="crimson", linestyle="--", lw=1.2,
               label=f"MCC = {HARD_MCC_MAX}")
    ax.set_xlabel("Scaffold size (n OOF compounds)")
    ax.set_ylabel("Per-scaffold MCC (OOF)")
    ax.set_title("Per-Scaffold MCC vs Scaffold Size")
    ax.legend(fontsize=9)
    savefig(fig, "fig_05_scaffold_mcc_vs_size")


def fig_domain_shift_similarity(domain: pd.DataFrame, hard: pd.DataFrame) -> None:
    """Fig 06 — Max Tanimoto similarity to outside-scaffold compounds,
    one panel per hard scaffold."""
    scaffolds = hard["scaffold"].tolist()
    n_scaf    = len(scaffolds)

    if n_scaf == 0:
        print("[SKIP] fig_06 — no hard scaffolds.")
        return

    mcc_map = hard.set_index("scaffold")["mcc"].to_dict()

    fig, axes = plt.subplots(
        1, n_scaf,
        figsize=(max(4.0, 3.0 * n_scaf), 4.5),
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]

    for ax, scaf in zip(axes, scaffolds):
        vals = domain.loc[domain["scaffold"] == scaf,
                          "max_sim_to_outside_scaffold"].dropna().values
        ax.hist(vals, bins=15, edgecolor="white", lw=0.4)
        ax.set_xlabel("Max Tanimoto similarity\nto outside-scaffold compounds", fontsize=8)
        ax.set_title(
            f"MCC = {mcc_map.get(scaf, float('nan')):.2f}\nn = {len(vals)}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)

    axes[0].set_ylabel("Count")
    fig.suptitle("Domain Shift: Hard Scaffold Compounds", fontsize=11, y=1.01)
    fig.tight_layout()
    savefig(fig, "fig_06_domain_shift_similarity")


# =============================================================================
# OPTIONAL: UMAP figures (require umap-learn + DATA_PATH)
# =============================================================================

def _morgan_fp_matrix(smiles_list) -> np.ndarray:
    from rdkit import Chem
    from rdkit.DataStructs import ConvertToNumpyArray
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=FP_USE_CHIRALITY
    )
    smiles_list = np.asarray(smiles_list, dtype=object)
    X   = np.zeros((len(smiles_list), FP_NBITS), dtype=np.float32)
    tmp = np.zeros((FP_NBITS,), dtype=np.int8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        bv = gen.GetFingerprint(mol)
        tmp[:] = 0
        ConvertToNumpyArray(bv, tmp)
        X[i, :] = tmp
    return X


def _compute_umap_embedding(smiles_list) -> np.ndarray:
    import umap as _umap
    X = _morgan_fp_matrix(smiles_list)
    reducer = _umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="jaccard",
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(X)


def fig_umap_hard_scaffolds(df_all: pd.DataFrame, hard: pd.DataFrame) -> None:
    """Fig 07 — UMAP of all compounds; hard scaffolds highlighted."""
    hard_set  = set(hard["scaffold"].tolist())
    mask_hard = df_all["scaffold"].isin(hard_set)

    emb = _compute_umap_embedding(df_all["smiles"].values)
    u1, u2 = emb[:, 0], emb[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(u1[~mask_hard], u2[~mask_hard],
               s=8, alpha=0.30, rasterized=True, label="All compounds")
    ax.scatter(u1[mask_hard],  u2[mask_hard],
               s=30, alpha=0.90, label=f"Hard scaffolds (n={mask_hard.sum()})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical Space (Morgan FP, Jaccard) — Hard Scaffolds")
    ax.legend(fontsize=9, markerscale=1.5)
    savefig(fig, "fig_07_umap_hard_scaffolds")


def fig_umap_misclassifications(df_all: pd.DataFrame, oof: pd.DataFrame) -> None:
    """Fig 08 — UMAP of all compounds; OOF FP and FN highlighted."""
    err_map = oof.set_index("row_index")["error_type"]
    df_all  = df_all.copy()
    # Use the preserved original row_index column, not df_all.index,
    # because index labels are unreliable after row filtering.
    df_all["error_oof"] = df_all["row_index"].map(err_map).fillna("NA")

    emb = _compute_umap_embedding(df_all["smiles"].values)
    u1, u2 = emb[:, 0], emb[:, 1]

    mask_ok = df_all["error_oof"] == "OK"
    mask_fp = df_all["error_oof"] == "FP"
    mask_fn = df_all["error_oof"] == "FN"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(u1[mask_ok], u2[mask_ok],
               s=8,  alpha=0.20, rasterized=True, label=f"OOF OK  (n={mask_ok.sum()})")
    ax.scatter(u1[mask_fp], u2[mask_fp],
               s=30, alpha=0.90, label=f"OOF FP  (n={mask_fp.sum()})")
    ax.scatter(u1[mask_fn], u2[mask_fn],
               s=30, alpha=0.90, label=f"OOF FN  (n={mask_fn.sum()})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical Space (Morgan FP, Jaccard) — OOF Misclassifications")
    ax.legend(fontsize=9, markerscale=1.5)
    savefig(fig, "fig_08_umap_misclassifications")


# =============================================================================
# MAIN
# =============================================================================

def main():
    plt.style.use(STYLE)

    # -------------------------------------------------------------------------
    # Load tables
    # -------------------------------------------------------------------------
    print("[LOAD] Reading results tables ...")
    oof    = pd.read_csv(MISCLASS_PATH)
    scaf   = pd.read_csv(SCAF_SUM_PATH)
    hard   = pd.read_csv(HARD_SCAF_PATH)
    domain = pd.read_csv(DOMAIN_SHIFT_PATH)

    print(f"  OOF predictions   : {len(oof)} rows")
    print(f"  Scaffolds         : {len(scaf)} (with ≥2 classes in OOF)")
    print(f"  Hard scaffolds    : {len(hard)}")
    print(f"  Domain shift rows : {len(domain)}")

    threshold = load_threshold()

    # -------------------------------------------------------------------------
    # Figures 01–06 (no external data needed)
    # -------------------------------------------------------------------------
    fig_oof_calibration(oof)
    fig_oof_prob_tp_tn(oof)
    fig_oof_prob_fp_fn(oof, threshold)
    fig_scaffold_mcc_dist(scaf)
    fig_scaffold_mcc_vs_size(scaf, hard)
    fig_domain_shift_similarity(domain, hard)

    # -------------------------------------------------------------------------
    # Figures 07–08: UMAP (optional)
    # -------------------------------------------------------------------------
    if DATA_PATH is None:
        print("\n[UMAP] DATA_PATH is None — skipping figs 07/08.")
        return

    if not os.path.exists(DATA_PATH):
        print(f"\n[UMAP] Data file not found: {DATA_PATH}")
        print("       Set DATA_PATH at the top of this script — skipping figs 07/08.")
        return

    try:
        import umap as _umap_check  # noqa: F401
    except ImportError:
        print("\n[UMAP] umap-learn not installed — skipping figs 07/08.")
        print("       Install with: pip install umap-learn")
        return

    print("\n[UMAP] Loading full dataset for UMAP projection ...")
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    POS_LABELS = {"active", "active_single"}
    NEG_LABELS = {"inactive", "inactive_single"}

    df_all = pd.read_csv(DATA_PATH)
    # Preserve original CSV row positions before any filtering drops rows.
    # These are used in fig_umap_misclassifications() to join OOF error labels.
    df_all["row_index"] = df_all.index

    df_all = df_all[df_all["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df_all = df_all.dropna(subset=["smiles"]).copy()
    df_all["smiles"] = df_all["smiles"].astype(str).str.strip()
    df_all = df_all[df_all["smiles"] != ""].copy()
    df_all["y"] = df_all["consensus_label"].isin(POS_LABELS).astype(int)

    def _to_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf else ""

    df_all["scaffold"] = df_all["smiles"].apply(_to_scaffold)
    df_all = df_all[df_all["scaffold"] != ""].copy()   # no reset_index — row_index preserved

    print(f"  {len(df_all)} compounds loaded for UMAP.")
    print("[UMAP] Computing embedding (this may take a few minutes) ...")

    fig_umap_hard_scaffolds(df_all, hard)
    fig_umap_misclassifications(df_all, oof)

    print("\n[DONE] All figures written to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
