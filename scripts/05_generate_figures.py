# =============================================================================
# 05_generate_figures.py
# Generate manuscript figures from saved pipeline outputs
#
# Phase 1 figures:
#   Figure 1  - OOF probability distributions (TP / TN)
#   Figure 2  - OOF probability distributions (FP / FN)
#   Figure 3  - OOF calibration curve
#   Figure 5  - Held-out test evaluation
#   Figure 8  - Distribution of per-scaffold MCC
#   Figure 9  - Scaffold size vs MCC
#   Figure 10 - Hard-scaffold similarity histograms
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)

# =============================================================================
# Paths
# =============================================================================

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

OOF_PATH = os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv")
OOF_CALIB_PATH = os.path.join(TABLES_DIR, "oof_calibration_bins.csv")
HELDOUT_PATH = os.path.join(TABLES_DIR, "heldout_test_predictions.csv")
HELDOUT_CALIB_PATH = os.path.join(TABLES_DIR, "heldout_calibration_bins.csv")

SCAF_SUM_PATH = os.path.join(TABLES_DIR, "scaffold_performance_summary.csv")
HARD_SCAF_PATH = os.path.join(TABLES_DIR, "hard_scaffolds.csv")
DOMAIN_SHIFT_PATH = os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_per_compound.csv")

# Output figure files
FIG1_PATH = os.path.join(FIGURES_DIR, "figure_1_oof_tp_tn_distributions.png")
FIG2_PATH = os.path.join(FIGURES_DIR, "figure_2_oof_fp_fn_distributions.png")
FIG3_PATH = os.path.join(FIGURES_DIR, "figure_3_oof_calibration_curve.png")
FIG5_PATH = os.path.join(FIGURES_DIR, "figure_5_heldout_test_evaluation.png")
FIG8_PATH = os.path.join(FIGURES_DIR, "figure_8_scaffold_mcc_distribution.png")
FIG9_PATH = os.path.join(FIGURES_DIR, "figure_9_scaffold_size_vs_mcc.png")
FIG10_PATH = os.path.join(FIGURES_DIR, "figure_10_hard_scaffold_similarity_histograms.png")

DPI = 300


# =============================================================================
# Load data
# =============================================================================

def load_required_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


# =============================================================================
# Figure 1
# =============================================================================

def figure1_oof_tp_tn(oof_df):
    tp = oof_df[(oof_df["true_label"] == 1) & (oof_df["pred_label"] == 1)]["p_active"].values
    tn = oof_df[(oof_df["true_label"] == 0) & (oof_df["pred_label"] == 0)]["p_active"].values

    plt.figure(figsize=(7, 5))
    plt.hist(tp, bins=30, alpha=0.7, label=f"True positives (n={len(tp)})")
    plt.hist(tn, bins=30, alpha=0.7, label=f"True negatives (n={len(tn)})")
    plt.xlabel("OOF predicted probability")
    plt.ylabel("Frequency")
    plt.title("Figure 1. OOF predicted probability distributions (TP / TN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG1_PATH, dpi=DPI)
    plt.close()
    print(f"[SAVED] {FIG1_PATH}")


# =============================================================================
# Figure 2
# =============================================================================

def figure2_oof_fp_fn(oof_df):
    fp = oof_df[(oof_df["true_label"] == 0) & (oof_df["pred_label"] == 1)]["p_active"].values
    fn = oof_df[(oof_df["true_label"] == 1) & (oof_df["pred_label"] == 0)]["p_active"].values

    plt.figure(figsize=(7, 5))
    if len(fp) > 0:
        plt.hist(fp, bins=20, alpha=0.7, label=f"False positives (n={len(fp)})")
    if len(fn) > 0:
        plt.hist(fn, bins=20, alpha=0.7, label=f"False negatives (n={len(fn)})")
    plt.xlabel("OOF predicted probability")
    plt.ylabel("Frequency")
    plt.title("Figure 2. OOF predicted probability distributions (FP / FN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG2_PATH, dpi=DPI)
    plt.close()
    print(f"[SAVED] {FIG2_PATH}")


# =============================================================================
# Figure 3
# =============================================================================

def figure3_oof_calibration(oof_calib_df, oof_df):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    plt.plot(
        oof_calib_df["mean_pred"].values,
        oof_calib_df["obs_freq"].values,
        marker="o",
        linewidth=1.5,
        label="OOF calibration",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Figure 3. Out-of-fold calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG3_PATH, dpi=DPI)
    plt.close()
    print(f"[SAVED] {FIG3_PATH}")


# =============================================================================
# Figure 5
# =============================================================================

def figure5_heldout_test(heldout_df, heldout_calib_df):
    y_true = heldout_df["true_label"].values
    p = heldout_df["p_active"].values
    y_pred = heldout_df["pred_label"].values

    fpr, tpr, _ = roc_curve(y_true, p)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, p)

    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # A ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, linewidth=1.8, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("A. ROC curve")
    ax.legend()

    # B PR
    ax = axes[0, 1]
    ax.plot(rec, prec, linewidth=1.8, color="black")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B. Precision–recall curve")

    # C Calibration
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    ax.plot(
        heldout_calib_df["mean_pred"].values,
        heldout_calib_df["obs_freq"].values,
        marker="o",
        linewidth=1.5,
        color="black",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("C. Calibration plot")

    # D Confusion matrix in grayscale with larger text
    ax = axes[1, 1]
    im = ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"], fontsize=11)
    ax.set_yticklabels(["Inactive", "Active"], fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title("D. Confusion matrix")

    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            pct = 100 * value / total
            text = f"{value}\n({pct:.1f}%)"
            color = "white" if value > cm.max() * 0.5 else "black"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=color,
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    fig.suptitle("Figure 5. Held-out test set evaluation", y=0.98)
    fig.tight_layout()
    fig.savefig(FIG5_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[SAVED] {FIG5_PATH}")


# =============================================================================
# Figure 8
# =============================================================================

def figure8_scaffold_mcc_distribution(scaf_df):
    vals = scaf_df["mcc"].dropna().values

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=25, alpha=0.8)
    plt.axvline(0.5, linestyle="--", linewidth=1.5, label="Hard-scaffold threshold (MCC = 0.5)")
    plt.xlabel("Per-scaffold MCC")
    plt.ylabel("Frequency")
    plt.title("Figure 8. Distribution of per-scaffold predictive performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG8_PATH, dpi=DPI)
    plt.close()
    print(f"[SAVED] {FIG8_PATH}")


# =============================================================================
# Figure 9
# =============================================================================

def figure9_scaffold_size_vs_mcc(scaf_df, hard_df):
    plt.figure(figsize=(7, 5))

    # all scaffolds
    plt.scatter(
        scaf_df["n"].values,
        scaf_df["mcc"].values,
        alpha=0.7,
        label="Scaffolds",
    )

    # hard scaffolds
    if len(hard_df) > 0:
        plt.scatter(
            hard_df["n"].values,
            hard_df["mcc"].values,
            marker="D",
            s=70,
            label="Hard scaffolds",
        )

    plt.axhline(0.5, linestyle="--", linewidth=1.5)
    plt.xlabel("Scaffold size (n compounds)")
    plt.ylabel("Per-scaffold MCC")
    plt.title("Figure 9. Scaffold size and predictive performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG9_PATH, dpi=DPI)
    plt.close()
    print(f"[SAVED] {FIG9_PATH}")


# =============================================================================
# Figure 10
# =============================================================================

def figure10_hard_scaffold_similarity(domain_df):
    scaffolds = domain_df["scaffold"].dropna().unique().tolist()
    n_scaf = len(scaffolds)

    if n_scaf == 0:
        print("[SKIP] Figure 10: no hard scaffolds found.")
        return

    n_cols = min(2, n_scaf)
    n_rows = math.ceil(n_scaf / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, scaf in zip(axes, scaffolds):
        sub = domain_df[domain_df["scaffold"] == scaf].copy()
        vals = sub["max_sim_to_outside_scaffold"].dropna().values

        ax.hist(vals, bins=15, alpha=0.8)
        ax.set_xlabel("Maximum Tanimoto similarity")
        ax.set_ylabel("Frequency")
        ax.set_title(f"n = {len(vals)}")

    for ax in axes[len(scaffolds):]:
        ax.axis("off")

    fig.suptitle("Figure 10. Maximum Tanimoto similarity distributions for hard-scaffold compounds", y=0.98)
    fig.tight_layout()
    fig.savefig(FIG10_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[SAVED] {FIG10_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("[LOAD] input tables")
    oof_df = load_required_csv(OOF_PATH, "OOF misclassifications")
    oof_calib_df = load_required_csv(OOF_CALIB_PATH, "OOF calibration bins")
    heldout_df = load_required_csv(HELDOUT_PATH, "Held-out predictions")
    heldout_calib_df = load_required_csv(HELDOUT_CALIB_PATH, "Held-out calibration bins")
    scaf_df = load_required_csv(SCAF_SUM_PATH, "Scaffold performance summary")
    hard_df = load_required_csv(HARD_SCAF_PATH, "Hard scaffolds")
    domain_df = load_required_csv(DOMAIN_SHIFT_PATH, "Hard scaffold domain shift")

    print("[MAKE] Figure 1")
    figure1_oof_tp_tn(oof_df)

    print("[MAKE] Figure 2")
    figure2_oof_fp_fn(oof_df)

    print("[MAKE] Figure 3")
    figure3_oof_calibration(oof_calib_df, oof_df)

    print("[MAKE] Figure 5")
    figure5_heldout_test(heldout_df, heldout_calib_df)

    print("[MAKE] Figure 8")
    figure8_scaffold_mcc_distribution(scaf_df)

    print("[MAKE] Figure 9")
    figure9_scaffold_size_vs_mcc(scaf_df, hard_df)

    print("[MAKE] Figure 10")
    figure10_hard_scaffold_similarity(domain_df)

    print("\n[DONE] Phase 1 figures written to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()