# =============================================================================
# 05_generate_figures.py
# Generates main-text Figures 1, 2, 3, 5, 6 (final manuscript numbering)
# directly from saved pipeline outputs. Figure 4 (Morgan bit importance) and
# Figure 7 (UMAP) are produced by 07_morgan_feature_importance.py and
# 08_umap_analysis.py respectively.
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
DPI = 300


def load(name):
    return pd.read_csv(os.path.join(TABLES_DIR, name))


def figure1_oof_distributions(oof_df):
    """Panel A: correctly classified (TP/TN); Panel B: misclassified (FP/FN)."""
    tp = oof_df.query("true_label==1 and pred_label==1")["p_active"].values
    tn = oof_df.query("true_label==0 and pred_label==0")["p_active"].values
    fp = oof_df.query("true_label==0 and pred_label==1")["p_active"].values
    fn = oof_df.query("true_label==1 and pred_label==0")["p_active"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].hist(tp, bins=30, alpha=0.7, label=f"True positives (n={len(tp)})")
    axes[0].hist(tn, bins=30, alpha=0.7, label=f"True negatives (n={len(tn)})")
    axes[0].set(xlabel="OOF predicted probability", ylabel="Frequency", title="A. Correctly classified compounds")
    axes[0].legend()
    axes[1].hist(fp, bins=20, alpha=0.7, label=f"False positives (n={len(fp)})")
    axes[1].hist(fn, bins=20, alpha=0.7, label=f"False negatives (n={len(fn)})")
    axes[1].set(xlabel="OOF predicted probability", ylabel="Frequency", title="B. Misclassified compounds")
    axes[1].legend()
    fig.suptitle("Distribution of out-of-fold predicted probabilities", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_1_oof_distributions.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure2_oof_calibration(calib_df):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")
    plt.plot(calib_df["mean_pred"], calib_df["obs_freq"], marker="o", linewidth=1.5, label="OOF calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Out-of-fold calibration curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure_2_oof_calibration.png"), dpi=DPI)
    plt.close()


def figure3_heldout_evaluation(heldout_df, calib_df):
    y, p, pred = heldout_df["true_label"].values, heldout_df["p_active"].values, heldout_df["pred_label"].values
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    cm = confusion_matrix(y, pred)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ax = axes[0, 0]
    ax.plot(fpr, tpr, linewidth=1.8, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="A. ROC curve"); ax.legend()

    ax = axes[0, 1]
    ax.plot(rec, prec, linewidth=1.8, color="black")
    ax.set(xlabel="Recall", ylabel="Precision", title="B. Precision\u2013recall curve")

    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
    ax.plot(calib_df["mean_pred"], calib_df["obs_freq"], marker="o", linewidth=1.5, color="black")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="C. Calibration plot")

    ax = axes[1, 1]
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"], fontsize=11); ax.set_yticklabels(["Inactive", "Active"], fontsize=11)
    ax.set(xlabel="Predicted class", ylabel="True class", title="D. Confusion matrix")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({100*v/total:.1f}%)", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white" if v > cm.max() * 0.5 else "black")

    fig.suptitle("Held-out test set evaluation", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_3_heldout_evaluation.png"), dpi=DPI)
    plt.close(fig)


def figure5_scaffold_performance(scaf_df, hard_df):
    """Panel A: per-scaffold MCC distribution; Panel B: scaffold size vs MCC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].hist(scaf_df["mcc"].dropna(), bins=25, alpha=0.8)
    axes[0].axvline(0.5, linestyle="--", linewidth=1.5, label="Difficult-scaffold threshold (MCC = 0.5)")
    axes[0].set(xlabel="Per-scaffold MCC", ylabel="Frequency", title="A. Distribution of per-scaffold performance")
    axes[0].legend(fontsize=10)

    axes[1].scatter(scaf_df["n"], scaf_df["mcc"], alpha=0.7, label="Scaffolds")
    if len(hard_df):
        axes[1].scatter(hard_df["n"], hard_df["mcc"], marker="D", s=70, label="Difficult scaffolds")
    axes[1].axhline(0.5, linestyle="--", linewidth=1.5)
    axes[1].set(xlabel="Scaffold size (n compounds)", ylabel="Per-scaffold MCC",
                title="B. Scaffold size and predictive performance")
    axes[1].legend(fontsize=10)

    fig.suptitle("Scaffold-level predictive performance", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_5_scaffold_performance.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure6_domain_shift(domain_df):
    scaffolds = domain_df["scaffold"].dropna().unique().tolist()
    n_cols = min(2, len(scaffolds))
    n_rows = math.ceil(len(scaffolds) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, scaf in zip(axes, scaffolds):
        vals = domain_df.loc[domain_df["scaffold"] == scaf, "max_sim_to_outside_scaffold"].dropna()
        ax.hist(vals, bins=15, alpha=0.8)
        ax.set(xlabel="Maximum Tanimoto similarity", ylabel="Frequency", title=f"n = {len(vals)}")
    fig.suptitle("Maximum Tanimoto similarity distributions for difficult-scaffold compounds", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_6_domain_shift.png"), dpi=DPI)
    plt.close(fig)


def main():
    figure1_oof_distributions(load("scaffold_cv_misclassifications.csv"))
    figure2_oof_calibration(load("oof_calibration_bins.csv"))
    figure3_heldout_evaluation(load("heldout_test_predictions.csv"), load("heldout_calibration_bins.csv"))
    figure5_scaffold_performance(load("scaffold_performance_summary.csv"), load("hard_scaffolds.csv"))
    figure6_domain_shift(load("hard_scaffolds_domain_shift_per_compound.csv"))
    print("[DONE] Figures 1, 2, 3, 5, 6 written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
