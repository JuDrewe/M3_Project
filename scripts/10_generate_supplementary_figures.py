# =============================================================================
# 07_generate_supplementary_figures.py
# Supplementary Figures S3-S6, built entirely from tables already saved by
# 06_additional_analyses.py (no recomputation).
#
#   S3: scaffold- vs. random-split cross-validation performance (bar chart)
#   S4: Tanimoto similarity histogram, scaffold- vs. random-split
#   S5: applicability-domain-restricted ROC curves (median-similarity split)
#   S6: class composition per CV fold, singles-ablation sensitivity analysis
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
FIG_DIR = os.path.join(_PROJECT, "results", "figures")
DPI = 300


def figure_s3_cv_comparison():
    with open(os.path.join(TABLES_DIR, "additional_analyses_summary.json")) as f:
        summary = json.load(f)
    metrics = ["roc_auc", "pr_auc", "bal_acc", "mcc"]
    labels = ["ROC-AUC", "PR-AUC", "Balanced\naccuracy", "MCC"]
    scaf, rand = summary["scaffold_split"]["inner_cv"], summary["random_split"]["inner_cv"]
    scaf_m, scaf_s = zip(*(scaf[m] for m in metrics))
    rand_m, rand_s = zip(*(rand[m] for m in metrics))

    x, w = np.arange(len(metrics)), 0.35
    plt.figure(figsize=(8, 5.5))
    plt.bar(x - w/2, scaf_m, w, yerr=scaf_s, capsize=4, label="Scaffold-grouped CV", color="#2471a3")
    plt.bar(x + w/2, rand_m, w, yerr=rand_s, capsize=4, label="Random (stratified) CV", color="#c0392b")
    plt.xticks(x, labels); plt.ylim(0, 1.05)
    plt.ylabel("Score (10-fold CV, mean \u00b1 SD)")
    plt.title("Scaffold-grouped vs. random cross-validation:\nperformance comparison on identical dev-set")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_s3_scaffold_vs_random_cv.png"), dpi=DPI)
    plt.close()


def figure_s4_similarity_histogram():
    sim_s = pd.read_csv(os.path.join(TABLES_DIR, "similarity_scaffold_split_test_to_dev.csv"))["scaffold_split_max_sim"].dropna()
    sim_r = pd.read_csv(os.path.join(TABLES_DIR, "similarity_random_split_test_to_dev.csv"))["random_split_max_sim"].dropna()
    bins = np.linspace(0, 1, 41)
    plt.figure(figsize=(7, 5))
    plt.hist(sim_s, bins=bins, alpha=0.6, color="#2471a3",
             label=f"Scaffold split (median={sim_s.median():.2f}, n={len(sim_s)})")
    plt.hist(sim_r, bins=bins, alpha=0.6, color="#c0392b",
             label=f"Random split (median={sim_r.median():.2f}, n={len(sim_r)})")
    plt.axvline(sim_s.median(), color="#2471a3", linestyle="--", linewidth=1.5)
    plt.axvline(sim_r.median(), color="#c0392b", linestyle="--", linewidth=1.5)
    plt.xlabel("Maximum Tanimoto similarity of test compound to development set")
    plt.ylabel("Frequency")
    plt.title("Test-to-development-set similarity:\nscaffold split vs. random split")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_s4_tanimoto_similarity.png"), dpi=DPI)
    plt.close()


def figure_s5_ad_restricted_roc():
    ad_df = pd.read_csv(os.path.join(TABLES_DIR, "ad_restricted_heldout_predictions.csv"))
    median_sim = float(ad_df["max_sim_to_dev"].median())
    in_ad = ad_df[ad_df["max_sim_to_dev"] >= median_sim]
    out_ad = ad_df[ad_df["max_sim_to_dev"] < median_sim]
    fpr_in, tpr_in, _ = roc_curve(in_ad["true_label"], in_ad["p_active"])
    fpr_out, tpr_out, _ = roc_curve(out_ad["true_label"], out_ad["p_active"])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_in, tpr_in, linewidth=2, color="#2471a3",
             label=f"In-AD (sim \u2265 {median_sim:.2f}, n={len(in_ad)}, AUC={auc(fpr_in, tpr_in):.3f})")
    plt.plot(fpr_out, tpr_out, linewidth=2, color="#c0392b",
             label=f"Out-of-AD (sim < {median_sim:.2f}, n={len(out_ad)}, AUC={auc(fpr_out, tpr_out):.3f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Held-out test performance stratified\nby applicability domain (median similarity split)")
    plt.legend(fontsize=9, loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_s5_ad_restricted_roc.png"), dpi=DPI)
    plt.close()


def figure_s6_singles_ablation():
    fold_df = pd.read_csv(os.path.join(TABLES_DIR, "singles_ablation_fold_composition.csv"))
    zero_inactive = fold_df.loc[fold_df["n_inactive"] == 0, "fold"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = fold_df["fold"].values
    ax.bar(x, fold_df["n_active"], label="Active (n)", color="#c0392b", alpha=0.85)
    ax.bar(x, fold_df["n_inactive"], bottom=fold_df["n_active"], label="Inactive (n)", color="#2471a3", alpha=0.85)
    for f in zero_inactive:
        y = fold_df.loc[fold_df["fold"] == f, "n_active"].values[0]
        ax.annotate("0 inactive", xy=(f, y), xytext=(f, y + 3), ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xticks(x)
    ax.set(xlabel="Cross-validation fold", ylabel="Number of compounds in test fold",
           title="Class composition per CV fold when excluding\nsingle-measurement compounds (n=303, active/inactive only)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_s6_singles_ablation.png"), dpi=DPI)
    plt.close()
    print(f"[STRICT] folds with 0 inactive test compounds: {zero_inactive}")


def main():
    figure_s3_cv_comparison()
    figure_s4_similarity_histogram()
    figure_s5_ad_restricted_roc()
    figure_s6_singles_ablation()
    print("[DONE]")


if __name__ == "__main__":
    main()
