# =============================================================================
# 03_compute_oof_metrics.py
# Compute OOF and held-out test metrics + calibration bins from saved
# prediction tables.
#
# Outputs: results/tables/{oof,heldout_test}_metrics_summary.csv,
#          combined_metrics_summary.csv,
#          {oof,heldout}_calibration_bins.csv
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, confusion_matrix,
)

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")


def compute_metrics(df, name):
    y, pred, p = df["true_label"].values, df["pred_label"].values, df["p_active"].values
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"dataset": name, "n": len(df), "prevalence": y.mean(),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
            "brier": brier_score_loss(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def calibration_bins(df, name, n_bins=10):
    bins = pd.cut(df["p_active"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    cal = df.groupby(bins, observed=False).agg(
        n=("true_label", "size"), mean_pred=("p_active", "mean"), obs_freq=("true_label", "mean")
    ).reset_index()
    cal["dataset"] = name
    return cal.rename(columns={"p_active": "bin_label"})


def main():
    oof_df = pd.read_csv(os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv"))
    heldout_df = pd.read_csv(os.path.join(TABLES_DIR, "heldout_test_predictions.csv"))

    oof_m = compute_metrics(oof_df, "OOF")
    held_m = compute_metrics(heldout_df, "Held-out test")
    for m in (oof_m, held_m):
        print(f"{m['dataset']}: n={m['n']} prevalence={m['prevalence']:.3f} "
              f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} MCC={m['mcc']:.3f} "
              f"BalAcc={m['bal_acc']:.3f} Brier={m['brier']:.3f} "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    pd.DataFrame([oof_m]).to_csv(os.path.join(TABLES_DIR, "oof_metrics_summary.csv"), index=False)
    pd.DataFrame([held_m]).to_csv(os.path.join(TABLES_DIR, "heldout_test_metrics_summary.csv"), index=False)
    pd.DataFrame([oof_m, held_m]).to_csv(os.path.join(TABLES_DIR, "combined_metrics_summary.csv"), index=False)

    calibration_bins(oof_df, "OOF").to_csv(os.path.join(TABLES_DIR, "oof_calibration_bins.csv"), index=False)
    calibration_bins(heldout_df, "Held-out test").to_csv(
        os.path.join(TABLES_DIR, "heldout_calibration_bins.csv"), index=False)
    print("[DONE]")


if __name__ == "__main__":
    main()
