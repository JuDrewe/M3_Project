# =============================================================================
# 03_compute_oof_metrics.py
# Compute OOF and held-out test metrics from saved prediction tables
#
# Inputs:
#   results/tables/scaffold_cv_misclassifications.csv
#   results/tables/heldout_test_predictions.csv
#
# Outputs:
#   results/tables/oof_metrics_summary.csv
#   results/tables/heldout_test_metrics_summary.csv
#   results/tables/combined_metrics_summary.csv
#   results/tables/oof_calibration_bins.csv
#   results/tables/heldout_calibration_bins.csv
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


# =============================================================================
# Configuration
# =============================================================================

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

OOF_PATH = os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv")
HELDOUT_PATH = os.path.join(TABLES_DIR, "heldout_test_predictions.csv")

OOF_SUMMARY_PATH = os.path.join(TABLES_DIR, "oof_metrics_summary.csv")
HELDOUT_SUMMARY_PATH = os.path.join(TABLES_DIR, "heldout_test_metrics_summary.csv")
COMBINED_SUMMARY_PATH = os.path.join(TABLES_DIR, "combined_metrics_summary.csv")

OOF_CALIB_PATH = os.path.join(TABLES_DIR, "oof_calibration_bins.csv")
HELDOUT_CALIB_PATH = os.path.join(TABLES_DIR, "heldout_calibration_bins.csv")

OOF_JSON_PATH = os.path.join(TABLES_DIR, "oof_metrics_summary.json")
HELDOUT_JSON_PATH = os.path.join(TABLES_DIR, "heldout_test_metrics_summary.json")


# =============================================================================
# Helpers
# =============================================================================

def load_predictions(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

    df = pd.read_csv(path)

    required = {"true_label", "pred_label", "p_active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")

    df = df.copy()
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df["p_active"] = df["p_active"].astype(float)

    return df


def compute_metrics(df: pd.DataFrame, dataset_name: str) -> dict:
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values
    p = df["p_active"].values

    roc = roc_auc_score(y_true, p)
    pr = average_precision_score(y_true, p)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, p)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    out = {
        "dataset": dataset_name,
        "n": int(len(df)),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "mcc": float(mcc),
        "bal_acc": float(bal_acc),
        "brier": float(brier),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return out


def calibration_bins(
    df: pd.DataFrame,
    n_bins: int = 10,
    dataset_name: str = "OOF",
) -> pd.DataFrame:
    """
    Fixed-width calibration bins between 0 and 1.
    """
    tmp = df.copy()
    tmp["bin"] = pd.cut(
        tmp["p_active"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )

    cal = (
        tmp.groupby("bin", observed=False)
        .agg(
            n=("true_label", "size"),
            mean_pred=("p_active", "mean"),
            obs_freq=("true_label", "mean"),
        )
        .reset_index()
    )

    cal["dataset"] = dataset_name
    cal["bin_label"] = cal["bin"].astype(str)
    cal = cal.drop(columns=["bin"])

    return cal


def save_metrics_dict(metrics: dict, csv_path: str, json_path: str):
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics: dict):
    print(f"\n=== {metrics['dataset']} ===")
    print(f"n           : {metrics['n']}")
    print(f"prevalence  : {metrics['prevalence']:.3f}")
    print(f"ROC-AUC     : {metrics['roc_auc']:.3f}")
    print(f"PR-AUC      : {metrics['pr_auc']:.3f}")
    print(f"MCC         : {metrics['mcc']:.3f}")
    print(f"BalAcc      : {metrics['bal_acc']:.3f}")
    print(f"Brier       : {metrics['brier']:.3f}")
    print("Confusion   :")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[LOAD] OOF predictions: {OOF_PATH}")
    oof_df = load_predictions(OOF_PATH, "OOF")

    print(f"[LOAD] Held-out predictions: {HELDOUT_PATH}")
    heldout_df = load_predictions(HELDOUT_PATH, "Held-out")

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    oof_metrics = compute_metrics(oof_df, dataset_name="OOF")
    heldout_metrics = compute_metrics(heldout_df, dataset_name="Held-out test")

    print_metrics(oof_metrics)
    print_metrics(heldout_metrics)

    # -------------------------------------------------------------------------
    # Save summaries
    # -------------------------------------------------------------------------
    save_metrics_dict(oof_metrics, OOF_SUMMARY_PATH, OOF_JSON_PATH)
    print(f"[SAVED] {OOF_SUMMARY_PATH}")
    print(f"[SAVED] {OOF_JSON_PATH}")

    save_metrics_dict(heldout_metrics, HELDOUT_SUMMARY_PATH, HELDOUT_JSON_PATH)
    print(f"[SAVED] {HELDOUT_SUMMARY_PATH}")
    print(f"[SAVED] {HELDOUT_JSON_PATH}")

    combined = pd.DataFrame([oof_metrics, heldout_metrics])
    combined.to_csv(COMBINED_SUMMARY_PATH, index=False)
    print(f"[SAVED] {COMBINED_SUMMARY_PATH}")

    print("\n[SUMMARY TABLE]")
    print(combined.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save calibration bins
    # -------------------------------------------------------------------------
    oof_cal = calibration_bins(oof_df, n_bins=10, dataset_name="OOF")
    heldout_cal = calibration_bins(heldout_df, n_bins=10, dataset_name="Held-out test")

    oof_cal.to_csv(OOF_CALIB_PATH, index=False)
    heldout_cal.to_csv(HELDOUT_CALIB_PATH, index=False)

    print(f"[SAVED] {OOF_CALIB_PATH}")
    print(f"[SAVED] {HELDOUT_CALIB_PATH}")

    print("\n[DONE] Metric summaries and calibration tables written.")


if __name__ == "__main__":
    main()