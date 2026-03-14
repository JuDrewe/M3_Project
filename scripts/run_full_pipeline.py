# =============================================================================
# run_full_pipeline.py
#
# Master script to run the complete M3 antagonist machine-learning workflow.
# Each step corresponds to a standalone script in the /scripts directory.
#
# This script ensures the pipeline runs in the correct order and stops
# immediately if any step fails.
#
# Usage:
#   python scripts/run_full_pipeline.py
# =============================================================================

import os
import sys
import subprocess
from datetime import datetime


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

SCRIPTS = [
    "01_curate_chembl.py",
    "02_train_model.py",
    "03_compute_oof_metrics.py",
    "04_y_randomization.py",
    "05_generate_figures.py",
    "06_umap_projection.py",
    "07_morgan_feature_importance.py",
]


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def run_script(script):
    script_path = os.path.join(HERE, script)

    print("\n" + "=" * 80)
    print(f"RUNNING: {script}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("\nERROR:")
        print(f"{script} failed.")
        sys.exit(result.returncode)

    print(f"\n[OK] {script} finished successfully.")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():

    print("\n")
    print("============================================================")
    print("M3 Antagonist ML Pipeline")
    print("============================================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Start time : {datetime.now()}")
    print("============================================================")

    for script in SCRIPTS:
        run_script(script)

    print("\n============================================================")
    print("Pipeline finished successfully.")
    print(f"End time : {datetime.now()}")
    print("============================================================")


if __name__ == "__main__":
    main()