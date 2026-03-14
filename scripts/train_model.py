# =============================================================================
# train_model.py
# M3 Muscarinic Receptor Antagonist — Scaffold-CV Training Pipeline
#
# Sections:
#   1.  Imports & configuration
#   2.  Helper functions (features, scaffold, pipeline)
#   3.  Load & filter data
#   4.  Scaffold computation + dev / held-out test split
#   5.  Scaffold-CV evaluation (FP-only vs FP+PhysChem)
#   6.  Fit final model on dev set
#   7.  OOF error analysis + threshold selection + per-scaffold performance
#   8.  Domain shift quantification for hard scaffolds
#   9.  Held-out test evaluation
#   10. Inference helper
#
# Outputs:
#   models/
#     m3_fp_physchem_scaffoldcv.joblib
#   results/tables/
#     scaffold_cv_misclassifications.csv
#     scaffold_performance_summary.csv
#     hard_scaffolds.csv
#     hard_scaffolds_domain_shift_per_compound.csv
#     hard_scaffolds_domain_shift_summary.csv
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -- Input data ---------------------------------------------------------------
# Update DATA_PATH to point to your local copy of the input CSV before running.
DATA_PATH = os.path.join(
    _PROJECT, "data", "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)

# -- Output directories -------------------------------------------------------
MODELS_DIR = os.path.join(_PROJECT, "models")
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

SAVE_MODEL_PATH   = os.path.join(MODELS_DIR, "m3_fp_physchem_scaffoldcv.joblib")
MISCLASS_PATH     = os.path.join(TABLES_DIR, "scaffold_cv_misclassifications.csv")
SCAF_SUM_PATH     = os.path.join(TABLES_DIR, "scaffold_performance_summary.csv")
HARD_SCAF_PATH    = os.path.join(TABLES_DIR, "hard_scaffolds.csv")
DOMAIN_SHIFT_PATH = os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_per_compound.csv")
DOMAIN_SUM_PATH   = os.path.join(TABLES_DIR, "hard_scaffolds_domain_shift_summary.csv")

# -- Labels -------------------------------------------------------------------
POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}

# Confidence-based sample weights — reflect label reliability only.
# These are NOT a class-balance mechanism. Add explicit class-balance
# weights or use class_weight in the classifier if imbalance correction
# is needed.
WEIGHTS = {
    "active":          1.0,
    "inactive":        1.0,
    "active_single":   0.5,
    "inactive_single": 0.7,
}

# -- Fingerprint --------------------------------------------------------------
FP_RADIUS        = 2
FP_NBITS         = 2048
FP_USE_CHIRALITY = True
FP_USE_FEATURES  = False    # True => FCFP-like (feature invariants)

# -- Cross-validation ---------------------------------------------------------
N_SPLITS           = 10     # inner scaffold-CV folds (model selection + OOF)
OUTER_N_SPLITS     = 5      # outer split: held-out test set (~20 %)
CV_RANDOM_STATE    = 42
OUTER_RANDOM_STATE = 0

# -- Model selection ----------------------------------------------------------
FINAL_USE_PHYSCHEM = True   # True => FP+PhysChem, False => FP-only

# -- Hard-scaffold thresholds -------------------------------------------------
HARD_MIN_N   = 10
HARD_MCC_MAX = 0.5


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def smiles_to_scaffold(smi: str) -> str:
    """Return canonical Murcko scaffold SMILES, or '' on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def _make_morgan_generator():
    atom_inv_gen = None
    if FP_USE_FEATURES:
        atom_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_NBITS,
        includeChirality=FP_USE_CHIRALITY,
        atomInvariantsGenerator=atom_inv_gen,
    )


def morgan_fp_matrix(smiles_list) -> np.ndarray:
    """Dense Morgan FP matrix (n_samples, FP_NBITS) float32."""
    smiles_list = np.asarray(smiles_list, dtype=object)
    gen = _make_morgan_generator()
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


def physchem_matrix(smiles_list) -> np.ndarray:
    """6 physicochemical descriptors: MolWt, LogP, HBD, HBA, TPSA, RotBonds.
    Missing values imputed with per-column median (computed within the batch).
    """
    smiles_list = np.asarray(smiles_list, dtype=object)
    X = np.zeros((len(smiles_list), 6), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            X[i, :] = np.nan
            continue
        X[i, 0] = Descriptors.MolWt(mol)
        X[i, 1] = Descriptors.MolLogP(mol)
        X[i, 2] = Descriptors.NumHDonors(mol)
        X[i, 3] = Descriptors.NumHAcceptors(mol)
        X[i, 4] = Descriptors.TPSA(mol)
        X[i, 5] = Descriptors.NumRotatableBonds(mol)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
    return X


def build_X(smiles_list, use_physchem: bool) -> np.ndarray:
    X_fp = morgan_fp_matrix(smiles_list)
    if not use_physchem:
        return X_fp
    return np.hstack([X_fp, physchem_matrix(smiles_list)]).astype(np.float32)


def make_pipe() -> Pipeline:
    # StandardScaler omitted: Morgan FP bits are already on a uniform binary
    # scale and do not benefit from variance scaling. Add a scaler here if
    # PhysChem features are used and scaling proves beneficial.
    # class_weight=None: class-balance correction is not applied here; use
    # explicit sample weights (WEIGHTS dict) or add class_weight if needed.
    return Pipeline([
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            class_weight=None,
        )),
    ])


def compute_fp_objects(smiles_list) -> list:
    """Return list of RDKit ExplicitBitVect (None for unparseable SMILES)."""
    gen = _make_morgan_generator()
    fps = []
    for smi in np.asarray(smiles_list, dtype=object):
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


# =============================================================================
# 3. LOAD & FILTER DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"molecule_chembl_id", "smiles", "consensus_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""].copy()

    # TODO: add canonical-SMILES deduplication here to remove duplicate
    # structures and resolve conflicting labels before any modelling.
    # Example approach:
    #   df["canonical_smiles"] = df["smiles"].apply(canonicalize)
    #   df = resolve_conflicts(df, key="canonical_smiles", label_col="consensus_label")
    #   df = df.drop_duplicates(subset=["canonical_smiles"])

    df["y"] = df["consensus_label"].isin(POS_LABELS).astype(int)
    df["w"] = df["consensus_label"].map(WEIGHTS).astype(float)
    return df


# =============================================================================
# 4. SCAFFOLD-CV EVALUATION
# =============================================================================

def eval_mode(
    name: str,
    use_physchem: bool,
    cv,
    X_smiles_dev, y_dev, w_dev, groups_dev,
) -> dict:
    """Run scaffold-stratified CV; print per-fold and summary metrics."""
    metrics = {"roc_auc": [], "pr_auc": [], "bal_acc": [], "mcc": []}
    print(f"\n=== {name} ===")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_smiles_dev, y_dev, groups=groups_dev), start=1
    ):
        X_train = build_X(X_smiles_dev[train_idx], use_physchem=use_physchem)
        X_test  = build_X(X_smiles_dev[test_idx],  use_physchem=use_physchem)
        y_train, y_test = y_dev[train_idx], y_dev[test_idx]
        w_train = w_dev[train_idx]

        pipe = make_pipe()
        pipe.fit(X_train, y_train, clf__sample_weight=w_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_test, proba)
        pr  = average_precision_score(y_test, proba)
        bal = balanced_accuracy_score(y_test, pred)
        mcc = matthews_corrcoef(y_test, pred)

        metrics["roc_auc"].append(roc)
        metrics["pr_auc"].append(pr)
        metrics["bal_acc"].append(bal)
        metrics["mcc"].append(mcc)

        print(
            f"  [Fold {fold:02d}] ROC-AUC={roc:.3f} | PR-AUC={pr:.3f} | "
            f"BalAcc={bal:.3f} | MCC={mcc:.3f} | "
            f"n_test={len(test_idx)} | pos_test={int(y_test.sum())}"
        )

    print(f"\n  --- {name} summary ---")
    for k, vals in metrics.items():
        arr = np.asarray(vals)
        print(f"  {k}: mean={arr.mean():.3f}  std={arr.std():.3f}")

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"[DATA] Loading: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"[DATA] Loaded & filtered: {df.shape}")
    print(df["consensus_label"].value_counts().to_string())

    # -------------------------------------------------------------------------
    # Scaffold computation
    # -------------------------------------------------------------------------
    print("\n[SCAFFOLDS] Computing Murcko scaffolds ...")
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    bad = df["scaffold"].eq("")
    if bad.any():
        print(f"  Dropping {bad.sum()} rows with invalid SMILES/scaffold.")
        df = df[~bad].copy()

    X_smiles = df["smiles"].values
    y        = df["y"].values.astype(int)
    w        = df["w"].values.astype(float)
    groups   = df["scaffold"].values

    # -------------------------------------------------------------------------
    # Dev / held-out test split (scaffold-stratified, ~20 % test)
    # The held-out test set (_test_idx) is never used for training or threshold
    # selection — it is reserved for final reported performance only.
    # -------------------------------------------------------------------------
    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_N_SPLITS, shuffle=True, random_state=OUTER_RANDOM_STATE
    )
    _dev_idx, _test_idx = next(outer_cv.split(X_smiles, y, groups=groups))

    X_smiles_dev,  X_smiles_test = X_smiles[_dev_idx], X_smiles[_test_idx]
    y_dev,         y_test_held   = y[_dev_idx],         y[_test_idx]
    w_dev,         groups_dev    = w[_dev_idx],          groups[_dev_idx]

    print(f"\n[SPLIT] dev={len(_dev_idx)}  held-out test={len(_test_idx)}")
    print(f"[PREVALENCE] dev={y_dev.mean():.3f}  test={y_test_held.mean():.3f}")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    # -------------------------------------------------------------------------
    # Section 5 — Scaffold-CV evaluation: FP-only vs FP+PhysChem
    # -------------------------------------------------------------------------
    m1 = eval_mode(
        "A) FP-only", use_physchem=False,
        cv=cv,
        X_smiles_dev=X_smiles_dev, y_dev=y_dev, w_dev=w_dev, groups_dev=groups_dev,
    )
    m2 = eval_mode(
        "B) FP + PhysChem", use_physchem=True,
        cv=cv,
        X_smiles_dev=X_smiles_dev, y_dev=y_dev, w_dev=w_dev, groups_dev=groups_dev,
    )

    # -------------------------------------------------------------------------
    # Section 6 — Fit final model on dev set only
    # The held-out test set remains untouched at this point.
    # -------------------------------------------------------------------------
    print("\n[MODEL] Fitting final model on dev set ...")
    X_dev_all  = build_X(X_smiles_dev, use_physchem=FINAL_USE_PHYSCHEM)
    final_pipe = make_pipe()
    final_pipe.fit(X_dev_all, y_dev, clf__sample_weight=w_dev)

    # -------------------------------------------------------------------------
    # Section 7 — OOF error analysis + OOF threshold selection
    # Collect predictions from the same 10-fold CV used for evaluation.
    # Best threshold is selected by maximising MCC on pooled OOF probabilities,
    # so it never touches the held-out test set.
    # -------------------------------------------------------------------------
    print("\n[OOF] Collecting out-of-fold predictions ...")
    oof_records = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_smiles_dev, y_dev, groups=groups_dev), start=1
    ):
        X_train = build_X(X_smiles_dev[train_idx], use_physchem=FINAL_USE_PHYSCHEM)
        X_test  = build_X(X_smiles_dev[test_idx],  use_physchem=FINAL_USE_PHYSCHEM)
        y_train, y_test = y_dev[train_idx], y_dev[test_idx]
        w_train = w_dev[train_idx]

        fold_pipe = make_pipe()
        fold_pipe.fit(X_train, y_train, clf__sample_weight=w_train)

        proba = fold_pipe.predict_proba(X_test)[:, 1]

        for i, idx in enumerate(test_idx):
            orig_idx = int(_dev_idx[idx])   # map dev-local index → original df row
            oof_records.append({
                "fold":       fold,
                "orig_idx":   orig_idx,
                "p_active":   float(proba[i]),
                "true_label": int(y_test[i]),
            })

    # OOF threshold selection: maximise MCC over pooled OOF probabilities
    oof_proba = np.array([r["p_active"]   for r in oof_records])
    oof_true  = np.array([r["true_label"] for r in oof_records])

    thresholds = np.linspace(0.05, 0.95, 181)
    mccs       = [
        matthews_corrcoef(oof_true, (oof_proba >= t).astype(int))
        for t in thresholds
    ]
    best_threshold = float(thresholds[np.argmax(mccs)])
    best_mcc       = float(np.max(mccs))
    print(
        f"[THRESHOLD] Best OOF threshold by MCC: "
        f"{best_threshold:.2f}  →  OOF MCC={best_mcc:.3f}"
    )

    # Save model bundle now that threshold is known
    model_bundle = {
        "pipe":             final_pipe,
        "use_physchem":     FINAL_USE_PHYSCHEM,
        "fp_radius":        FP_RADIUS,
        "fp_nbits":         FP_NBITS,
        "fp_use_chirality": FP_USE_CHIRALITY,
        "fp_use_features":  FP_USE_FEATURES,
        "physchem_dim":     6,
        "threshold":        best_threshold,
    }
    joblib.dump(model_bundle, SAVE_MODEL_PATH)
    print(f"[SAVED] {SAVE_MODEL_PATH}")

    # Build per-compound OOF error table using the selected threshold
    rows = []
    for r in oof_records:
        orig_idx = r["orig_idx"]
        pred_i   = int(r["p_active"] >= best_threshold)
        true_i   = r["true_label"]
        err = "OK"
        if pred_i == 1 and true_i == 0:
            err = "FP"
        elif pred_i == 0 and true_i == 1:
            err = "FN"
        rows.append({
            "fold":               r["fold"],
            "row_index":          orig_idx,
            "molecule_chembl_id": df.iloc[orig_idx]["molecule_chembl_id"],
            "consensus_label":    df.iloc[orig_idx]["consensus_label"],
            "smiles":             df.iloc[orig_idx]["smiles"],
            "scaffold":           df.iloc[orig_idx]["scaffold"],
            "true_label":         true_i,
            "pred_label":         pred_i,
            "p_active":           r["p_active"],
            "error_type":         err,
        })

    err_df = pd.DataFrame(rows)
    err_df.to_csv(MISCLASS_PATH, index=False)
    print(f"[SAVED] {MISCLASS_PATH}")
    print(err_df["error_type"].value_counts().to_string())

    # Per-scaffold performance summary
    scaf_stats = []
    for scaf, g in err_df.groupby("scaffold"):
        y_true = g["true_label"].values
        y_pred = g["pred_label"].values
        if len(np.unique(y_true)) < 2:      # MCC undefined with single class
            continue
        scaf_stats.append({
            "scaffold": scaf,
            "n":        int(len(g)),
            "mcc":      float(matthews_corrcoef(y_true, y_pred)),
            "bal_acc":  float(balanced_accuracy_score(y_true, y_pred)),
            "n_errors": int((g["error_type"] != "OK").sum()),
            "pos_frac": float(np.mean(y_true)),
        })

    scaf_df = pd.DataFrame(scaf_stats).sort_values(
        ["mcc", "n"], ascending=[True, False]
    )
    scaf_df.to_csv(SCAF_SUM_PATH, index=False)
    print(f"[SAVED] {SCAF_SUM_PATH}")

    hard_scaffolds = scaf_df.query(
        f"n >= {HARD_MIN_N} and mcc < {HARD_MCC_MAX}"
    ).copy()
    hard_scaffolds.to_csv(HARD_SCAF_PATH, index=False)
    print(f"[SAVED] {HARD_SCAF_PATH}")
    print(
        f"  Hard scaffolds (n>={HARD_MIN_N} & MCC<{HARD_MCC_MAX}): "
        f"{len(hard_scaffolds)}"
    )

    # -------------------------------------------------------------------------
    # Section 8 — Domain shift quantification for hard scaffolds
    # -------------------------------------------------------------------------
    hard_scaf_set = set(hard_scaffolds["scaffold"].tolist())

    if not hard_scaf_set:
        print("\n[DOMAIN SHIFT] No hard scaffolds found — skipping.")
    else:
        print(f"\n[DOMAIN SHIFT] Analysing {len(hard_scaf_set)} hard scaffolds ...")
        all_fps   = compute_fp_objects(df["smiles"].values)
        valid_idx = np.array(
            [i for i, fp in enumerate(all_fps) if fp is not None], dtype=int
        )

        domain_rows = []
        for scaf in hard_scaf_set:
            mask_scaf = df["scaffold"].values == scaf
            scaf_idx  = valid_idx[mask_scaf[valid_idx]]
            train_idx = valid_idx[~mask_scaf[valid_idx]]
            train_fps = [all_fps[i] for i in train_idx]

            for idx in scaf_idx:
                sims = DataStructs.BulkTanimotoSimilarity(all_fps[idx], train_fps)
                domain_rows.append({
                    "scaffold":                    scaf,
                    "row_index":                   int(idx),
                    "molecule_chembl_id":           df.iloc[idx]["molecule_chembl_id"],
                    "consensus_label":              df.iloc[idx]["consensus_label"],
                    "true_label":                  int(df.iloc[idx]["y"]),
                    "max_sim_to_outside_scaffold":  float(np.max(sims))            if sims else np.nan,
                    "mean_sim_to_outside_scaffold": float(np.mean(sims))           if sims else np.nan,
                    "p95_sim_to_outside_scaffold":  float(np.percentile(sims, 95)) if sims else np.nan,
                })

        domain_df = pd.DataFrame(domain_rows)
        domain_df.to_csv(DOMAIN_SHIFT_PATH, index=False)
        print(f"[SAVED] {DOMAIN_SHIFT_PATH}")

        domain_sum = (
            domain_df.groupby("scaffold")
            .agg(
                n=("row_index",                      "count"),
                maxSim_mean=("max_sim_to_outside_scaffold",  "mean"),
                maxSim_min=("max_sim_to_outside_scaffold",   "min"),
                p95Sim_mean=("p95_sim_to_outside_scaffold",  "mean"),
                meanSim_mean=("mean_sim_to_outside_scaffold","mean"),
            )
            .reset_index()
            .sort_values("maxSim_mean")
        )
        domain_sum.to_csv(DOMAIN_SUM_PATH, index=False)
        print(f"[SAVED] {DOMAIN_SUM_PATH}")
        print(domain_sum.to_string(index=False))

    # -------------------------------------------------------------------------
    # Section 9 — Final held-out test evaluation
    # Run once only. These numbers are the ones reported in the paper.
    # -------------------------------------------------------------------------
    print("\n[TEST] Evaluating on held-out test set ...")
    X_test_final = build_X(X_smiles_test, use_physchem=FINAL_USE_PHYSCHEM)
    proba_test   = final_pipe.predict_proba(X_test_final)[:, 1]
    pred_test    = (proba_test >= best_threshold).astype(int)

    print("=== HELD-OUT TEST SET PERFORMANCE ===")
    print(f"  Prevalence : {y_test_held.mean():.3f}")
    print(f"  ROC-AUC    : {roc_auc_score(y_test_held, proba_test):.3f}")
    print(f"  PR-AUC     : {average_precision_score(y_test_held, proba_test):.3f}")
    print(f"  MCC        : {matthews_corrcoef(y_test_held, pred_test):.3f}")
    print(f"  BalAcc     : {balanced_accuracy_score(y_test_held, pred_test):.3f}")

    print("\n[DONE] All outputs written.")
    print(f"  models/  → {MODELS_DIR}")
    print(f"  tables/  → {TABLES_DIR}")


# =============================================================================
# 10. INFERENCE HELPER
# =============================================================================

def load_model_and_predict(smiles_list, model_path: str = SAVE_MODEL_PATH):
    """Load a saved model bundle and predict activity for a list of SMILES.

    Returns
    -------
    proba : np.ndarray  shape (n,)  — predicted probability of activity
    pred  : np.ndarray  shape (n,)  — binary prediction at the OOF-selected threshold
    """
    bundle       = joblib.load(model_path)
    pipe         = bundle["pipe"]
    use_physchem = bundle["use_physchem"]
    threshold    = bundle["threshold"]

    X     = build_X(np.asarray(smiles_list, dtype=object), use_physchem=use_physchem)
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return proba, pred


if __name__ == "__main__":
    main()
