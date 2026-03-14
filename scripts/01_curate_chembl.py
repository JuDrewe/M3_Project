# =============================================================================
# 01_curate_chembl.py
# ChEMBL M3 activity curation + consensus labeling
#
# Rules:
# - Actives: Ki / IC50 can create active evidence
# - Inactives: Ki / IC50 / EC50 can create inactive evidence
# - Exports curated CSV (optional Excel)
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_p_from_value(value, unit):
    """
    Convert value + unit to p-scale:
      p = -log10(M)

    Supports pM, nM, uM, mM, M.
    Handles ChEMBL artifact '-M' as molar ('M').
    """
    if pd.isna(value):
        return np.nan

    try:
        v = float(value)
    except Exception:
        return np.nan

    if v <= 0:
        return np.nan

    u = str(unit).strip().lower()
    u = u.replace("μ", "u").replace("µ", "u")
    if u == "-m":
        u = "m"

    factors = {
        "pm": 1e-12,
        "nm": 1e-9,
        "um": 1e-6,
        "mm": 1e-3,
        "m":  1.0,
    }

    if u not in factors:
        return np.nan

    return -np.log10(v * factors[u])


def relation_evidence(rel, p, p_active=6.0, p_inactive=5.0):
    """
    Translate relation + p into qualitative evidence.

    '< x'  -> true value could be more potent -> supports activity
    '> x'  -> true value could be less potent -> supports inactivity
    '='    -> evaluate by thresholds
    """
    if pd.isna(p):
        return "missing"

    r = "" if pd.isna(rel) else str(rel).strip()

    if r in ("<", "<="):
        return "active_strong" if p >= p_active else "active_weak"

    if r in (">", ">="):
        return "inactive_strong" if p <= p_inactive else "inactive_weak"

    if p >= p_active:
        return "active"
    if p <= p_inactive:
        return "inactive"

    return "gray"


def first_nonnull(series):
    series = series.dropna()
    return series.iloc[0] if len(series) else pd.NA


# -----------------------------------------------------------------------------
# Consensus labeling
# -----------------------------------------------------------------------------
def label_consensus(
    df: pd.DataFrame,
    id_col: str = "molecule_chembl_id",
    type_col: str = "standard_type",
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
    relation_col: str = "standard_relation",
    p_active: float = 6.0,
    p_inactive: float = 5.0,
    p_active_single: float = 6.5,
    p_inactive_single: float = 5.0,
    min_n: int = 2,
    frac_required: float = 0.75,
    active_types: tuple[str, ...] = ("Ki", "IC50"),
    inactive_types: tuple[str, ...] = ("Ki", "IC50", "EC50"),
) -> pd.DataFrame:
    d = df.copy()
    d[type_col] = d[type_col].astype(str).str.strip()

    keep_types = set(active_types) | set(inactive_types)
    d = d[d[type_col].isin(keep_types)].copy()

    d["p"] = [to_p_from_value(v, u) for v, u in zip(d[value_col], d[unit_col])]

    d["evidence"] = [
        relation_evidence(r, p, p_active=p_active, p_inactive=p_inactive)
        for r, p in zip(d[relation_col], d["p"])
    ]

    d["is_active_type"] = d[type_col].isin(active_types)
    d["is_inactive_type"] = d[type_col].isin(inactive_types)

    d["v_active"] = (
        d["is_active_type"] & d["evidence"].isin(["active", "active_strong"])
    ).astype(float)

    d["v_inactive"] = (
        d["is_inactive_type"] & d["evidence"].isin(["inactive", "inactive_strong"])
    ).astype(float)

    def median_kiki(p_series):
        mask = d.loc[p_series.index, "is_active_type"].values
        vals = p_series.values[mask]
        if vals.size == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanmedian(vals))

    agg = (
        d.groupby(id_col, dropna=False)
        .agg(
            n=("p", lambda x: int(np.sum(~pd.isna(x)))),
            p_median=("p", "median"),
            p_median_kiki=("p", median_kiki),
            frac_active=("v_active", "mean"),
            frac_inactive=("v_inactive", "mean"),
            n_strong_active=("evidence", lambda x: int((pd.Series(x) == "active_strong").sum())),
            n_strong_inactive=("evidence", lambda x: int((pd.Series(x) == "inactive_strong").sum())),
            any_active_vote=("v_active", lambda x: bool(np.nansum(x) > 0)),
        )
        .reset_index()
    )

    agg["p_median"] = agg["p_median"].round(2)
    agg["p_median_kiki"] = agg["p_median_kiki"].round(2)

    def consensus(row):
        if row["n"] == 1:
            if (not pd.isna(row["p_median_kiki"])) and (row["p_median_kiki"] >= p_active_single):
                return "active_single"

            if (not row["any_active_vote"]) and (not pd.isna(row["p_median"])) and (row["p_median"] <= p_inactive_single):
                return "inactive_single"

            return "insufficient"

        if row["n"] < min_n:
            return "insufficient"

        if row["n_strong_active"] > 0 and row["n_strong_inactive"] > 0:
            return "ambiguous"

        if (
            (not pd.isna(row["p_median_kiki"]))
            and (row["p_median_kiki"] >= p_active)
            and (row["frac_active"] >= frac_required)
        ):
            return "active"

        if (
            (not row["any_active_vote"])
            and (not pd.isna(row["p_median"]))
            and (row["p_median"] <= p_inactive)
            and (row["frac_inactive"] >= frac_required)
        ):
            return "inactive"

        return "ambiguous"

    agg["consensus_label"] = agg.apply(consensus, axis=1)
    return agg.sort_values(id_col).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def curate_chembl(input_csv: Path, output_csv: Path, output_xlsx: Path | None = None):
    print(f"[LOAD] {input_csv}")

    df = pd.read_csv(
        input_csv,
        sep=";",
        engine="python",
        header=0,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        on_bad_lines="skip",
    )

    print("Loaded df shape:", df.shape)
    print("Columns (first 20):", df.columns.tolist()[:20])

    colmap = {
        "Molecule ChEMBL ID": "molecule_chembl_id",
        "Standard Type": "standard_type",
        "Standard Relation": "standard_relation",
        "Standard Value": "standard_value",
        "Standard Units": "standard_units",
        "Smiles": "smiles",
        "Molecular Weight": "molecular_weight",
        "Molecular Formula": "molecular_formula",
        "Molecule Name": "molecule_name",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    required = [
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    labels = label_consensus(df)

    print("\nConsensus table shape:", labels.shape)
    print(labels["consensus_label"].value_counts(dropna=False))

    meta_cols = ["molecule_chembl_id"]
    for c in ["smiles", "molecular_weight", "molecular_formula", "molecule_name"]:
        if c in df.columns:
            meta_cols.append(c)

    agg_dict = {}
    if "smiles" in meta_cols:
        agg_dict["smiles"] = first_nonnull
    if "molecular_formula" in meta_cols:
        agg_dict["molecular_formula"] = first_nonnull
    if "molecule_name" in meta_cols:
        agg_dict["molecule_name"] = first_nonnull
    if "molecular_weight" in meta_cols:
        agg_dict["molecular_weight"] = "median"

    meta = (
        df[meta_cols]
        .groupby("molecule_chembl_id", dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    labels = labels.merge(meta, on="molecule_chembl_id", how="left")

    front = ["molecule_chembl_id"]
    for c in ["smiles", "molecular_formula", "molecular_weight", "molecule_name"]:
        if c in labels.columns:
            front.append(c)
    rest = [c for c in labels.columns if c not in front]
    labels = labels[front + rest]

    labels = labels[~labels["consensus_label"].isin(["insufficient", "ambiguous"])].copy()

    print("\nFiltered label counts:")
    print(labels["consensus_label"].value_counts())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_csv, index=False)
    print(f"[SAVED] {output_csv}")

    if output_xlsx is not None:
        output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        labels.to_excel(output_xlsx, index=False)
        print(f"[SAVED] {output_xlsx}")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]

    default_input = project_root / "data" / "raw" / "ChEMBL_M3_raw.csv"
    default_output_csv = (
        project_root / "data" / "raw" / "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv"
    )
    default_output_xlsx = (
        project_root / "data" / "raw" / "ChEMBL_M3_consensus_DB.xlsx"
    )

    parser = argparse.ArgumentParser(
        description="Curate ChEMBL M3 activity data and create consensus labels."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Input raw ChEMBL CSV.")
    parser.add_argument("--output-csv", type=Path, default=default_output_csv, help="Output curated CSV.")
    parser.add_argument("--output-xlsx", type=Path, default=default_output_xlsx, help="Optional output Excel file.")
    return parser.parse_args()


def main():
    args = parse_args()
    curate_chembl(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
    )


if __name__ == "__main__":
    main()