# =============================================================================
# 07_morgan_feature_importance.py
# Extract and visualise the top Morgan fingerprint bits from the trained
# logistic regression model.
#
# Outputs:
#   results/tables/figure_6_7_top_morgan_bits.csv
#   results/figures/figure_6_7_morgan_feature_importance.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


# =============================================================================
# Configuration
# =============================================================================

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

MODEL_PATH  = os.path.join(_PROJECT, "models", "m3_fp_physchem_scaffoldcv.joblib")
DATA_PATH   = os.path.join(
    _PROJECT, "data", "raw",
    "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv",
)
TABLES_DIR  = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TOP_N            = 20
FP_RADIUS        = 2
FP_NBITS         = 2048
FP_USE_CHIRALITY = True

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}

DPI = 300


# =============================================================================
# Helpers
# =============================================================================

def env_to_smiles(mol, center_atom: int, radius: int) -> str:
    """Return SMILES for the Morgan atom environment."""
    if radius == 0:
        sym = mol.GetAtomWithIdx(center_atom).GetSymbol()
        return f"[{sym}]"

    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
    if not env:
        return mol.GetAtomWithIdx(center_atom).GetSymbol()

    atoms_in_env = set()
    for bond_idx in env:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms_in_env.add(bond.GetBeginAtomIdx())
        atoms_in_env.add(bond.GetEndAtomIdx())

    try:
        smi = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(atoms_in_env),
            bondsToUse=list(env),
            canonical=True,
        )
        return smi if smi else "?"
    except Exception:
        return "?"


def env_highlight(mol, center_atom: int, radius: int):
    """Return atoms and bonds to highlight for a Morgan environment."""
    if radius == 0:
        return [center_atom], []

    env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom))
    if not env_bonds:
        return [center_atom], []

    atoms_in_env = set()
    for bond_idx in env_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms_in_env.add(bond.GetBeginAtomIdx())
        atoms_in_env.add(bond.GetEndAtomIdx())

    return list(atoms_in_env), env_bonds


# =============================================================================
# Main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1. Load model and extract Morgan coefficients
    # -------------------------------------------------------------------------
    print(f"[LOAD MODEL] {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipe"]
    clf = pipe.named_steps["clf"]

    coef = clf.coef_[0]
    fp_coef = coef[:FP_NBITS]

    print(f"  Total features : {len(coef)}")
    print(f"  Morgan bits    : {FP_NBITS}")
    print(f"  Extra features : {len(coef) - FP_NBITS}")

    top_bits = np.argsort(np.abs(fp_coef))[::-1][:TOP_N]

    # -------------------------------------------------------------------------
    # 2. Load curated compounds and recover representative environments
    # -------------------------------------------------------------------------
    print(f"\n[LOAD DATA] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""].copy()

    print(f"  Compounds available: {len(df)}")

    top_bits_set = set(int(b) for b in top_bits)
    bit_to_example = {}

    print("[SCAN] Searching for representative Morgan environments ...")
    for smi in df["smiles"]:
        if len(bit_to_example) == len(top_bits_set):
            break

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            FP_RADIUS,
            FP_NBITS,
            useChirality=FP_USE_CHIRALITY,
            bitInfo=bit_info,
        )

        for bit_id, info_list in bit_info.items():
            if bit_id in top_bits_set and bit_id not in bit_to_example:
                center_atom, radius = info_list[0]
                bit_to_example[bit_id] = (mol, center_atom, radius)

    print(f"  Found examples for {len(bit_to_example)}/{len(top_bits_set)} bits")

    # -------------------------------------------------------------------------
    # 3. Save output table
    # -------------------------------------------------------------------------
    rows = []
    for bit_id in top_bits:
        bit_id = int(bit_id)
        coef_val = float(fp_coef[bit_id])
        direction = "positive" if coef_val > 0 else "negative"

        frag_smiles = "?"
        if bit_id in bit_to_example:
            mol_ex, center, radius = bit_to_example[bit_id]
            frag_smiles = env_to_smiles(mol_ex, center, radius)

        rows.append({
            "bit_id": bit_id,
            "coefficient": round(coef_val, 6),
            "direction": direction,
            "example_fragment_smiles": frag_smiles,
        })

    out_df = pd.DataFrame(rows)

    csv_path = os.path.join(TABLES_DIR, "figure_6_7_top_morgan_bits.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
    print(out_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Make combined figure
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(22, 10))
    outer_gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1, 2.8],
        wspace=0.08,
        left=0.04, right=0.98, top=0.91, bottom=0.06,
    )

    # Left panel: coefficient bar chart
    ax_bar = fig.add_subplot(outer_gs[0])

    bit_labels = [f"bit {r['bit_id']}" for r in rows]
    coef_vals = [r["coefficient"] for r in rows]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in coef_vals]
    y_pos = np.arange(len(bit_labels))

    ax_bar.barh(y_pos, coef_vals, color=colors, edgecolor="white", height=0.72)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bit_labels, fontsize=8)
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Logistic regression coefficient", fontsize=9)
    ax_bar.set_title(
        f"Top {TOP_N} Morgan fingerprint bits by |coefficient|",
        fontsize=10, pad=8,
    )

    pos_patch = mpatches.Patch(color="#c0392b", label="Pro-active")
    neg_patch = mpatches.Patch(color="#2471a3", label="Pro-inactive")
    ax_bar.legend(handles=[pos_patch, neg_patch], fontsize=7.5, loc="lower right")
    ax_bar.tick_params(axis="x", labelsize=8)

    # Right panel: fragment examples
    n_rows, n_cols = 4, 5
    inner_gs = GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=outer_gs[1],
        hspace=0.55, wspace=0.15,
    )

    plt.style.use("default")

    for idx, row in enumerate(rows):
        r_i = idx // n_cols
        c_i = idx % n_cols
        ax = fig.add_subplot(inner_gs[r_i, c_i])
        ax.axis("off")

        bit_id = row["bit_id"]
        coef_val = row["coefficient"]

        if bit_id in bit_to_example:
            mol_ex, center, radius = bit_to_example[bit_id]
            hi_atoms, hi_bonds = env_highlight(mol_ex, center, radius)
            try:
                img = Draw.MolToImage(
                    mol_ex,
                    size=(180, 130),
                    highlightAtoms=hi_atoms,
                    highlightBonds=hi_bonds,
                )
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey")

        title_color = "#c0392b" if coef_val > 0 else "#2471a3"
        ax.set_title(
            f"bit {bit_id}  {coef_val:+.3f}",
            fontsize=7, color=title_color, pad=3,
        )

    fig.suptitle(
        "Figures 6–7. Most influential Morgan fingerprint bits and representative molecular environments\n"
        "Red = pro-active  |  Blue = pro-inactive",
        fontsize=11, y=0.97,
    )

    fig_path = os.path.join(FIGURES_DIR, "figure_6_7_morgan_feature_importance.png")
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {fig_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()