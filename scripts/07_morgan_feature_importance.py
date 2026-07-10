# =============================================================================
# 07_morgan_feature_importance.py
# Identifies the top-20 Morgan fingerprint bits by logistic-regression
# coefficient magnitude, finds a representative molecular environment for
# each, and renders Figure 4 (bar chart + fragment grid).
#
# Outputs: results/tables/figure_4_top_morgan_bits.csv,
#          results/figures/figure_4_morgan_feature_importance.png
# =============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT, "models", "m3_fp_physchem_scaffoldcv.joblib")
DATA_PATH = os.path.join(_PROJECT, "data", "raw",
                         "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")

TOP_N, FP_RADIUS, FP_NBITS = 20, 2, 2048
POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
DPI = 300


def env_to_smiles(mol, center, radius):
    if radius == 0:
        return f"[{mol.GetAtomWithIdx(center).GetSymbol()}]"
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    if not env:
        return mol.GetAtomWithIdx(center).GetSymbol()
    atoms = {a for b in env for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())}
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=list(env), canonical=True) or "?"
    except Exception:
        return "?"


def env_highlight(mol, center, radius):
    if radius == 0:
        return [center], []
    env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    if not env:
        return [center], []
    atoms = {a for b in env for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())}
    return list(atoms), env


def main():
    bundle = joblib.load(MODEL_PATH)
    fp_coef = bundle["pipe"].named_steps["clf"].coef_[0][:FP_NBITS]
    top_bits = np.argsort(np.abs(fp_coef))[::-1][:TOP_N]

    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()

    top_bits_set = set(int(b) for b in top_bits)
    bit_to_example = {}
    for smi in df["smiles"]:
        if len(bit_to_example) == len(top_bits_set):
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_NBITS, useChirality=True, bitInfo=bit_info)
        for bit_id, info in bit_info.items():
            if bit_id in top_bits_set and bit_id not in bit_to_example:
                center, radius = info[0]
                bit_to_example[bit_id] = (mol, center, radius)
    print(f"[SCAN] examples found for {len(bit_to_example)}/{len(top_bits_set)} bits")

    rows = []
    for bit_id in top_bits:
        bit_id = int(bit_id)
        coef_val = float(fp_coef[bit_id])
        frag = env_to_smiles(*bit_to_example[bit_id]) if bit_id in bit_to_example else "?"
        rows.append({"bit_id": bit_id, "coefficient": round(coef_val, 6),
                     "direction": "positive" if coef_val > 0 else "negative", "example_fragment_smiles": frag})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "figure_4_top_morgan_bits.csv"), index=False)

    # --- Figure 4: bar chart (A) + fragment grid (B) --------------------------
    bit_labels = [f"bit {r['bit_id']}" for r in rows]
    coef_vals = [r["coefficient"] for r in rows]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in coef_vals]

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.08)

    ax_bar = fig.add_subplot(gs[0])
    y_pos = np.arange(len(bit_labels))
    ax_bar.barh(y_pos, coef_vals, color=colors, edgecolor="white", height=0.72)
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(bit_labels, fontsize=13)
    ax_bar.axvline(0, color="black", lw=0.8); ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Logistic regression coefficient", fontsize=15)
    ax_bar.set_title(f"A. Top {TOP_N} Morgan fingerprint bits by |coefficient|", fontsize=16, pad=10)
    ax_bar.legend(handles=[mpatches.Patch(color="#c0392b", label="Pro-active"),
                            mpatches.Patch(color="#2471a3", label="Pro-inactive")], fontsize=13, loc="lower right")
    ax_bar.tick_params(axis="x", labelsize=13)

    gs_frag = gs[1].subgridspec(4, 5, wspace=0.15, hspace=0.55)
    for idx, row in enumerate(rows):
        ax = fig.add_subplot(gs_frag[idx // 5, idx % 5])
        ax.axis("off")
        if row["bit_id"] in bit_to_example:
            mol_ex, center, radius = bit_to_example[row["bit_id"]]
            hi_atoms, hi_bonds = env_highlight(mol_ex, center, radius)
            img = Draw.MolToImage(mol_ex, size=(220, 160), highlightAtoms=hi_atoms, highlightBonds=hi_bonds)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=15, color="grey")
        color = "#c0392b" if row["coefficient"] > 0 else "#2471a3"
        ax.set_title(f"bit {row['bit_id']}  {row['coefficient']:+.3f}", fontsize=14, color=color, pad=4)
    fig.text(0.66, 0.94, "B. Representative molecular environments (red = pro-active, blue = pro-inactive)",
              fontsize=16, ha="center")

    fig.suptitle("Figure 4. Morgan fingerprint bit importance and representative molecular environments",
                 fontsize=18, y=0.99)
    fig.savefig(os.path.join(FIGURES_DIR, "figure_4_morgan_feature_importance.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[DONE]")


if __name__ == "__main__":
    main()
