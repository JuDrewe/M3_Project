# =============================================================================
# 09_umap_split_comparison.py
# Compares development vs. held-out test compounds in UMAP space, for the
# scaffold-based split and the random (stratified) split, using the cached
# UMAP embedding (Supplementary Figure S2).
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT, "data", "raw", "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
EMB_PATH = os.path.join(_PROJECT, "results", "tables", "umap_train_embedding.csv")
OUT_PATH = os.path.join(_PROJECT, "results", "figures", "figure_s2_umap_split_comparison.png")

POS_LABELS = {"active", "active_single"}
NEG_LABELS = {"inactive", "inactive_single"}
SEED, TEST_FRACTION = 0, 0.20


def smiles_to_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    scaf = MurckoScaffold.GetScaffoldForMol(mol) if mol else None
    return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else ""


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["consensus_label"].isin(POS_LABELS | NEG_LABELS)].copy()
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["y"] = df["consensus_label"].isin(POS_LABELS).astype(int)
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    df = df[df["scaffold"] != ""].reset_index(drop=True)

    y, scaffolds = df["y"].values, df["scaffold"].values
    _, test_s = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                      .split(df["smiles"], y, groups=scaffolds))
    _, test_r = train_test_split(np.arange(len(y)), test_size=TEST_FRACTION, stratify=y, random_state=SEED)
    print(f"[SPLITS] scaffold test={len(test_s)}  random test={len(test_r)}")

    df["scaffold_split"] = "dev"; df.loc[df.index[test_s], "scaffold_split"] = "test"
    df["random_split"] = "dev"; df.loc[df.index[test_r], "random_split"] = "test"

    emb = pd.read_csv(EMB_PATH)
    merged = emb.merge(df[["molecule_chembl_id", "scaffold_split", "random_split"]],
                        on="molecule_chembl_id", how="inner")
    assert len(merged) == len(emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, col, title in [(axes[0], "scaffold_split", "Scaffold split"),
                            (axes[1], "random_split", "Random (stratified) split")]:
        dev, test = merged[merged[col] == "dev"], merged[merged[col] == "test"]
        ax.scatter(dev["UMAP1"], dev["UMAP2"], s=10, alpha=0.5, color="#1f77b4",
                   label=f"Development set (n={len(dev)})")
        ax.scatter(test["UMAP1"], test["UMAP2"], s=14, alpha=0.7, color="#d62728",
                   label=f"Held-out test set (n={len(test)})")
        ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title=title)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("UMAP projection of development vs. held-out test compounds,\n"
                 "compared between scaffold-based and random (stratified) splitting", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[DONE]")


if __name__ == "__main__":
    main()
