# =============================================================================
# 08_umap_analysis.py
# UMAP projection of the M3 training chemical space and the COCONUT natural-
# product screen; produces the 4-panel Figure 7 directly.
#
#   Panel A: training compounds by activity class
#   Panel B: training chemical-space density (KDE contours)
#   Panel C: projected COCONUT background + top-1% predicted hits
#   Panel D: top hits classified as inside/outside the training chemical
#            space, using the 95% highest-density-region threshold of the
#            training-set KDE (5th percentile of training-point densities)
#
# Outputs: results/tables/umap_train_embedding.csv,
#          results/tables/umap_coconut_projection.csv,
#          results/tables/umap_tophits_inside_outside.csv,
#          results/figures/figure_7_umap_chemical_space.png
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(_PROJECT, "data", "raw", "ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv")
COCONUT_PATH = os.path.join(_PROJECT, "data", "raw", "coconut_screen_ranked.csv")
TABLES_DIR = os.path.join(_PROJECT, "results", "tables")
FIGURES_DIR = os.path.join(_PROJECT, "results", "figures")

N_BITS, RADIUS = 2048, 2
TOP_PERCENTILE = 0.01
BACKGROUND_SAMPLE_N = 50000
SEED = 42
UMAP_KW = dict(n_neighbors=30, min_dist=0.1, metric="jaccard", random_state=SEED)
N_LEVELS = 6

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)


def fps_to_numpy(smiles_series):
    fps, keep = [], []
    for i, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(FPGEN.GetFingerprint(mol))
            keep.append(i)
    arr = np.zeros((len(fps), N_BITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr, keep


def main():
    train = pd.read_csv(TRAIN_PATH)
    train = train[train["consensus_label"].isin(
        ["active", "active_single", "inactive", "inactive_single"])].copy()
    train = train.dropna(subset=["smiles"]).copy()
    train["smiles"] = train["smiles"].astype(str).str.strip()
    X_train, keep = fps_to_numpy(train["smiles"])
    train = train.iloc[keep].reset_index(drop=True)
    train["active"] = train["consensus_label"].isin(["active", "active_single"]).astype(int)
    print(f"[TRAIN] n={len(train)}")

    coconut = pd.read_csv(COCONUT_PATH).dropna(subset=["SMILES", "p_antagonist"]).copy()
    coconut["SMILES"] = coconut["SMILES"].astype(str).str.strip()
    print(f"[COCONUT] n={len(coconut)}")

    threshold = coconut["p_antagonist"].quantile(1 - TOP_PERCENTILE)
    coconut["is_top_hit"] = coconut["p_antagonist"] >= threshold
    hits = coconut[coconut["is_top_hit"]]
    background = coconut[~coconut["is_top_hit"]].sample(
        n=min(BACKGROUND_SAMPLE_N, (~coconut["is_top_hit"]).sum()), random_state=SEED)
    plot_df = pd.concat([background, hits], ignore_index=True)

    # Fingerprint only the ~54k compounds actually needed for the plot
    # (not all 406k COCONUT compounds) to keep memory usage bounded.
    X_plot, keep_plot = fps_to_numpy(plot_df["SMILES"])
    plot_df = plot_df.iloc[keep_plot].reset_index(drop=True)
    print(f"[HITS] n={plot_df['is_top_hit'].sum()} threshold={threshold:.6g}  "
          f"[BACKGROUND] n={(~plot_df['is_top_hit']).sum()}")

    print("[UMAP] fitting on training compounds")
    reducer = umap.UMAP(**UMAP_KW)
    emb_train = reducer.fit_transform(X_train)
    print("[UMAP] transforming COCONUT set")
    emb_coco = reducer.transform(X_plot)

    train[["molecule_chembl_id", "smiles", "consensus_label", "active"]].assign(
        UMAP1=emb_train[:, 0], UMAP2=emb_train[:, 1]).to_csv(
        os.path.join(TABLES_DIR, "umap_train_embedding.csv"), index=False)

    keep_cols = [c for c in ["NP-ID", "molecular_name", "molecular_formula", "MW", "XLogP", "SMILES",
                              "InChIKey", "Murcko_scaffold", "p_antagonist"] if c in plot_df.columns]
    plot_df[keep_cols].assign(is_top_hit=plot_df["is_top_hit"].values,
                              UMAP1=emb_coco[:, 0], UMAP2=emb_coco[:, 1]).to_csv(
        os.path.join(TABLES_DIR, "umap_coconut_projection.csv"), index=False)

    # --- 95% highest-density-region threshold (5th percentile of training densities) ---
    x, y = emb_train[:, 0], emb_train[:, 1]
    kde = gaussian_kde(np.vstack([x, y]))
    outer_level = float(np.percentile(kde(np.vstack([x, y])), 5))

    hits_mask = plot_df["is_top_hit"].values
    hit_density = kde(np.vstack([emb_coco[hits_mask, 0], emb_coco[hits_mask, 1]]))
    inside = hit_density >= outer_level
    print(f"[AD] hits inside={inside.sum()} outside={(~inside).sum()}  threshold={outer_level:.6g}")

    hits_out = plot_df[hits_mask].copy()
    hits_out["UMAP1"], hits_out["UMAP2"] = emb_coco[hits_mask, 0], emb_coco[hits_mask, 1]
    hits_out["density"], hits_out["inside_training_space"] = hit_density, inside
    hits_out.to_csv(os.path.join(TABLES_DIR, "umap_tophits_inside_outside.csv"), index=False)

    # --- Figure 7: 4 panels ----------------------------------------------------
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    inact, act = train["active"].values == 0, train["active"].values == 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    ax = axes[0, 0]
    ax.scatter(x[inact], y[inact], s=12, alpha=0.6, color="#1f77b4", label="Training inactive")
    ax.scatter(x[act], y[act], s=12, alpha=0.7, color="#ff7f0e", label="Training active")
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="A. Training compounds"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(x[inact], y[inact], s=10, alpha=0.5, color="#1f77b4", label="Training inactive")
    ax.scatter(x[act], y[act], s=10, alpha=0.6, color="#ff7f0e", label="Training active")
    ax.contour(xx, yy, zz, levels=N_LEVELS, colors="black", linewidths=1.0, alpha=0.85)
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="B. Training chemical-space density"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    bg_mask = ~hits_mask
    ax.scatter(emb_coco[bg_mask, 0], emb_coco[bg_mask, 1], s=4, alpha=0.15, color="#7fb3d5",
               label="COCONUT background sample")
    ax.scatter(emb_coco[hits_mask, 0], emb_coco[hits_mask, 1], s=10, alpha=0.55, color="#ff7f0e",
               label="Top 1% predicted hits")
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="C. Projected COCONUT compounds"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    hx, hy = emb_coco[hits_mask, 0], emb_coco[hits_mask, 1]
    ax.contour(xx, yy, zz, levels=N_LEVELS, colors="black", linewidths=0.8, alpha=0.6)
    ax.scatter(hx[inside], hy[inside], s=18, facecolors="none", edgecolors="black", linewidths=0.8,
               label=f"Inside training space (n={inside.sum()})")
    ax.scatter(hx[~inside], hy[~inside], s=18, facecolors="none", edgecolors="red", linewidths=0.8,
               label=f"Outside training space (n={(~inside).sum()})")
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="D. Top hits inside vs outside training space")
    ax.legend(fontsize=7.5)

    fig.suptitle("UMAP representation of the chemical space of M3 receptor ligands "
                 "and screened natural products", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGURES_DIR, "figure_7_umap_chemical_space.png"), dpi=300)
    plt.close(fig)
    print("[DONE]")


if __name__ == "__main__":
    main()
