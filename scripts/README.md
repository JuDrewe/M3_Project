# Analysis pipeline

Run scripts in numerical order. Each script reads only the input files listed
and writes to `results/tables/` and/or `results/figures/`.

| Script | Purpose | Key outputs |
|---|---|---|
| `01_curate_chembl.py` | ChEMBL raw export \u2192 relation-aware consensus labelling (active/active_single/inactive/inactive_single), as described in Methods 2.1 | `data/raw/ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv` |
| `02_train_model.py` | Scaffold-stratified training, OOF predictions, threshold selection, difficult-scaffold identification, domain-shift analysis, held-out evaluation | `models/m3_fp_physchem_scaffoldcv.joblib`, misclassification/scaffold/domain-shift tables |
| `03_compute_oof_metrics.py` | OOF and held-out metric summaries + calibration bins | `oof_metrics_summary.csv`, `heldout_test_metrics_summary.csv`, calibration bin tables |
| `04_y_randomization.py` | Label-permutation control (100 permutations), identical CV protocol to `02_train_model.py` | `y_randomization_summary.csv`, Supplementary Figure S1 |
| `05_generate_figures.py` | Main-text Figures 1, 2, 3, 5, 6 | `figure_{1,2,3,5,6}_*.png` |
| `06_umap_analysis.py` | UMAP fit on training data, transform of COCONUT screen, in/out-of-domain classification (Figure 7) | `umap_*.csv`, `figure_7_umap_chemical_space.png` |
| `07_morgan_feature_importance.py` | Top-20 Morgan bits + representative fragments (Figure 4) | `figure_4_top_morgan_bits.csv`, `figure_4_morgan_feature_importance.png` |
| `08_umap_split_comparison.py` | UMAP comparison of development/test compounds, scaffold vs. random split (Supplementary Figure S2) | `figure_s2_umap_split_comparison.png` |
| `09_additional_analyses.py` | Random-split baseline, Tanimoto similarity, AD-restricted performance, singles-ablation (Tables 6-9) | corresponding `*.csv` tables, `additional_analyses_summary.json` |
| `10_generate_supplementary_figures.py` | Supplementary Figures S3-S6 (reads only tables already saved by script 09; no recomputation) | `figure_s{3,4,5,6}_*.png` |

## Requirements
scikit-learn 1.7.1, rdkit 2026.3.1, umap-learn 0.5.11 (see `environment.yml`).

## Note on `01_curate_chembl.py`
Verified by exact match of output column structure, output file path, and
consensus-label class counts (active=286, active_single=1502, inactive=17,
inactive_single=463) against the dataset used throughout scripts 02-10.
The raw ChEMBL export (`data/raw/ChEMBL_M3_raw.csv`) that serves as its input
was not available in this environment, so the script itself could not be
re-executed end-to-end here; the match above was established by static
comparison rather than by rerunning it. A separate variant,
`01a_curate_m3_consensus_with_meta_database.py`, exists with additional
metadata columns and a different output path/filename; it does not match the
dataset actually used and is not part of this pipeline.

