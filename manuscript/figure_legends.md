# Figure Legends

## Main Figures

---

**Figure 1. Out-of-fold calibration curve.**
Reliability diagram for the FP + PhysChem logistic regression model evaluated by 10-fold
scaffold-stratified cross-validation (1,813 OOF predictions). Blue circles show the mean
predicted probability vs. observed fraction of positives within each probability bin
(quantile binning, 10 bins); the dashed diagonal represents perfect calibration. Brier
score = 0.038 (baseline = p(1-p) for the observed prevalence) is annotated in the panel.
`results/figures/fig_01_oof_calibration.png`

---

**Figure 2. OOF predicted probability distributions — true positives and true negatives.**
Overlapping histograms (30 bins) of the out-of-fold predicted probability of activity
(p_active) for true positive (active compounds correctly predicted active, blue) and true
negative (inactive compounds correctly predicted inactive, orange) OOF predictions. The
y-axis shows compound count. Both groups appear well separated; some overlap is visible at
intermediate probability values.
`results/figures/fig_02_oof_prob_tp_tn.png`

---

**Figure 3. OOF predicted probability distributions — false positives and false negatives.**
Overlapping histograms (20 bins) of p_active for false positive (inactive compounds
predicted active, blue) and false negative (active compounds predicted inactive, orange)
OOF predictions. The vertical dashed line marks the OOF-optimised classification threshold
(0.70). The y-axis shows compound count.
`results/figures/fig_03_oof_prob_fp_fn.png`

---

**Figure 4. Distribution of per-scaffold OOF MCC.**
Histogram of Matthews Correlation Coefficient computed per scaffold for all scaffolds where
both active and inactive compounds appeared among OOF predictions (13 scaffolds total). The
vertical dashed line (crimson) marks the hard-scaffold threshold (MCC = 0.5). The x-axis
spans [-1, 1] with major gridlines at 0.25 intervals.
`results/figures/fig_04_scaffold_mcc_dist.png`

---

**Figure 5. Per-scaffold OOF MCC as a function of scaffold size.**
Scatter plot of OOF MCC vs. number of OOF compounds per scaffold for scaffolds where both
classes are represented. Standard scaffolds are shown as circles; hard scaffolds (MCC < 0.5)
are shown as diamond markers. The horizontal dashed line marks the hard-scaffold threshold
(MCC = 0.5). Individual scaffolds are not labelled in the figure.
`results/figures/fig_05_scaffold_mcc_vs_size.png`

---

**Figure 6. Tanimoto similarity distributions for hard-scaffold compounds.**
Three-panel figure (one panel per hard scaffold) showing histograms of the per-compound
maximum Tanimoto similarity (Morgan fingerprints, radius = 2) to all compounds outside the
same scaffold. The x-axis is fixed at [0, 1] for all panels. Each panel title shows the
scaffold OOF MCC and compound count (n). Higher values indicate structural similarity to
the rest of the dataset.
`results/figures/fig_06_domain_shift_similarity.png`

---

**Figure 7. UMAP embedding coloured by hard-scaffold membership.**
Two-dimensional UMAP projection (n_neighbors = 15, min_dist = 0.1, metric = Jaccard,
random_state = 42) of Morgan fingerprints for all development-set compounds. Background
points (all compounds, n = 1,813) are shown in small, semi-transparent circles.
Hard-scaffold compounds are shown as larger, more opaque points. No individual scaffold
or compound labels are shown.
`results/figures/fig_07_umap_hard_scaffolds.png`

---

**Figure 8. UMAP embedding coloured by OOF prediction outcome.**
Same UMAP projection as Figure 7. Points are coloured by out-of-fold prediction outcome:
correctly classified compounds (OOF OK, small semi-transparent circles), false positives
(OOF FP, larger markers), and false negatives (OOF FN, larger markers). Compound counts
for each category are shown in the legend. No individual compound labels are shown.
`results/figures/fig_08_umap_misclassifications.png`

---

## Supplementary Figures

---

**Figure S1. M3 ML pipeline overview.**
Schematic of the ten-step machine learning workflow: (1) ChEMBL data curation and consensus
labelling; (2) SMILES parsing and Murcko scaffold computation; (3) confidence-based sample
weight assignment; (4) outer scaffold-stratified dev/test split (~80/20); (5) Morgan
fingerprint generation (ECFP4, 2,048 bits, chirality); (6) physicochemical descriptor
computation (MolWt, LogP, HBD, HBA, TPSA, RotBonds); (7) 10-fold scaffold-stratified
cross-validation; (8) OOF threshold selection by MCC maximisation; (9) final model fitting
on the development set; (10) single held-out test evaluation. Steps are colour-coded by
phase: data preparation (blue), feature engineering (green), modelling (orange), evaluation
(purple).
`results/figures/fig_pipeline_overview.png`
