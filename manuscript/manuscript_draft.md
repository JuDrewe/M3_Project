# Scaffold-Stratified Machine Learning for M3 Muscarinic Receptor Antagonist Activity Prediction: Out-of-Fold Analysis of Per-Scaffold Generalisation Limits

**Author:** Jürgen Drewe^1,2*

^1 Max Zeller Soehne AG, Romanshorn, Switzerland
^2 University Hospital Basel, Basel, Switzerland

*Corresponding author:*
Prof. Dr. Jürgen Drewe
Max Zeller Söhne AG  
Romanshorn, Switzerland  
Email: <juergen.drewe@zellerag.ch>

---

## Abstract

Machine learning (ML) classifiers trained on public bioactivity data are increasingly used to
prioritise virtual screening hits, yet their generalisation across scaffold space remains
incompletely characterised. We report a scaffold-stratified ML pipeline for predicting antagonist
activity at the M3 muscarinic acetylcholine receptor (mAChR-M3). A dataset of 2,268 compounds
with four-tier consensus activity labels was derived from ChEMBL and split into a development set
(n = 1,813) and a scaffold-stratified held-out test set (n = 455) that was never used during
training or threshold selection. A logistic regression classifier on Morgan fingerprints (ECFP4,
2,048 bits) augmented with six physicochemical descriptors, evaluated by 10-fold scaffold-stratified
cross-validation, achieved out-of-fold (OOF) ROC-AUC = 0.983 and MCC = 0.860. On the held-out
test set the final model reached ROC-AUC = 0.971, PR-AUC = 0.993, MCC = 0.750, and balanced
accuracy = 0.897. OOF per-scaffold analysis identified three scaffolds (n >= 10 compounds) where
the model performs poorly (MCC < 0.5); for two of these the low MCC reflects extreme within-scaffold
class imbalance rather than systematic misprediction, while the third represents a genuinely
difficult mixed-activity scaffold. Domain shift quantification shows that compounds in hard
scaffolds retain substantial Tanimoto similarity to compounds outside those scaffolds, indicating
that chemical proximity to the rest of the training data does not reliably guarantee predictive
accuracy for all scaffold families. All code is openly available.

**Keywords:** M3 muscarinic receptor, antagonist prediction, scaffold-stratified cross-validation,
Morgan fingerprints, out-of-fold analysis, domain shift, UMAP

**Running title:** Scaffold-aware ML for M3 receptor antagonists
---

## 1. Introduction

The M3 muscarinic acetylcholine receptor is a therapeutically important GPCR targeted by
clinically approved antagonists for overactive bladder, chronic obstructive pulmonary disease,
and related conditions [@wess2007]. ChEMBL now contains thousands of M3 ligand activity measurements,
making data-driven ML classifiers a practical option for compound prioritisation. However, the
generalisation properties of such classifiers across scaffold space — a prerequisite for
prospective use — are rarely characterised in detail.

Scaffold-stratified cross-validation (CV), in which all compounds sharing a Murcko scaffold are
assigned to the same fold, provides a more realistic performance estimate than random CV because
it mimics querying the model with chemotypes not seen during training
[@sheridan2013;@wallach2018;@tilborg2022]. Nevertheless, aggregate scaffold-CV metrics do not reveal *which*
scaffolds the model fails on, nor do they distinguish whether failure arises from chemical
distance to training data (applicability domain limits) or from limitations of the feature
representation within the training distribution.

Here we address these questions with a three-part analysis: (1) OOF per-scaffold performance
profiling to identify hard scaffolds, (2) Tanimoto-based domain shift quantification, and (3)
UMAP visualisation to contextualise hard scaffolds within fingerprint space. We use logistic
regression — a deliberately simple, interpretable model — so that performance limitations can
be attributed primarily to the feature representation and data rather than to model capacity.

---

## 2. Materials and Methods

### 2.1 Dataset

Activity data for the M3 muscarinic receptor were derived from ChEMBL. The pre-curated input
file (`ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv`) contains ChEMBL compound
identifiers, SMILES strings, and four-tier consensus activity labels reflecting the aggregation
of multiple ChEMBL assay records per compound. The four label classes are:

| Label             | Description                                        |
|-------------------|----------------------------------------------------|
| `active`          | Active, corroborated by multiple ChEMBL records    |
| `inactive`        | Inactive, corroborated by multiple ChEMBL records  |
| `active_single`   | Active, single-record annotation                   |
| `inactive_single` | Inactive, single-record annotation                 |

The raw file contained 2,268 compounds. All records with one of the four consensus labels and
a non-empty SMILES string were retained. No additional deduplication was applied. The final
dataset comprised **2,268 compounds**: 1,788 active or active_single (78.8%) and 480 inactive
or inactive_single (21.2%).

### 2.2 Confidence-Based Sample Weights

To reflect label reliability without applying class-balance correction (which would distort
probability estimates used for threshold selection), we assigned confidence-based sample weights:

| Label             | Weight |
|-------------------|--------|
| `active`          | 1.0    |
| `inactive`        | 1.0    |
| `active_single`   | 0.5    |
| `inactive_single` | 0.7    |

These weights were passed as `sample_weight` to the logistic regression `fit()` call. No
`class_weight` parameter was used, deliberately avoiding double-weighting.

### 2.3 Molecular Features

**Morgan fingerprints (ECFP4):** Circular fingerprints with radius = 2, 2,048 bits, with
chirality encoding, generated with RDKit `rdFingerprintGenerator.GetMorganGenerator`. Each
compound is represented as a binary 2,048-dimensional vector [@rogers2010].

**Physicochemical descriptors:** Six RDKit descriptors — molecular weight (MolWt), partition
coefficient (MolLogP), hydrogen-bond donor count (NumHDonors), hydrogen-bond acceptor count
(NumHAcceptors), topological polar surface area (TPSA), and number of rotatable bonds
(NumRotatableBonds) — were concatenated to the Morgan fingerprint vector without feature
scaling. Because the Morgan fingerprint dimensions are binary and uniformly bounded, and
logistic regression with liblinear optimisation is robust to moderate feature scale differences
in this setting, standardisation was not applied.

### 2.4 Scaffold Assignment

Murcko scaffolds [@bemis1996] were computed with
`rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol()`, with canonical SMILES generated
by `Chem.MolToSmiles(isomericSmiles=False)`. For the present dataset, no compounds produced
an empty scaffold string; all 2,268 compounds were therefore retained for scaffold-stratified
steps. The code silently drops any future compound yielding an empty scaffold.

### 2.5 Cross-Validation Design

An outer scaffold-stratified split (`StratifiedGroupKFold`, n_splits = 5, random_state = 0)
produced a **development set** (n = 1,813, ~80%) and a **held-out test set** (n = 455, ~20%).
The test set was set aside and not examined until final reporting (Section 3.3).

Within the development set, **10-fold scaffold-stratified cross-validation**
(`StratifiedGroupKFold`, n_splits = 10, random_state = 42) was applied using the Murcko
scaffold as the group variable. All compounds sharing a scaffold appear in exactly one
held-out fold.

### 2.6 Model

Logistic regression (`sklearn.linear_model.LogisticRegression`, solver = "liblinear",
max_iter = 5,000, C = 1.0, class_weight = None) was applied directly to the concatenated
feature matrix. The final model was fitted on the full development set.

### 2.7 Classification Threshold Selection

The classification threshold was selected by maximising Matthews Correlation Coefficient
(MCC) [@matthews1975] on pooled OOF probabilities
across 181 candidate thresholds in [0.05, 0.95]:

```python
thresholds = np.linspace(0.05, 0.95, 181)
mccs = [matthews_corrcoef(oof_true, (oof_proba >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(mccs)]
```

The selected threshold (0.70 for FP + PhysChem) was stored in the model bundle and applied
unchanged to the held-out test set.

### 2.8 Per-Scaffold OOF Performance and Hard-Scaffold Identification

OOF predictions were assembled into a compound-level record. For each scaffold where both
active and inactive compounds appeared among OOF predictions, MCC and balanced accuracy were
computed. Scaffolds with >= 10 OOF compounds and MCC < 0.5 were flagged as *hard scaffolds*.

### 2.9 Domain Shift Quantification

For each compound in a hard scaffold, the maximum Tanimoto similarity to all compounds
*outside that scaffold* was computed using RDKit `BulkTanimotoSimilarity` on Morgan
fingerprints. Per-scaffold summary statistics (mean and minimum of the per-compound maximum
similarity) characterise how structurally similar each hard scaffold is to the rest of the
dataset, which serves as a proxy for training coverage.

### 2.10 UMAP Visualisation

A 2D UMAP embedding [@mcinnes2018] (`umap-learn` 0.5.x, n_neighbors = 15,
min_dist = 0.1, metric = "jaccard", random_state = 42) of the binary Morgan fingerprint
matrix was computed
for all development-set compounds. OOF error labels were joined to UMAP coordinates via
original DataFrame row indices captured before any filtering.

### 2.11 Software

RDKit 2023.x [@landrum2006]; scikit-learn >= 1.1 [@pedregosa2011]; umap-learn
0.5.x [@mcinnes2018]; numpy; pandas; matplotlib; joblib. Python 3.10. Full environment
specification: `environment.yml`.

---

## 3. Results

### 3.1 Dataset Composition

The dataset comprised **2,268 compounds** across **1,022 unique Murcko scaffolds**. The class
distribution was 78.8% active/active_single and 21.2% inactive/inactive_single. The outer
scaffold-stratified split yielded 1,813 development-set compounds (78.8% active) and 455
held-out test compounds (78.9% active).

### 3.2 Cross-Validation Performance

Ten-fold scaffold-stratified CV on the development set (1,813 OOF predictions) yielded:

| Metric            | FP-only (thr = 0.61) | FP + PhysChem (thr = 0.70) |
|-------------------|----------------------|---------------------------|
| ROC-AUC           | 0.983                | 0.983                     |
| PR-AUC            | 0.996                | 0.995                     |
| MCC               | 0.857                | 0.860                     |
| Balanced Accuracy | 0.934                | 0.946                     |
| Brier score       | 0.040                | 0.038                     |

Adding six physicochemical descriptors produced a marginal improvement in OOF MCC
(0.860 vs. 0.857) and balanced accuracy (0.946 vs. 0.934), with near-identical ROC-AUC
and PR-AUC. The differences are small and were not subject to a formal significance test
given the limited number of folds; FP + PhysChem was selected as the final feature set
based on its nominally higher MCC and lower Brier score.

OOF probability distributions showed good separation between true actives and true inactives
(Figure 2a-b), with the main overlap in the 0.3-0.6 probability range. The calibration curve
(Figure 1) indicated generally reasonable calibration, with a Brier score of 0.038 for
FP + PhysChem.

> **Figure 1.** OOF calibration curve. See figure_legends.md.
> `results/figures/fig_01_oof_calibration.png`

> **Figure 2a-b.** OOF predicted probability distributions. See figure_legends.md.
> `results/figures/fig_02_oof_prob_tp_tn.png`, `fig_03_oof_prob_fp_fn.png`

### 3.3 Held-Out Test Set Performance

The final model (FP + PhysChem, trained on the full development set, threshold = 0.70) was
evaluated once on the held-out test set:

| Metric            | Value |
|-------------------|-------|
| Prevalence (test) | 0.789 |
| ROC-AUC           | 0.971 |
| PR-AUC            | 0.993 |
| MCC               | 0.750 |
| Balanced Accuracy | 0.897 |

The held-out ROC-AUC (0.971) is somewhat lower than the OOF ROC-AUC (0.983), consistent
with expected optimism in the OOF estimate. The held-out MCC (0.750) is notably lower than
the OOF MCC (0.860), which likely reflects threshold transfer: the OOF-optimised threshold
of 0.70 may not be optimal for the specific scaffold composition of the held-out fold.

### 3.4 Per-Scaffold OOF Performance

Across 1,022 unique scaffolds in the development set, MCC was computable for 13 scaffolds
(those for which both active and inactive compounds appeared in OOF predictions). Of these 13,
**three scaffolds with >= 10 OOF compounds had MCC < 0.5** and were classified as hard
scaffolds (Table 1; Figures 3-4).

**Table 1.** Hard scaffolds (n >= 10 OOF compounds, MCC < 0.5).

| Scaffold (Murcko SMILES, abbreviated)              | n  | MCC   | Bal. Acc. | Errors | Active frac. |
|----------------------------------------------------|----|-------|-----------|--------|--------------|
| `c1ccc(C(CC2CC3CCC(C2)[NH2+]3)c2ccccc2)cc1` (S1)  | 29 | 0.000 | 0.500     | 2      | 0.931        |
| `O=C(Nc1ccccc1)NC(Cc1ccccc1)C(=O)NC1CCN(...)C1` (S2) | 17 | 0.000 | 0.500  | 1      | 0.941        |
| `c1ccc2c(c1)CCN1Cc3c(ccc4[nH]ccc34)OC21` (S3)     | 18 | 0.193 | 0.545     | 10     | 0.611        |

**S1 and S2** are near-pure active series: S1 contains 27 actives and 2 inactives; S2
contains 16 actives and 1 inactive. MCC = 0 in both cases because extreme within-scaffold
class imbalance drives the true-negative count to zero — the model, presented with an
overwhelmingly active scaffold, predicts all compounds as active. When TN = 0 the MCC
numerator evaluates to zero regardless of TP count, making MCC an unreliable performance
metric for these scaffolds. The 1-2 errors are false positives (the rare inactives predicted
active) and do not indicate systematic failure to identify actives. **S3** (active fraction
0.611, 10 of 18 predictions incorrect, MCC = 0.193) is qualitatively different: the model
cannot reliably separate actives from inactives within this mixed-activity scaffold.

> **Figure 3.** Per-scaffold OOF MCC distribution. See figure_legends.md.
> `results/figures/fig_04_scaffold_mcc_dist.png`

> **Figure 4.** OOF MCC vs. scaffold size. See figure_legends.md.
> `results/figures/fig_05_scaffold_mcc_vs_size.png`

### 3.5 Domain Shift Analysis

To examine whether hard-scaffold difficulties correlate with chemical distance from the rest
of the dataset, we computed, for each compound in a hard scaffold, the maximum Tanimoto
similarity to all compounds *outside that scaffold* (Table 2; Figure 5).

**Table 2.** Domain shift summary for hard scaffolds (maximum Tanimoto similarity to
compounds outside the scaffold).

| Scaffold | n  | Mean max-sim | Min max-sim |
|----------|----|--------------|-------------|
| S3       | 18 | 0.572        | 0.360       |
| S1       | 29 | 0.702        | 0.596       |
| S2       | 17 | 0.787        | 0.708       |

All three hard scaffolds retain substantial Tanimoto similarity to compounds outside their
scaffold group. S1 and S2 are structurally close to the rest of the dataset (mean max-sim
0.70 and 0.79 respectively); their MCC = 0 reflects within-scaffold class imbalance, as
discussed above. S3 has lower mean max-similarity (0.572) and a minimum of 0.360, meaning
some S3 compounds are more structurally distinct from the rest of the dataset. Whether this
partial structural isolation contributes to S3's higher error rate cannot be determined from
Tanimoto similarity alone; more targeted analyses (e.g., matched molecular pair analysis,
per-bit feature attribution) would be needed to draw firmer conclusions.

> **Figure 5.** Tanimoto similarity distributions for hard scaffolds. See figure_legends.md.
> `results/figures/fig_06_domain_shift_similarity.png`

### 3.6 UMAP Embedding

UMAP projection of Morgan fingerprints (Figures 6-7) shows that hard-scaffold compounds are
distributed throughout the training-data embedding rather than isolated at the periphery,
consistent with the domain shift results. False negatives from S3 appear at the boundary
between active- and inactive-dense regions, consistent with the mixed activity composition
of that scaffold.

> **Figure 6.** UMAP embedding coloured by hard-scaffold membership. See figure_legends.md.
> `results/figures/fig_07_umap_hard_scaffolds.png`

> **Figure 7.** UMAP embedding coloured by OOF prediction outcome. See figure_legends.md.
> `results/figures/fig_08_umap_misclassifications.png`

### 3.7 Morgan Fingerprint Feature Importance

To examine which structural features contribute most to model predictions, we ranked all
2,048 Morgan fingerprint bits by the absolute value of their logistic regression
coefficients. The top 20 bits are tabulated in `results/tables/top_morgan_bits.csv` and
illustrated in Figure 9. Bits with positive coefficients are associated with higher
predicted probability of activity across the training set; bits with negative coefficients
are associated with lower predicted probability.

The visualised substructures are example atom environments at ECFP4 radius 2, taken from
the first training compound in which each bit is active. These environments show the
structural contexts the model weights most strongly, but should be interpreted
conservatively: Morgan fingerprint bits are correlated across structurally similar compounds,
and the coefficient of any individual bit reflects its marginal contribution given all other
bits in the regularised model, not an independent measure of fragment activity.

> **Figure 9.** Top 20 Morgan fingerprint bits ranked by logistic regression coefficient
> magnitude. Left panel: horizontal bar chart coloured by direction (red = positive
> coefficient; blue = negative coefficient). Right panel: example atom environments at
> radius 2 drawn from the first training compound in which each bit is active.
> Full data in `results/tables/top_morgan_bits.csv`.
> `results/figures/fig_feature_importance_morgan_bits.png`

---

## 4. Discussion

### 4.1 Model Performance in Context

The held-out ROC-AUC of 0.971, obtained on a scaffold-stratified held-out test set,
indicates strong discriminative performance for this ChEMBL-derived M3 dataset. Direct
comparison to published M3 QSAR benchmarks is complicated by differences in dataset
composition, label definitions, and validation protocols; a rigorous benchmark comparison
is outside the scope of this work. The held-out MCC of 0.750 reflects a
practically useful classifier; it falls below the OOF MCC (0.860) primarily because the
OOF-optimised threshold (0.70) may not transfer optimally to the test fold's scaffold
composition.

### 4.2 Hard Scaffolds and the Limits of Scaffold-Stratified CV

Three scaffolds met the hard-scaffold criterion (n >= 10, MCC < 0.5), but their nature
differs substantially. S1 and S2 are near-pure active series where MCC = 0 is a direct
consequence of extreme within-scaffold class imbalance (1-2 inactives in 17-29 compounds):
with near-zero true-negative counts, MCC is numerically unstable and should not be
interpreted as evidence of a generalisation failure. S3 is the genuinely difficult scaffold:
10 of 18 OOF predictions are errors across a mixed-activity series, and its lower Tanimoto
similarity to compounds outside the scaffold suggests limited structural coverage in the
training data may be a contributing factor.

These observations highlight two practical points. First, per-scaffold MCC should be
interpreted alongside class composition; scaffolds with extreme active-fraction values
(> 0.9 or < 0.1) require alternative metrics such as sensitivity or specificity. Second,
aggregate scaffold-CV metrics alone conceal this scaffold-level heterogeneity; per-scaffold
OOF profiling should accompany aggregate reporting.

### 4.3 Marginal Value of Physicochemical Descriptors

The six physicochemical descriptors produced only marginal OOF improvement over Morgan
fingerprints alone (delta-MCC = 0.003, delta-Brier = 0.002). This suggests that, for this
dataset and model class, the bulk of discriminative information is captured by the 2,048-bit
ECFP4 vector. The improvement in balanced accuracy (0.946 vs. 0.934) may partly reflect
better threshold calibration at 0.70 vs. 0.61.

### 4.4 Limitations

- **Class imbalance:** 78.8% of compounds are active or active_single. Per-scaffold MCC is
  unreliable for near-pure series with few negatives.
- **No canonical deduplication:** Structurally near-duplicate SMILES were not removed prior
  to modelling; structural redundancy could inflate apparent performance.
- **Linear model:** Logistic regression cannot capture non-linear activity cliffs; non-linear
  models may improve performance on hard scaffolds but introduce calibration and
  interpretability trade-offs.
- **Binary labels:** Continuous potency values (e.g., pK_i) are binarised, discarding
  potency gradient information that could help distinguish borderline cases.

### 4.5 Applicability Domain

The reliability of model predictions depends on adequate coverage of the query compound's
chemical neighbourhood in the training data. Scaffold-stratified cross-validation provides
a conservative estimate of this reliability by explicitly evaluating performance on scaffold
families absent from training, making it a more realistic proxy for prospective use than
random-split validation [@sheridan2013]. The Tanimoto similarity analysis in Section 3.5
provides a compound-level proxy for applicability domain: for each query compound, the
maximum Tanimoto similarity to training-set compounds (computed on Morgan fingerprints) gives
an indication of whether the compound falls within the structural space the model has
observed. The hard-scaffold analysis demonstrates, however, that high similarity to the
training set is not a sufficient condition for reliable prediction; per-scaffold OOF error
rates should be consulted alongside similarity metrics when assessing prediction confidence
for a specific scaffold family.

---

## 5. Conclusions

A scaffold-stratified ML pipeline for M3 antagonist prediction achieved ROC-AUC = 0.971
and MCC = 0.750 on a scaffold-stratified held-out test set. OOF per-scaffold analysis
identified three chemical series with MCC < 0.5. Two of these (S1, S2) are near-pure active
scaffolds where MCC = 0 is a consequence of extreme within-scaffold class imbalance and
should not be interpreted as systematic model failure. The third (S3, n = 18, 10/18 errors)
is a genuinely difficult mixed-activity scaffold whose compounds retain moderate Tanimoto
similarity to the rest of the dataset. Per-scaffold profiling of this kind should accompany
aggregate scaffold-CV metrics as a standard component of bioactivity model evaluation.

---

## Acknowledgements

[FILL]

---

## Data and Code Availability

All code used for data processing, model training, and figure generation is available in
this repository. The machine learning pipeline, including the scaffold-stratified
cross-validation procedure and figure generation scripts, is implemented in Python and
documented in the `scripts/` directory (`train_model.py`, `generate_figures.py`,
`compute_oof_metrics.py`, `morgan_feature_importance.py`). The computational environment
required to reproduce the analysis is fully specified in `environment.yml`. Input data are
derived from ChEMBL and can be reconstructed from public records following the curation
protocol described in Section 2.1. The trained model bundle
(`m3_fp_physchem_scaffoldcv.joblib`) is available upon reasonable request.

---

## References

See `references.bib` for the full bibliography. In-text citations use BibTeX keys.

\[wess2007\] Wess et al., *Nat Rev Drug Discov* 2007
\[sheridan2013\] Sheridan, *J Chem Inf Model* 2013, 53:783–790
\[wallach2018\] Wallach et al., *J Chem Inf Model* 2018, 58:916–932
\[tilborg2022\] van Tilborg et al., *J Chem Inf Model* 2022, 62:5938–5951
\[bemis1996\] Bemis & Murcko, *J Med Chem* 1996, 39:2887–2893
\[rogers2010\] Rogers & Hahn, *J Chem Inf Model* 2010, 50:742–754
\[matthews1975\] Matthews, *Biochim Biophys Acta* 1975, 405:442–451
\[pedregosa2011\] Pedregosa et al., *J Mach Learn Res* 2011, 12:2825–2830
\[landrum2006\] Landrum, RDKit: Open-source cheminformatics, rdkit.org
\[mcinnes2018\] McInnes et al., arXiv:1802.03426, 2018

---

## Supplementary Material

**Figure S1 — Pipeline Overview.** End-to-end ML workflow (10 steps) from data curation to
held-out test evaluation. `results/figures/fig_pipeline_overview.png`

**Table S1 — Full Per-Scaffold OOF Performance.**
`results/tables/scaffold_performance_summary.csv`

**Table S2 — OOF Misclassification Record.**
`results/tables/scaffold_cv_misclassifications.csv`

**Table S3 — Domain Shift, Per Compound.**
`results/tables/hard_scaffolds_domain_shift_per_compound.csv`
