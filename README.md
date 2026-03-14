# M3 Muscarinic Receptor Antagonist ML Pipeline

Machine learning pipeline for predicting M3 muscarinic receptor antagonist
activity using scaffold-stratified cross-validation and Morgan fingerprints.

## Project structure

    data/        Raw, processed, and external datasets
    notebooks/   Jupyter notebooks
    src/         Reusable source modules
    models/      Saved model artefacts
    results/     Tables and figures
    scripts/     Standalone runnable scripts

## Setup

    conda env create -f environment.yml
    conda activate m3_ml

## Data

Input: ChEMBL_M3_consensus_labels_more_negatives_with_meta.csv
Labels: active / active_single / inactive / inactive_single
        (consensus from ChEMBL + BindingDB)

## Key notebook

notebooks/M3_ML_classic_20260225.ipynb
  Scaffold-stratified 10-fold CV, Morgan FP + PhysChem,
  Logistic Regression, domain shift quantification, UMAP projection.

## Reproducibility

1. create environment
conda env create -f environment.yml

2. activate
conda activate m3_ml

3. train model
python scripts/train_model.py

4. generate figures
python scripts/generate_figures.py
