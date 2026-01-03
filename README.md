# Dual-Stage Transformer–CNN Framework for CRISPR Off-Target Prediction
This repository contains the official implementation of a **dual-stage deep learning framework** for predicting CRISPR–Cas9 off-target effects. The model integrates **frozen sequence embeddings from a pretrained DNA language model (DNABERT)** with a **CNN–Transformer hybrid architecture** to improve generalization to experimentally validated off-target datasets.

The implementation is designed with **reproducibility and ablation clarity** as first-class goals and directly corresponds to the results reported in the accompanying student research paper.

## Overview

CRISPR off-target prediction remains challenging due to limited experimentally validated data and strong sequence dependence. To address this, we propose a **two-stage approach**:

- **Stage-1 (Sequence Prior)**  
  Frozen DNABERT embeddings are used to encode sgRNA sequences, providing a contextualized sequence-level prior.

- **Stage-2 (Off-target Modeling)**  
  A CNN–Transformer hybrid network models local mismatches, positional effects, and long-range dependencies between sgRNA and off-target sequences. Stage-1 embeddings are fused at the classification stage.

An ablation setting without Stage-1 embeddings is provided to quantify the contribution of pretrained sequence priors.

## Datasets

- **Proxy dataset**  
  Used for model training and validation.  
  File: `Proxy_TrainCV.csv`

- **TrueOT dataset**  
  Used exclusively for out-of-distribution generalization evaluation.  
  File: `TrueOT_1806uniqueTriplet_gRNA_OT_label.csv`

No data leakage occurs between training and evaluation.

## Dataset Citation and Availability

This project uses the **TrueOT benchmark dataset**, a curated collection of experimentally validated CRISPR–Cas9 off-target sites introduced in the following study:

Park, S.-H., Kim, H. H., et al.  
*The Need for Transfer Learning in CRISPR–Cas Off-Target Scoring*.  
bioRxiv, 2021. https://doi.org/10.1101/2021.08.28.457846

The TrueOT benchmark is constructed by aggregating off-target measurements from multiple experimental techniques (e.g., GUIDE-seq, CIRCLE-seq, Digenome-seq) and is commonly used to evaluate model generalization performance.

The file `TrueOT_1806uniqueTriplet_gRNA_OT_label.csv` used in this repository corresponds to a **processed version** of the TrueOT benchmark derived from the sources described in the above publication.

The original unprocessed data sources are publicly available from the authors’ repository:
https://github.com/baolab-rice/CRISPR_OT_scoring/tree/master/TrueOT_unprocessed_data

Due to licensing and redistribution constraints, the processed dataset used in this work is **not redistributed** in this repository. Users should obtain the data from the original sources and place the files in the `data/` directory as described in the documentation.


## Model Variants

Two experimental settings are supported:

1. **Full model (Stage-1 + Stage-2)**  
   Uses frozen DNABERT embeddings during training and evaluation.

2. **Stage-2 only (ablation)**  
   Replaces Stage-1 embeddings with zero vectors of equal dimensionality.

The setting is controlled by the `USE_STAGE1` flag in `train.py` and `evaluate.py`.

## Results (TrueOT Generalization)

| Model Variant              | AUC  | AUPR |
|---------------------------|------|------|
| Stage-2 only              | ~0.64 | ~0.22 |
| Full model (Stage-1 + 2)  | 0.704 | 0.301 |

These results demonstrate that incorporating pretrained sequence priors significantly improves generalization to experimentally validated off-target sites.

## Running the Code

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the model
```bash
python -m stage2.train
```
### 3. Evaluate on TrueOT
```bash
python -m stage2.evaluate
```
To run the ablation study, set:
```bash
USE_STAGE1 = False
```
in both train.py and evaluate.py

## Notes

- Stage-1 (DNABERT) is never fine-tuned
- All random seeds are fixed
- Evaluation uses the same preprocessing pipeline as training
- This repository is a direct refactor of the original experimental implementation used to generate the reported results.

## License and Usage
This code is intended for academic and educational use. Please cite appropriately if used in derivative work.
