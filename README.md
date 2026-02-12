# A Gated Multi-Stage Architecture for High-Reliability CRISPR Off-Target Prediction
This repository contains the official implementation of a **dual-stage deep learning framework** for CRISPR-Cas9 off-target prediction with a focus on **uncertainty reliability and probabilistic awareness**.

The framework integrates frozen DNABERT sequence embeddings with a CNN-Transformer hybrid architecture and evaluates not only predictive performance but also calibration, ensemble stability, and uncertainty-error alignment.

This implementation corresponds directly to the experiments and statistical analyses reported in the accompanying student research study.

## Project Overview

CRISPR off-target prediction remains challenging due to:

- Limited experimentally validated datasets
- Strong sequence-dependent behavior
- Poor uncertainty awareness in existing models

To address these challenges, we propose a **two-stage reliability-focused framework**.

### Stage-1: Sequence Prior (Frozen DNABERT)

- Uses pretrained DNABERT embeddings
- Encodes contextualized sgRNA sequence representations
- Embeddings are frozen (no fine-tuning)

### Stage-2: CNN–Transformer Hybrid

- CNN captures local mismatch patterns
- Transformer models long-range sequence dependencies
- Stage-1 embeddings are fused at classification stage

An ablation setting (Stage-2 only) is provided to isolate the contribution of pretrained sequence priors.

### Key Experimental Findings (TrueOT Benchmark)

Evaluation performed across **10 independent random seeds**.

| Metric | Baseline | Stage-1 + Stage-2 | p-value |
|--------|----------|------------------|---------|
| ROC-AUC | 0.7030 ± 0.0318 | 0.7040 ± 0.0288 | 0.841 |
| Brier Score | 0.12736 | 0.12750 | 0.908 |
| ECE | 0.09841 | 0.09879 | 0.922 |
| Uncertainty–Error Spearman ρ | 0.6652 | 0.7682 | **0.00395** |

### Interpretation

- Predictive performance remains statistically comparable.
- Calibration metrics remain similar.
- **Stage-1 significantly improves uncertainty–error alignment.**

This indicates stronger probabilistic awareness without sacrificing accuracy.

The contribution of this work lies in **improving reliability rather than raw performance gains**.

## Datasets
### Proxy Dataset
Used for training and validation  
File: `Proxy_TrainCV.csv`

### TrueOT Benchmark
Used exclusively for out-of-distribution evaluation  
File: `TrueOT_1806uniqueTriplet_gRNA_OT_label.csv`

### Citation

Park, S.-H., Kim, H. H., et al.  
**The Need for Transfer Learning in CRISPR–Cas Off-Target Scoring**  
bioRxiv, 2021  
https://doi.org/10.1101/2021.08.28.457846

The TrueOT dataset aggregates experimentally validated off-target sites from GUIDE-seq, CIRCLE-seq, Digenome-seq, and related assays.

Due to redistribution constraints, users must obtain the dataset from the google drive given in data/ and place it in the `data/` directory.

## Installation
### Clone repository
1️⃣ Clone Repository
```bash
git clone https://https://github.com/nyla-bhargava/Transformer-CNN-CRISPR.git
cd Transformer-CNN-CRISPR
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Training
Baseline (Stage-2 Only):
```bash
python -m stage2.train --seed 0
```

Full Model (Stage-1 + Stage-2):
```bash
python -m stage2.train --seed 0 --use_stage1
```

### Evaluation
Baseline (Stage-2 Only):
```bash
python -m stage2.evaluate --seed 0
```

Full Model (Stage-1 + Stage-2):
```bash
python -m stage2.evaluate --seed 0 --use_stage1
```

### Run All 10 Seeds
```bash
scripts/run_10_seeds.sh
```

### Statistical Analysis
```bash
scripts/benchmark_stats_1.py
scripts/benchmark_stats_2.py
scripts/benchmark_stats_3.py
```
This generates:
- Statistical summaries
- CSV files
- Reliability metrics
- Uncertainty–error correlation analysis

### Deterministic GPU Execution (Important)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Contribution Summary

This work demonstrates that:

- Pretrained sequence priors do not significantly change mean ROC-AUC.
- They significantly improve uncertainty awareness.
- Ensemble aggregation enhances deployment stability.

The framework emphasizes **reliability, uncertainty alignment, and probabilistic robustness** for CRISPR off-target modeling.

# License

This repository is intended for academic and educational use.

Please cite appropriately if used in derivative research.

# Author
Nyla Bhargava

Student Research Project  
Biomedical Engineering, SRM University
