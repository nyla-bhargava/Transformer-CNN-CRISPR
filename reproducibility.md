# `reproducibility.md`

This document describes the steps and conditions required to reproduce the results reported in this repository.

## Environment

- Python version: 3.9+
- PyTorch: latest stable (tested on CUDA and CPU)
- Transformers: 4.45.2
- Hardware used for original experiments:
  - Google Colab (NVIDIA T4 GPU)
  - Experiments are reproducible on CPU with longer runtime

All required packages are listed in `requirements.txt`.

## Randomness Control

To ensure deterministic behavior:

- Python, NumPy, and PyTorch random seeds are fixed (`seed = 42`)
- CUDA deterministic mode is enabled
- cuDNN benchmarking is disabled

The seed configuration is implemented in `utils/seed.py` and invoked at the beginning of training and evaluation.

## Data Splits

- The Proxy dataset is split into training and validation sets using:
  - Validation fraction: 15%
  - Stratification on class labels
  - Fixed random state (`random_state = 42`)

- The TrueOT dataset is **never used during training or validation** and is reserved exclusively for evaluation.

## Model Configuration

- Optimizer: AdamW  
- Learning rate: 2e-4  
- Weight decay: 1e-4  
- Epochs: 30  
- Batch size:
  - Training: 64
  - Validation / Test: 128

All hyperparameters are fixed and not tuned post hoc.

## Evaluation Protocol

- Metrics:
  - Area Under the ROC Curve (AUC)
  - Area Under the Precision–Recall Curve (AUPR)

- Evaluation is performed using:
  - Deterministic forward pass
  - Monte Carlo dropout (T = 30) for uncertainty estimation

ROC and Precision–Recall curves are saved in the `results/` directory.

## Reproduction Steps

1. Clone the repository
2. Install dependencies
3. Place datasets in the `data/` directory
4. Run:
   ```bash
   python stage2/train.py
   python stage2/evaluate.py
 ```
