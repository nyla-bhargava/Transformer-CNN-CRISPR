"""
Computes:
- ROC-AUC statistics
- Variance tests (Levene, F-test)
- Brier score
- Expected Calibration Error (ECE)
- Uncertainty–Error correlation (Spearman)
- Paired statistical tests
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss

# Configuration
RESULTS_DIR = "results"
N_SEEDS = 10
N_BINS = 15

# Helper Functions

def compute_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0

    for i in range(n_bins):
        idx = binids == i
        if np.sum(idx) > 0:
            acc = np.mean(y_true[idx])
            conf = np.mean(y_prob[idx])
            ece += np.abs(acc - conf) * np.sum(idx) / len(y_true)

    return ece


def load_seed(seed):
    b = np.load(
        f"{RESULTS_DIR}/baseline_seed{seed}_results.npy",
        allow_pickle=True
    ).item()

    s = np.load(
        f"{RESULTS_DIR}/full_seed{seed}_results.npy",
        allow_pickle=True
    ).item()

    return b, s

# Metric Containers

baseline_auc = []
stage1_auc = []

baseline_brier = []
stage1_brier = []

baseline_ece = []
stage1_ece = []

baseline_corr = []
stage1_corr = []

# Main Loop

for seed in range(N_SEEDS):

    b, s = load_seed(seed)

    # AUC
    baseline_auc.append(b["auc"])
    stage1_auc.append(s["auc"])

    # Brier
    baseline_brier.append(
        brier_score_loss(b["y_true"], b["mean_pred"].squeeze())
    )
    stage1_brier.append(
        brier_score_loss(s["y_true"], s["mean_pred"].squeeze())
    )

    # ECE
    baseline_ece.append(
        compute_ece(b["y_true"], b["mean_pred"].squeeze(), N_BINS)
    )
    stage1_ece.append(
        compute_ece(s["y_true"], s["mean_pred"].squeeze(), N_BINS)
    )

    # Uncertainty–Error Correlation
    error_b = np.abs(b["y_true"] - b["mean_pred"].squeeze())
    error_s = np.abs(s["y_true"] - s["mean_pred"].squeeze())

    uncert_b = b["std_pred"].squeeze()
    uncert_s = s["std_pred"].squeeze()

    baseline_corr.append(stats.spearmanr(error_b, uncert_b)[0])
    stage1_corr.append(stats.spearmanr(error_s, uncert_s)[0])


# Convert to arrays
baseline_auc = np.array(baseline_auc)
stage1_auc = np.array(stage1_auc)

baseline_brier = np.array(baseline_brier)
stage1_brier = np.array(stage1_brier)

baseline_ece = np.array(baseline_ece)
stage1_ece = np.array(stage1_ece)

baseline_corr = np.array(baseline_corr)
stage1_corr = np.array(stage1_corr)

# Statistical Tests

# AUC Paired t-test
auc_t, auc_p = stats.ttest_rel(baseline_auc, stage1_auc)

# Variance Tests
levene_stat, levene_p = stats.levene(baseline_auc, stage1_auc)

var_baseline = np.var(baseline_auc, ddof=1)
var_stage1 = np.var(stage1_auc, ddof=1)

f_stat = var_baseline / var_stage1
df1 = len(baseline_auc) - 1
df2 = len(stage1_auc) - 1
f_p = 1 - stats.f.cdf(f_stat, df1, df2)

# Brier Paired t-test
brier_t, brier_p = stats.ttest_rel(baseline_brier, stage1_brier)

# ECE Paired t-test
ece_t, ece_p = stats.ttest_rel(baseline_ece, stage1_ece)

# Correlation Paired t-test
corr_t, corr_p = stats.ttest_rel(baseline_corr, stage1_corr)

# Print Report

print("=" * 75)
print("CRISPR OFF-TARGET PREDICTION — STATISTICAL REPORT")
print("=" * 75)

print("\nROC-AUC:")
print(f"Baseline : {baseline_auc.mean():.4f} ± {baseline_auc.std():.4f}")
print(f"Stage-1  : {stage1_auc.mean():.4f} ± {stage1_auc.std():.4f}")
print(f"Paired t-test p-value: {auc_p:.5f}")

print("\nVariance Tests:")
print(f"Levene p-value: {levene_p:.5f}")
print(f"F-test p-value: {f_p:.5f}")

print("\nBrier Score:")
print(f"Baseline : {baseline_brier.mean():.5f}")
print(f"Stage-1  : {stage1_brier.mean():.5f}")
print(f"Paired t-test p-value: {brier_p:.5f}")

print("\nExpected Calibration Error (ECE):")
print(f"Baseline : {baseline_ece.mean():.5f}")
print(f"Stage-1  : {stage1_ece.mean():.5f}")
print(f"Paired t-test p-value: {ece_p:.5f}")

print("\nUncertainty–Error Correlation (Spearman ρ):")
print(f"Baseline : {baseline_corr.mean():.4f}")
print(f"Stage-1  : {stage1_corr.mean():.4f}")
print(f"Paired t-test p-value: {corr_p:.5f}")

print("=" * 75)
