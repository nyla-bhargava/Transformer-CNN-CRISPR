import numpy as np
import os
import pandas as pd
from scipy import stats

# 1. Configuration
RESULTS_DIR = "results"
# "full" corresponds to Stage-1 Gated Model
METHODS = {"baseline": "Baseline", "full": "Stage-1 (Gated)"}
SEEDS = range(10) # N=10 for reliability

def calculate_cohen_d(group1, group2):
    """Calculates the Effect Size (Cohen's d)"""
    diff = np.array(group1) - np.array(group2)
    return np.mean(diff) / np.std(diff, ddof=1)

def get_95_ci(data):
    """Calculates the 95% Confidence Interval"""
    mean = np.mean(data)
    sem = stats.sem(data)
    # t-value for 95% CI with df = len(data)-1
    h = sem * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
    return mean, mean - h, mean + h

# 2. Data Aggregation
print("="*70)
print("   CRISPR OFF-TARGET PREDICTION: STATISTICAL REPORT")
print("="*70)

# Dictionary to store raw AUC/AUPR values for stats
raw_metrics = {m: {'auc': [], 'aupr': []} for m in METHODS}

for m_key in METHODS:
    for s in SEEDS:
        fname = f"{RESULTS_DIR}/{m_key}_seed{s}_results.npy"
        if os.path.exists(fname):
            # Load the results dictionary saved during evaluate.py
            res = np.load(fname, allow_pickle=True).item()
            raw_metrics[m_key]['auc'].append(res['auc'])
            raw_metrics[m_key]['aupr'].append(res['aupr'])

# Verify we have enough data
if not raw_metrics['baseline']['auc'] or not raw_metrics['full']['auc']:
    print(f"ERROR: No results found in {RESULTS_DIR}/. Run your training seeds first.")
    exit()

# 3. Performance and Reliability Table
print(f"{'MODEL CONFIGURATION':<20} | {'MEAN AUC ± SD':<18} | {'95% CONF. INTERVAL':<20} | {'RELIABILITY'}")
print("-" * 85)

for m_key, m_name in METHODS.items():
    aucs = raw_metrics[m_key]['auc']
    mean, sd = np.mean(aucs), np.std(aucs)
    m_val, low, high = get_95_ci(aucs)
    
    # Reliability % = (1 - CV) * 100
    reliability = (1 - (sd / mean)) * 100
    
    print(f"{m_name:<20} | {mean:.4f} ± {sd:.4f} | [{low:.3f} - {high:.3f}] | {reliability:.2f}%")

# 4. Ensemble Stability Analysis
print("\n" + "="*70)
print("   ENSEMBLE STABILITY ANALYSIS")
print("="*70)

s1_aucs = raw_metrics['full']['auc']
ens_mean = np.mean(s1_aucs)
# Ensemble variance is individual variance reduced by sqrt(N)
ens_sd = np.std(s1_aucs) / np.sqrt(len(s1_aucs))
ens_rel = (1 - (ens_sd / ens_mean)) * 100

print(f"Ensemble Mean ROC-AUC:    {ens_mean:.4f}")
print(f"Ensemble Stability (SD):  {ens_sd:.4f}")
print(f"Final Reliability Score:  {ens_rel:.2f}%")

# 5. Statistical Significance Testing
print("\n" + "="*70)
print("   P-VALUE & EFFECT SIZE (STAGE-1 vs BASELINE)")
print("="*70)

b_aucs = raw_metrics['baseline']['auc']
s_aucs = raw_metrics['full']['auc']

if len(b_aucs) == len(s_aucs):
    t_stat, p_val = stats.ttest_rel(s_aucs, b_aucs)
    d = calculate_cohen_d(s_aucs, b_aucs)
    
    print(f"Paired t-test p-value:   {p_val:.5f}")
    print(f"Cohen's d (Effect Size): {d:.2f}")
    
    if p_val < 0.05:
        print("RESULT: Statistically Significant (p < 0.05)")
    else:
        print("RESULT: Not Significant (Need more seeds or data)")
        
    if d > 0.8:
        print("EFFECT: Large Effect Size - The improvement is robust and consistent.")

print("="*70)
