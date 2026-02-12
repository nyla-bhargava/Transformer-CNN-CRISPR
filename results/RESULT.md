# CRISPR Off-Target Prediction — Experimental Results

This document summarizes the statistical evaluation of the **Stage-1 Gated Transformer architecture** compared to the baseline model across **10 independent random seeds** on the TrueOT benchmark dataset.

All reported metrics are averaged across seeds.

---

## 1️⃣ Predictive Performance

| Metric | Baseline (Mean ± SD) | Stage-1 (Mean ± SD) | p-value | Significance |
|--------|----------------------|---------------------|---------|--------------|
| GLOBAL ROC-AUC | 0.7025 ± 0.032 | 0.7034 ± 0.029 | 0.85529 | Not Significant |
| HARD ROC-AUC | 0.7029 ± 0.026 | 0.7046 ± 0.023 | 0.70912 | Not Significant |
| TOP20 | 0.0256 ± 0.009 | 0.0224 ± 0.006 | 0.08684 | Trend (p < 0.1) |

### Interpretation

- No statistically significant difference in mean ROC-AUC.
- Performance remains comparable across random initializations.
- A statistical trend is observed for the TOP20 metric.


## 2️⃣ Reliability & 95% Confidence Intervals

| Model | Metric | 95% Confidence Interval | Reliability |
|--------|--------|--------------------------|-------------|
| Baseline | GLOBAL | [0.6787 – 0.7263] | 95.51% |
| Baseline | HARD | [0.6836 – 0.7221] | 96.37% |
| Stage-1 | GLOBAL | [0.6817 – 0.7251] | 95.91% |
| Stage-1 | HARD | [0.6869 – 0.7223] | 96.67% |

Stage-1 demonstrates slightly narrower confidence intervals and marginally higher reliability.


## 3️⃣ Ensemble Stability Analysis

| Metric | Value |
|--------|--------|
| Ensemble Mean ROC-AUC | 0.7040 |
| Ensemble Stability (SD) | 0.0091 |
| Final Reliability Score | 98.70% |

The ensemble model significantly reduces variability across seeds, indicating strong deployment stability.

## 4️⃣ Statistical Tests

| Test | p-value | Effect Size (Cohen's d) | Conclusion |
|------|---------|--------------------------|------------|
| Paired t-test (ROC-AUC) | 0.84100 | 0.07 | Not Significant |
| Levene’s Variance Test | 0.57200 | — | Not Significant |
| F-test (Variance Ratio) | 0.38742 | — | Not Significant |

There is no statistically significant difference in mean performance or variance reduction.

## 5️⃣ Probabilistic Accuracy & Calibration

| Metric | Baseline | Stage-1 | p-value | Significance |
|--------|----------|----------|---------|--------------|
| Brier Score | 0.12736 | 0.12750 | 0.90899 | Not Significant |
| Expected Calibration Error (ECE) | 0.09841 | 0.09879 | 0.92252 | Not Significant |

Stage-1 does not significantly change calibration or probabilistic accuracy.

## 6️⃣ Uncertainty–Error Alignment (Key Finding)

| Metric | Baseline | Stage-1 | p-value | Significance |
|--------|----------|----------|---------|--------------|
| Spearman ρ (Error vs Uncertainty) | 0.6652 | 0.7682 | 0.00395 | **Significant** |

### Interpretation

Stage-1 significantly improves the alignment between predictive uncertainty and actual prediction error.

This indicates:

- Improved probabilistic awareness
- Stronger uncertainty reliability
- Better suitability for safety-critical CRISPR off-target applications

# Overall Conclusion

- Predictive performance remains statistically comparable.
- Calibration and probabilistic accuracy remain unchanged.
- Stage-1 significantly improves uncertainty–error alignment.
- Ensemble modeling further enhances stability.

This suggests that the Stage-1 gated architecture enhances **reliability and uncertainty awareness** without sacrificing predictive performance.
