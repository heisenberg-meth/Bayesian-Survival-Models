# Phase 5 — Model Evaluation Framework Architecture Report

## Executive Summary

Phase 5 (**Model Evaluation Framework**) establishes a unified, publication-grade evaluation suite for comparing the survival modeling performance of **Cox Proportional Hazards**, **Bayesian Cox**, **Random Survival Forests (RSF)**, and **DeepSurv**.

By centralizing metric calculations and evaluation protocols, this framework guarantees that all four models are benchmarked under identical mathematical definitions, cross-validation splits, and time-dependent horizons.

---

## 1. Unified Evaluation Metrics

### 1.1 Harrell's Concordance Index ($C$-index)
* **Definition**: Measures the proportion of all comparable patient pairs where the model correctly predicts a higher risk score ($\hat{\eta}_i > \hat{\eta}_j$) for the patient with shorter survival time ($T_i < T_j$).
* **Implementation**: `concordance_index()` in `src/evaluation/metrics.py`.
* **Standard Error**: Computed via asymptotic variance approximation:
  $$\text{SE}(C) = \sqrt{\frac{C(1 - C)}{N_{\text{pairs}}}}$$

---

### 1.2 Kaplan-Meier IPCW Censoring Distribution
* **Definition**: Models the censoring survival distribution $G(t) = P(C > t)$ by fitting a Kaplan-Meier estimator on the inverted event indicator $1 - E$.
* **Implementation**: `KaplanMeierCensoringEstimator` in `src/evaluation/metrics.py`.
* **Purpose**: Provides unbiased Inverse Probability of Censoring Weights (IPCW) for Brier Score and calibration calculations in the presence of right censoring.

---

### 1.3 Time-Dependent Brier Score ($BS(t)$)
* **Definition**: Evaluates survival probability prediction accuracy $S(t | X_i)$ at a specific time milestone $t$, adjusted for IPCW weights $G(t)$:
  $$BS(t) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{(0 - \hat{S}(t|X_i))^2 \cdot \mathbb{I}(T_i \le t, E_i = 1)}{\hat{G}(T_i-)} + \frac{(1 - \hat{S}(t|X_i))^2 \cdot \mathbb{I}(T_i > t)}{\hat{G}(t)} \right]$$
* **Implementation**: `brier_score_at_time()` in `src/evaluation/metrics.py`.

---

### 1.4 Integrated Brier Score (IBS)
* **Definition**: Integrates the time-dependent Brier Score across the evaluation interval $[t_{\min}, t_{\max}]$ using numerical trapezoidal integration:
  $$\text{IBS} = \frac{1}{t_{\max} - t_{\min}} \int_{t_{\min}}^{t_{\max}} BS(t) \, dt$$
* **Implementation**: `integrated_brier_score()` in `src/evaluation/metrics.py`.

---

## 2. Cross-Validation Evaluator (`src/evaluation/cross_validation.py`)

* **Protocol**: Executes 5-fold cross-validation across pre-generated stratified splits (`cv_folds.json`).
* **Outputs**: Computes cross-validation mean ($\mu$) and standard deviation ($\sigma$) for both $C$-index and Integrated Brier Score.

---

## 3. Metric Comparison Protocol Across Models

| Metric | Target Ideal | Meaning | Evaluated At |
| :--- | :---: | :--- | :--- |
| **$C$-Index** | $1.0$ (Random = 0.5) | Relative risk discrimination | Full Test & 5-Fold CV |
| **Brier Score $BS(t)$** | $0.0$ | Time-specific probability calibration error | $t_{25}, t_{50}, t_{75}$ percentiles |
| **Integrated Brier (IBS)** | $0.0$ | Overall time-integrated probability accuracy | Full evaluation window |

---

## Conclusion & Readiness for Model Implementations

The evaluation framework is fully operational and validated via `scripts/test_evaluation_framework.py`.

**Next Phase:** **Phase 6 — Cox Proportional Hazards Model** (`src/models/cox_ph.py`, `scripts/run_cox_ph.py`).
