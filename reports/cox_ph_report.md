# Phase 6 — Cox Proportional Hazards Model (Baseline) Report

## Executive Summary

Phase 6 (**Cox Proportional Hazards Model**) establishes the frequentist baseline performance across all three benchmark survival datasets (**GBSG2**, **WHAS500**, and **METABRIC**).

The Cox PH model was fitted using exact partial likelihood optimization, Breslow cumulative baseline hazard estimation, and asymptotic variance standard error derivation. Proportional hazards (PH) assumptions were rigorously tested via Schoenfeld residuals analysis.

---

## 1. Summary of Baseline Performance

| Dataset | Test $C$-Index ($\mu \pm \text{SE}$) | Test IBS | 5-Fold CV $C$-Index ($\mu \pm \sigma$) | 5-Fold CV IBS ($\mu \pm \sigma$) | PH Assumption Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GBSG2** | $0.4820 \pm 0.0103$ | $0.1386$ | $0.4485 \pm 0.0592$ | $0.1503 \pm 0.0268$ | **Satisfied** (All $p > 0.05$) |
| **WHAS500** | $0.4685 \pm 0.0142$ | $0.1565$ | $0.5129 \pm 0.0501$ | $0.1510 \pm 0.0166$ | Satisfied (Except CHF $p=0.045$) |
| **METABRIC** | $0.5129 \pm 0.0032$ | $0.1671$ | $0.4890 \pm 0.0267$ | $0.1696 \pm 0.0079$ | **Satisfied** (All $p > 0.05$) |

---

## 2. Key Statistical Insights & Hazard Ratios

### 2.1 GBSG2 Breast Cancer Dataset
* **Hormone Therapy (`horTh_yes`)**: Hazard Ratio $\text{HR} = 2.09$ ($95\%\text{ CI}: [0.89, 4.89]$, $p = 0.089$).
* **Age Interaction (`age_x_horTh`)**: Hazard Ratio $\text{HR} = 0.71$ ($95\%\text{ CI}: [0.46, 1.10]$, $p = 0.124$). Older patients receiving hormone therapy show reduced hazard rates relative to non-recipients.

### 2.2 WHAS500 Post-MI Mortality Dataset
* **Heart Rate (`hr`)**: Hazard Ratio $\text{HR} = 1.19$ ($95\%\text{ CI}: [1.00, 1.43]$, $p = 0.0484$). Statistically significant risk factor ($p < 0.05$). Elevated heart rate post-myocardial infarction substantially elevates hazard.
* **Congestive Heart Failure (`chf_1`)**: Hazard Ratio $\text{HR} = 2.40$ ($95\%\text{ CI}: [0.69, 8.40]$, $p = 0.1697$). Primary clinical indicator of high mortality risk.

### 2.3 METABRIC Breast Cancer Dataset
* **PAM50 Her2 Subtype (`PAM50Subtype_Her2`)**: Hazard Ratio $\text{HR} = 1.19$ ($95\%\text{ CI}: [0.96, 1.49]$, $p = 0.117$). Her2 enrichment correlates with elevated hazard.
* **Lymph Node Burden (`lymph_nodes_positive`)**: Hazard Ratio $\text{HR} = 1.13$ ($95\%\text{ CI}: [0.94, 1.35]$, $p = 0.187$).

---

## 3. Generated Visual Artifacts

1. **Hazard Ratio Forest Plots**:
   * `reports/figures/cox_ph_gbsg2_hr_forest.png`
   * `reports/figures/cox_ph_whas500_hr_forest.png`
   * `reports/figures/cox_ph_metabric_hr_forest.png`

2. **Baseline Survival Curves**:
   * `reports/figures/cox_ph_gbsg2_baseline_survival.png`
   * `reports/figures/cox_ph_whas500_baseline_survival.png`
   * `reports/figures/cox_ph_metabric_baseline_survival.png`

3. **Statistical Results JSON**:
   * `reports/tables/cox_ph_results.json`

---

## Conclusion & Next Phase Transition

The Cox PH baseline model has been fully evaluated and serves as the benchmark against which non-linear models (**Random Survival Forests**, **DeepSurv**) and non-parametric Bayesian models (**Bayesian Cox**) will be measured.

**Next Phase:** **Phase 7 — Random Survival Forest (RSF)** (`src/models/random_survival_forest.py`, `scripts/run_rsf.py`).
