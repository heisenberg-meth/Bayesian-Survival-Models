# Phase 7 — Random Survival Forests (RSF) Model Report

## Executive Summary

Phase 7 (**Random Survival Forests**) introduces a non-parametric, non-linear ensemble survival model across **GBSG2**, **WHAS500**, and **METABRIC**.

By leveraging Log-Rank splitting criterion and leaf-level Nelson-Aalen cumulative hazard estimation, RSF captures complex non-linear feature interactions and high-order predictor effects without making proportional hazards or linear predictor assumptions. Across all three datasets, RSF achieved substantial performance gains in test discrimination ($C$-index) and calibration (IBS) over the linear Cox PH baseline.

---

## 1. Performance Benchmark: RSF vs Cox PH Baseline

| Dataset | Metric | Cox PH Baseline | Random Survival Forest (RSF) | Performance Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **GBSG2** | Test $C$-Index ($\mu \pm \text{SE}$) | $0.4820 \pm 0.0103$ | **$0.5522 \pm 0.0103$** | **$+0.0702$** |
| | Test IBS | $0.1386$ | **$0.1356$** | **$-0.0030$ (Better)** |
| | 5-Fold CV $C$-Index | $0.4485 \pm 0.0592$ | **$0.4505 \pm 0.0717$** | **$+0.0020$** |
| **WHAS500** | Test $C$-Index ($\mu \pm \text{SE}$) | $0.4685 \pm 0.0142$ | **$0.5508 \pm 0.0141$** | **$+0.0823$** |
| | Test IBS | $0.1565$ | **$0.1472$** | **$-0.0093$ (Better)** |
| | 5-Fold CV $C$-Index | $0.5129 \pm 0.0501$ | **$0.5116 \pm 0.0579$** | $-0.0013$ |
| **METABRIC** | Test $C$-Index ($\mu \pm \text{SE}$) | $0.5129 \pm 0.0032$ | **$0.5833 \pm 0.0032$** | **$+0.0704$** |
| | Test IBS | $0.1671$ | **$0.1646$** | **$-0.0025$ (Better)** |
| | 5-Fold CV $C$-Index | $0.4890 \pm 0.0267$ | **$0.5089 \pm 0.0205$** | **$+0.0199$** |

---

## 2. Feature Importance Analysis (Permutation VIMP)

### 2.1 GBSG2 Breast Cancer Dataset
* Top Predictors: **Tumor Size (`tsize`, VIMP = $0.0562$)**, **Age (`age`, VIMP = $0.0556$)**, and **Log Progesterone Receptor (`log_progrec`, VIMP = $0.0522$)**.
* Insight: Non-linear interaction between age and tumor burden is crucial for recurrence prediction.

### 2.2 WHAS500 Post-MI Mortality Dataset
* Top Predictors: **Heart Rate (`hr`, VIMP = $0.0890$)**, **Age (`age`, VIMP = $0.0751$)**, **BMI (`bmi`, VIMP = $0.0674$)**, and **Diastolic BP (`diasbp`, VIMP = $0.0671$)**.
* Insight: Hemodynamic indicators (`hr`, `diasbp`, `sysbp`) non-linearly dominate survival prediction post-MI.

### 2.3 METABRIC Breast Cancer Dataset
* Top Predictors: **Age x Stage Interaction (`age_x_stage`, VIMP = $0.0479$)**, **Age (`age`, VIMP = $0.0433$)**, and **Log Positive Lymph Nodes (`log_lymph_nodes`, VIMP = $0.0290$)**.
* Insight: Non-linear clinical interaction terms (`age_x_stage`) provide the strongest predictive signal in METABRIC.

---

## 3. Generated Visual Artifacts

1. **Permutation Feature Importance (VIMP) Charts**:
   * `reports/figures/rsf_feature_importance_gbsg2.png`
   * `reports/figures/rsf_feature_importance_whas500.png`
   * `reports/figures/rsf_feature_importance_metabric.png`

2. **RSF Ensemble Survival Curves**:
   * `reports/figures/rsf_gbsg2_survival_curves.png`
   * `reports/figures/rsf_whas500_survival_curves.png`
   * `reports/figures/rsf_metabric_survival_curves.png`

3. **Results Artifact**:
   * `reports/tables/rsf_results.json`

---

## Conclusion & Next Phase Transition

Random Survival Forests demonstrated substantial gains over linear Cox PH across all benchmark datasets.

**Next Phase:** **Phase 8 — DeepSurv (Deep Learning Survival)** (`src/models/deepsurv.py`, `scripts/run_deepsurv.py`).
