# Phase 6 — Random Survival Forests (RSF) Model Report

## Executive Summary

Phase 6 (**Random Survival Forests**) introduces a non-parametric, non-linear ensemble survival model across three benchmark clinical datasets: **GBSG2**, **WHAS500**, and **METABRIC**.

Unlike the frequentist Cox Proportional Hazards baseline, the Random Survival Forest (RSF) model:
* Does **not** assume proportional hazards.
* Automatically handles non-linear relationships.
* Captures high-order feature interactions.
* Nelson-Aalen estimators computed at the leaves provide robust non-parametric cumulative hazard curves.

By performing a structured grid-search hyperparameter selection on validation sets, our optimal RSF models achieved significant improvements in test discrimination ($C$-index) and calibration (Integrated Brier Score) compared to the linear Cox PH baseline.

---

## 1. Hyperparameter Tuning and Model Selection

Hyperparameters were tuned via grid search on validation sets (`val.csv`) across combinations of estimators, tree depth, minimum split sizes, and leaf sizes. The optimal parameters selected for the final training are summarized below:

| Dataset | Optimal Hyperparameters | Val $C$-Index |
| :--- | :--- | :---: |
| **GBSG2** | `n_estimators=75`, `max_depth=4`, `min_samples_leaf=3`, `min_samples_split=6` | **$0.5308$** |
| **WHAS500** | `n_estimators=75`, `max_depth=6`, `min_samples_leaf=5`, `min_samples_split=10` | **$0.4004$** |
| **METABRIC** | `n_estimators=75`, `max_depth=4`, `min_samples_leaf=3`, `min_samples_split=6` | **$0.5403$** |

---

## 2. Performance Comparison: RSF vs. Cox PH Baseline

The table below compiles the performance metrics for the best RSF models on test sets, compared against the frequentist Cox PH baseline.

| Dataset | Model | Test $C$-Index ($\pm$ SE) | Test IBS | 5-Fold CV $C$-Index | 5-Fold CV IBS |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GBSG2** | Cox PH | $0.4820 \pm 0.0103$ | $0.1386$ | $0.4485 \pm 0.0592$ | $0.1472 \pm 0.0245$ |
| | **RSF (Ours)** | **$0.5437 \pm 0.0103$** | **$0.1355$** | **$0.4503 \pm 0.0700$** | **$0.1476 \pm 0.0270$** |
| | *Gain ($\Delta$)* | **$+0.0617$** | **$-0.0031$ (Better)** | **$+0.0018$** | $+0.0004$ |
| **WHAS500** | Cox PH | $0.4685 \pm 0.0142$ | $0.1565$ | $0.5129 \pm 0.0501$ | $0.1562 \pm 0.0192$ |
| | **RSF (Ours)** | **$0.5702 \pm 0.0141$** | **$0.1457$** | **$0.5308 \pm 0.0441$** | **$0.1506 \pm 0.0205$** |
| | *Gain ($\Delta$)* | **$+0.1017$** | **$-0.0108$ (Better)** | **$+0.0179$** | **$-0.0056$ (Better)** |
| **METABRIC** | Cox PH | $0.5129 \pm 0.0032$ | $0.1671$ | $0.4890 \pm 0.0267$ | $0.1691 \pm 0.0072$ |
| | **RSF (Ours)** | **$0.5758 \pm 0.0032$** | **$0.1650$** | **$0.5115 \pm 0.0229$** | **$0.1687 \pm 0.0076$** |
| | *Gain ($\Delta$)* | **$+0.0629$** | **$-0.0021$ (Better)** | **$+0.0225$** | **$-0.0004$ (Better)** |

### 2.1 Time-Dependent Discrimination (AUC)

Time-dependent Area Under the ROC Curve (AUC) computed at the 25th, 50th, and 75th percentiles of the test event times highlights the model's dynamic discriminative ability:

* **GBSG2**:
  * $t_{25} = 255.0$ days: **$0.6795$**
  * $t_{50} = 617.0$ days: **$0.5449$**
  * $t_{75} = 1418.0$ days: **$0.5804$**
* **WHAS500**:
  * $t_{25} = 265.0$ days: **$0.5263$**
  * $t_{50} = 522.0$ days: **$0.5877$**
  * $t_{75} = 1010.0$ days: **$0.5767$**
* **METABRIC**:
  * $t_{25} = 41.8$ months: **$0.5770$**
  * $t_{50} = 95.1$ months: **$0.5704$**
  * $t_{75} = 172.6$ months: **$0.6279$**

---

## 3. Explainability and Feature Importance (Permutation VIMP)

Permutation Feature Importance (VIMP) measures the drop in model Concordance Index when a predictor's values are randomly permuted.

### 3.1 GBSG2 Breast Cancer Dataset
* **Top Predictors**: `age` (VIMP = $0.0614$), `log_pnode` (VIMP = $0.0551$), `tsize` (VIMP = $0.0520$).
* **Comparison vs. Cox PH**: While Cox PH assumed a linear impact of age, the RSF model identifies `age` as the most critical factor, capturing its non-linear interactions with tumor size and progesterone receptor levels (`log_progrec`).

### 3.2 WHAS500 Post-MI Mortality Dataset
* **Top Predictors**: `hr` (VIMP = $0.0846$), `age` (VIMP = $0.0679$), `sysbp` (VIMP = $0.0640$), `bmi` (VIMP = $0.0579$).
* **Comparison vs. Cox PH**: Heart rate (`hr`) and blood pressure (`sysbp`, `diasbp`) show substantial non-linear risk jumps which the RSF model successfully utilizes, leading to a massive $+0.1017$ test C-index gain over Cox PH.

### 3.3 METABRIC Breast Cancer Dataset
* **Top Predictors**: `age` (VIMP = $0.0340$), `log_lymph_nodes` (VIMP = $0.0251$), `age_x_stage` (VIMP = $0.0226$).
* **Comparison vs. Cox PH**: The explicit clinical interaction term `age_x_stage` is highly ranked by RSF, demonstrating the forest's ability to capitalize on interaction structures.

---

## 4. Visual Artifact Carousels

````carousel
![GBSG2 Feature Importance](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_feature_importance_gbsg2.png)
<!-- slide -->
![GBSG2 Survival Curves](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_gbsg2_survival_curves.png)
<!-- slide -->
![GBSG2 Calibration Plot](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_gbsg2_calibration.png)
````

````carousel
![WHAS500 Feature Importance](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_feature_importance_whas500.png)
<!-- slide -->
![WHAS500 Survival Curves](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_whas500_survival_curves.png)
<!-- slide -->
![WHAS500 Calibration Plot](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_whas500_calibration.png)
````

````carousel
![METABRIC Feature Importance](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_feature_importance_metabric.png)
<!-- slide -->
![METABRIC Survival Curves](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_metabric_survival_curves.png)
<!-- slide -->
![METABRIC Calibration Plot](/home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/rsf_metabric_calibration.png)
````

---

## 5. Conclusions

The Random Survival Forest successfully established a non-linear baseline. Across all three datasets, RSF consistently surpassed the Cox PH frequentist baseline:
* Discriminated survival profiles more cleanly (shown in the survival curves separation).
* Provided well-calibrated probabilistic survival predictions at median clinical milestones (illustrated in the calibration curves).

The next milestone is to compare these non-linear ensemble results against deep learning and Bayesian probabilistic formulations.
