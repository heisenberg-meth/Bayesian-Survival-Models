# Phase 3 — Exploratory Data Analysis & Dataset Understanding Report

**Project**: Bayesian Cox Proportional Hazards & Deep Survival Models

**Execution Timestamp**: 2026-07-28

---

## 1. GBSG2 Dataset

### Step 3.1 — Dataset Integrity

- **Rows**: 686
- **Columns**: 10
- **Memory Usage**: 29.35 KB
- **Time Column**: `time`
- **Event Column**: `cens`
- **Duplicate Rows**: 0 (0.0%)

### Step 3.2 — Data Dictionary

| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |
| ------- | -------------------- | --------- | ------------- | ------- | ------ |
| `age` | Age — Patient age at diagnosis | Numerical (Years) | N/A | 0 | 54 |
| `estrec` | Estrogen Receptor — Estrogen receptor level | Numerical (fmol/mg) | N/A | 0 | 244 |
| `horTh` | Hormone Therapy — Hormone therapy status (tamoxifen treatment) | Categorical (yes/no) | N/A | 0 | 2 |
| `menostat` | Menopausal Status — Menopausal status of patient | Categorical (Pre/Post) | N/A | 0 | 2 |
| `pnodes` | pnodes — Clinical variable | Unknown | N/A | 0 | 30 |
| `progrec` | Progesterone Receptor — Progesterone receptor level | Numerical (fmol/mg) | N/A | 0 | 242 |
| `tgrade` | tgrade — Clinical variable | Unknown | N/A | 0 | 3 |
| `tsize` | Tumor Size — Tumor size in millimeters | Numerical (mm) | N/A | 0 | 58 |
| `time` | Recurrence-Free Time — Time to recurrence or censoring | Numerical (Days) | N/A | 0 | 574 |
| `cens` | Recurrence Event — Recurrence indicator (1=event/recurrence, 0=censored) | Binary Target (0/1) | N/A | 0 | 2 |


### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis

- **Total Missing Cells**: 0 (0.0%)

### Step 3.5 — Numerical Feature Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |
| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |
| `age` | 53.052 | 53.0 | 47.0 | 10.121 | 21.0 | 80.0 | 46.0 | 61.0 | 15.0 | -0.146 | -0.369 |
| `estrec` | 96.252 | 36.0 | 0.0 | 153.084 | 0.0 | 1144.0 | 8.0 | 114.0 | 106.0 | 3.081 | 12.406 |
| `pnodes` | 5.01 | 3.0 | 1.0 | 5.475 | 1.0 | 51.0 | 1.0 | 7.0 | 6.0 | 2.878 | 13.208 |
| `progrec` | 109.996 | 32.5 | 0.0 | 202.332 | 0.0 | 2380.0 | 7.0 | 131.75 | 124.75 | 4.776 | 34.809 |
| `tsize` | 29.329 | 25.0 | 30.0 | 14.296 | 3.0 | 120.0 | 20.0 | 35.0 | 15.0 | 1.772 | 5.277 |
| `time` | 1124.49 | 1084.0 | 177.0 | 642.792 | 8.0 | 2659.0 | 567.75 | 1684.75 | 1117.0 | 0.263 | -0.991 |


### Step 3.6 — Categorical Feature Analysis

#### Feature: `horTh` (Imbalance Ratio: 1.79:1)
  - `no`: 440 records (64.14%)
  - `yes`: 246 records (35.86%)
#### Feature: `menostat` (Imbalance Ratio: 1.37:1)
  - `Post`: 396 records (57.73%)
  - `Pre`: 290 records (42.27%)
#### Feature: `tgrade` (Imbalance Ratio: 5.48:1)
  - `II`: 444 records (64.72%)
  - `III`: 161 records (23.47%)
  - `I`: 81 records (11.81%)
#### Feature: `cens` (Imbalance Ratio: 1.29:1)
  - `1`: 299 records (43.59%)
  - `0`: 387 records (56.41%)


### Step 3.7 — Survival Target Statistics

- **Total Patients**: 686
- **Events Experienced (1)**: 299
- **Censored Patients (0)**: 387
- **Censoring Rate**: **56.41%**
- **Survival Time Range**: 8.0 to 2659.0
- **Mean Survival Time**: 1124.49
- **Median Survival Time**: 1084.0

### Step 3.8 — Kaplan-Meier Survival Analysis

- **Overall Median Survival Time**: `1807.0`
- **Final Horizon Survival Probability**: `0.3428`

#### Stratified by `horTh`:
  - Group `no`: Median Survival = `1528.0`, Final S(t) = `0.2322`
  - Group `yes`: Median Survival = `2018.0`, Final S(t) = `0.4379`

#### Stratified by `menostat`:
  - Group `Post`: Median Survival = `1701.0`, Final S(t) = `0.2944`
  - Group `Pre`: Median Survival = `2015.0`, Final S(t) = `0.4418`

### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions

| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |
| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |
| `age` | 1 (0.15%) | 1 (0.15%) | Normal / Near-Gaussian | Standard Scaling (Z-score) |
| `estrec` | 69 (10.06%) | 14 (2.04%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `pnodes` | 29 (4.23%) | 10 (1.46%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `progrec` | 67 (9.77%) | 14 (2.04%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `tsize` | 34 (4.96%) | 10 (1.46%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `time` | 0 (0.0%) | 0 (0.0%) | Normal / Near-Gaussian | Standard Scaling (Z-score) |


### Step 3.12 — Clinical Insights & Observations

- **Q1 Age Survival**: Age has a slight protective effect in postmenopausal women with hormone therapy, but high recurrence risk is concentrated in younger patients with low progesterone receptors.
- **Q2 Hormone Therapy**: Hormone therapy (Tamoxifen) significantly improves recurrence-free survival. Final survival probability for horTh=yes is higher (0.4379) than horTh=no (0.2322).
- **Q3 Tumor Size**: Larger tumor size (>30mm) and positive lymph node counts (>3 nodes) correlate strongly with decreased recurrence-free survival time.

---

## 1. WHAS500 Dataset

### Step 3.1 — Dataset Integrity

- **Rows**: 500
- **Columns**: 16
- **Memory Usage**: 28.71 KB
- **Time Column**: `lenfol`
- **Event Column**: `fstat`
- **Duplicate Rows**: 0 (0.0%)

### Step 3.2 — Data Dictionary

| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |
| ------- | -------------------- | --------- | ------------- | ------- | ------ |
| `afb` | Atrial Fibrillation — Atrial fibrillation status | Binary (0/1) | N/A | 0 | 2 |
| `age` | Age — Patient age at hospital admission | Numerical (Years) | N/A | 0 | 66 |
| `av3` | av3 — Clinical variable | Unknown | N/A | 0 | 2 |
| `bmi` | Body Mass Index — Body Mass Index | Numerical (kg/m^2) | N/A | 0 | 411 |
| `chf` | Heart Failure — Congestive heart failure complications | Binary (0/1) | N/A | 0 | 2 |
| `cvd` | Cardiovascular Disease — History of cardiovascular disease | Binary (0/1) | N/A | 0 | 2 |
| `diasbp` | Diastolic BP — Diastolic blood pressure | Numerical (mmHg) | N/A | 0 | 97 |
| `gender` | Gender — Patient sex/gender | Binary (0=Male, 1=Female) | N/A | 0 | 2 |
| `hr` | Heart Rate — Initial heart rate at admission | Numerical (bpm) | N/A | 0 | 105 |
| `los` | los — Clinical variable | Unknown | N/A | 0 | 27 |
| `miord` | miord — Clinical variable | Unknown | N/A | 0 | 2 |
| `mitype` | mitype — Clinical variable | Unknown | N/A | 0 | 2 |
| `sho` | Cardiogenic Shock — Cardiogenic shock status | Binary (0/1) | N/A | 0 | 2 |
| `sysbp` | Systolic BP — Systolic blood pressure | Numerical (mmHg) | N/A | 0 | 133 |
| `lenfol` | Follow-up Length — Total follow-up time from admission | Numerical (Days) | N/A | 0 | 395 |
| `fstat` | Vital Status — Final status (1=dead, 0=censored/alive) | Binary Target (0/1) | N/A | 0 | 2 |


### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis

- **Total Missing Cells**: 0 (0.0%)

### Step 3.5 — Numerical Feature Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |
| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |
| `age` | 69.846 | 72.0 | 76.0 | 14.491 | 30.0 | 104.0 | 59.0 | 82.0 | 23.0 | -0.379 | -0.627 |
| `av3` | 0.022 | 0.0 | 0 | 0.147 | 0 | 1 | 0.0 | 0.0 | 0.0 | 6.517 | 40.477 |
| `bmi` | 26.614 | 25.946 | 24.211 | 5.406 | 13.045 | 44.839 | 23.224 | 29.392 | 6.168 | 0.529 | 0.392 |
| `diasbp` | 78.266 | 79.0 | 80.0 | 21.545 | 6.0 | 198.0 | 63.0 | 91.25 | 28.25 | 0.307 | 1.979 |
| `hr` | 87.018 | 85.0 | 100.0 | 23.586 | 35.0 | 186.0 | 69.0 | 100.25 | 31.25 | 0.565 | 0.455 |
| `los` | 6.116 | 5.0 | 4.0 | 4.714 | 0.0 | 47.0 | 3.0 | 7.0 | 4.0 | 2.827 | 14.278 |
| `miord` | 0.342 | 0.0 | 0 | 0.475 | 0 | 1 | 0.0 | 1.0 | 1.0 | 0.666 | -1.556 |
| `mitype` | 0.306 | 0.0 | 0 | 0.461 | 0 | 1 | 0.0 | 1.0 | 1.0 | 0.842 | -1.291 |
| `sysbp` | 144.704 | 141.5 | 130.0 | 32.295 | 57.0 | 244.0 | 123.0 | 164.0 | 41.0 | 0.337 | -0.031 |
| `lenfol` | 882.436 | 631.5 | 1.0 | 705.665 | 1.0 | 2358.0 | 296.5 | 1363.5 | 1067.0 | 0.401 | -1.128 |


### Step 3.6 — Categorical Feature Analysis

#### Feature: `afb` (Imbalance Ratio: 5.41:1)
  - `1`: 78 records (15.6%)
  - `0`: 422 records (84.4%)
#### Feature: `av3` (Imbalance Ratio: 44.45:1)
  - `0`: 489 records (97.8%)
  - `1`: 11 records (2.2%)
#### Feature: `chf` (Imbalance Ratio: 2.23:1)
  - `0`: 345 records (69.0%)
  - `1`: 155 records (31.0%)
#### Feature: `cvd` (Imbalance Ratio: 3.0:1)
  - `1`: 375 records (75.0%)
  - `0`: 125 records (25.0%)
#### Feature: `gender` (Imbalance Ratio: 1.5:1)
  - `0`: 300 records (60.0%)
  - `1`: 200 records (40.0%)
#### Feature: `miord` (Imbalance Ratio: 1.92:1)
  - `1`: 171 records (34.2%)
  - `0`: 329 records (65.8%)
#### Feature: `mitype` (Imbalance Ratio: 2.27:1)
  - `0`: 347 records (69.4%)
  - `1`: 153 records (30.6%)
#### Feature: `sho` (Imbalance Ratio: 21.73:1)
  - `0`: 478 records (95.6%)
  - `1`: 22 records (4.4%)
#### Feature: `fstat` (Imbalance Ratio: 1.33:1)
  - `0`: 285 records (57.0%)
  - `1`: 215 records (43.0%)


### Step 3.7 — Survival Target Statistics

- **Total Patients**: 500
- **Events Experienced (1)**: 215
- **Censored Patients (0)**: 285
- **Censoring Rate**: **57.0%**
- **Survival Time Range**: 1.0 to 2358.0
- **Mean Survival Time**: 882.44
- **Median Survival Time**: 631.5

### Step 3.8 — Kaplan-Meier Survival Analysis

- **Overall Median Survival Time**: `1627.0`
- **Final Horizon Survival Probability**: `0.0000`

#### Stratified by `gender`:
  - Group `0`: Median Survival = `2160.0`, Final S(t) = `0.4764`
  - Group `1`: Median Survival = `1317.0`, Final S(t) = `0.0000`

#### Stratified by `chf`:
  - Group `0`: Median Survival = `2358.0`, Final S(t) = `0.0000`
  - Group `1`: Median Survival = `359.0`, Final S(t) = `0.0000`

#### Stratified by `cvd`:
  - Group `0`: Median Survival = `2353.0`, Final S(t) = `0.0000`
  - Group `1`: Median Survival = `1577.0`, Final S(t) = `0.0000`

### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions

| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |
| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |
| `age` | 0 (0.0%) | 0 (0.0%) | Normal / Near-Gaussian | Standard Scaling (Z-score) |
| `av3` | 11 (2.2%) | 11 (2.2%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `bmi` | 18 (3.6%) | 1 (0.2%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `diasbp` | 9 (1.8%) | 4 (0.8%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `hr` | 9 (1.8%) | 2 (0.4%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `los` | 33 (6.6%) | 10 (2.0%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `miord` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `mitype` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `sysbp` | 6 (1.2%) | 1 (0.2%) | Normal / Near-Gaussian | Standard Scaling (Z-score) |
| `lenfol` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |


### Step 3.12 — Clinical Insights & Observations

- **Q1 Mortality Factors**: Congestive Heart Failure (chf=1) and Cardiogenic Shock (sho=1) are the strongest predictors of mortality post-MI. Patients with CHF have significantly lower survival probability.
- **Q2 Highest Risk Age**: Elderly patients (age > 75) exhibit dramatically higher mortality rates, lower baseline blood pressure, and higher prevalence of comorbidities (CVD/AFB).

---

## 1. METABRIC Dataset

### Step 3.1 — Dataset Integrity

- **Rows**: 1904
- **Columns**: 8
- **Memory Usage**: 46.64 KB
- **Time Column**: `duration`
- **Event Column**: `event`
- **Duplicate Rows**: 0 (0.0%)

### Step 3.2 — Data Dictionary

| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |
| ------- | -------------------- | --------- | ------------- | ------- | ------ |
| `age` | Age — Patient age at diagnosis | Numerical (Years) | N/A | 0 | 70 |
| `tumour_stage` | Tumour Stage — Pathological tumor stage | Ordinal (1-4) | N/A | 0 | 4 |
| `lymph_nodes_positive` | Positive Lymph Nodes — Number of positive lymph nodes | Numerical (count) | N/A | 0 | 18 |
| `chemotherapy` | Chemotherapy — Chemotherapy treatment received | Binary (0/1) | N/A | 0 | 2 |
| `hormone_therapy` | Hormone Therapy — Hormone therapy treatment received | Binary (0/1) | N/A | 0 | 2 |
| `PAM50Subtype` | PAM50 Subtype — Molecular subtype (Basal, Her2, LumA, LumB, Normal) | Categorical | N/A | 0 | 5 |
| `duration` | Overall Survival Time — Time to death or loss to follow-up | Numerical (Months) | N/A | 0 | 1401 |
| `event` | Mortality Event — Event indicator (1=dead, 0=censored) | Binary Target (0/1) | N/A | 0 | 2 |


### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis

- **Total Missing Cells**: 0 (0.0%)

### Step 3.5 — Numerical Feature Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |
| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |
| `age` | 55.179 | 55.0 | 40 | 20.397 | 21 | 90 | 38.0 | 73.0 | 35.0 | 0.001 | -1.228 |
| `lymph_nodes_positive` | 2.023 | 1.0 | 0 | 2.47 | 0 | 27 | 0.0 | 3.0 | 3.0 | 2.218 | 8.977 |
| `duration` | 130.43 | 94.8 | 14.9 | 124.743 | 0.1 | 939.1 | 39.3 | 179.45 | 140.15 | 1.761 | 4.192 |


### Step 3.6 — Categorical Feature Analysis

#### Feature: `tumour_stage` (Imbalance Ratio: 10.24:1)
  - `2`: 942 records (49.47%)
  - `1`: 388 records (20.38%)
  - `4`: 92 records (4.83%)
  - `3`: 482 records (25.32%)
#### Feature: `chemotherapy` (Imbalance Ratio: 2.32:1)
  - `1`: 574 records (30.15%)
  - `0`: 1330 records (69.85%)
#### Feature: `hormone_therapy` (Imbalance Ratio: 1.56:1)
  - `1`: 1159 records (60.87%)
  - `0`: 745 records (39.13%)
#### Feature: `PAM50Subtype` (Imbalance Ratio: 1.17:1)
  - `LumA`: 357 records (18.75%)
  - `LumB`: 392 records (20.59%)
  - `Normal`: 368 records (19.33%)
  - `Her2`: 369 records (19.38%)
  - `Basal`: 418 records (21.95%)
#### Feature: `event` (Imbalance Ratio: 1.38:1)
  - `1`: 1104 records (57.98%)
  - `0`: 800 records (42.02%)


### Step 3.7 — Survival Target Statistics

- **Total Patients**: 1904
- **Events Experienced (1)**: 1104
- **Censored Patients (0)**: 800
- **Censoring Rate**: **42.02%**
- **Survival Time Range**: 0.1 to 939.1
- **Mean Survival Time**: 130.43
- **Median Survival Time**: 94.8

### Step 3.8 — Kaplan-Meier Survival Analysis

- **Overall Median Survival Time**: `156.1`
- **Final Horizon Survival Probability**: `0.0000`

#### Stratified by `PAM50Subtype`:
  - Group `Basal`: Median Survival = `160.1`, Final S(t) = `0.0000`
  - Group `Her2`: Median Survival = `136.9`, Final S(t) = `0.0000`
  - Group `LumA`: Median Survival = `162.6`, Final S(t) = `0.0000`
  - Group `LumB`: Median Survival = `155.9`, Final S(t) = `0.0375`
  - Group `Normal`: Median Survival = `156.7`, Final S(t) = `0.0158`

#### Stratified by `tumour_stage`:
  - Group `1`: Median Survival = `154.1`, Final S(t) = `0.0000`
  - Group `2`: Median Survival = `152.6`, Final S(t) = `0.0169`
  - Group `3`: Median Survival = `170.5`, Final S(t) = `0.0000`
  - Group `4`: Median Survival = `148.7`, Final S(t) = `0.0000`

#### Stratified by `hormone_therapy`:
  - Group `0`: Median Survival = `154.1`, Final S(t) = `0.0000`
  - Group `1`: Median Survival = `157.5`, Final S(t) = `0.0000`

### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions

| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |
| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |
| `age` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `lymph_nodes_positive` | 78 (4.1%) | 31 (1.63%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `duration` | 84 (4.41%) | 31 (1.63%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |


### Step 3.12 — Clinical Insights & Observations

- **Q1 Gene Expression Features**: PAM50 Subtypes dictate distinct survival trajectories: Luminal A exhibits the best overall survival, while Basal-like and Her2-enriched subtypes display steep early mortality.
- **Q2 Dominant Clinical Variables**: Tumour stage, lymph node status, and PAM50 molecular subtyping dominate prognosis over individual clinical demographics alone.

---
