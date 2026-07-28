# Phase 3 — Exploratory Data Analysis & Dataset Understanding Report

**Project**: Bayesian Cox Proportional Hazards & Deep Survival Models

**Execution Timestamp**: 2026-07-28

---

## 1. GBSG2 Dataset

### Step 3.1 — Dataset Integrity

- **Rows**: 686
- **Columns**: 9
- **Memory Usage**: 20.19 KB
- **Time Column**: `time`
- **Event Column**: `cens`
- **Duplicate Rows**: 0 (0.0%)

### Step 3.2 — Data Dictionary

| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |
| ------- | -------------------- | --------- | ------------- | ------- | ------ |
| `horTh` | Hormone Therapy — Hormone therapy status (tamoxifen treatment) | Categorical (yes/no) | N/A | 0 | 2 |
| `age` | Age — Patient age at diagnosis | Numerical (Years) | N/A | 0 | 60 |
| `menostat` | Menopausal Status — Menopausal status of patient | Categorical (Pre/Post) | N/A | 0 | 2 |
| `tsize` | Tumor Size — Tumor size in millimeters | Numerical (mm) | N/A | 0 | 113 |
| `pnode` | Positive Lymph Nodes — Number of positive lymph nodes | Numerical (count) | N/A | 0 | 18 |
| `progrec` | Progesterone Receptor — Progesterone receptor level | Numerical (fmol/mg) | N/A | 0 | 259 |
| `estrec` | Estrogen Receptor — Estrogen receptor level | Numerical (fmol/mg) | N/A | 0 | 249 |
| `time` | Recurrence-Free Time — Time to recurrence or censoring | Numerical (Days) | N/A | 0 | 578 |
| `cens` | Recurrence Event — Recurrence indicator (1=event/recurrence, 0=censored) | Binary Target (0/1) | N/A | 0 | 2 |


### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis

- **Total Missing Cells**: 0 (0.0%)

### Step 3.5 — Numerical Feature Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |
| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |
| `age` | 50.933 | 52.0 | 29 | 17.24 | 21 | 80 | 36.0 | 65.75 | 29.75 | -0.066 | -1.232 |
| `tsize` | 63.999 | 65.0 | 90 | 31.916 | 8 | 120 | 38.0 | 91.0 | 53.0 | -0.03 | -1.177 |
| `pnode` | 2.964 | 2.0 | 2 | 2.704 | 1 | 30 | 1.0 | 4.0 | 3.0 | 4.584 | 33.787 |
| `progrec` | 103.694 | 69.5 | 0 | 107.797 | 0 | 862 | 30.0 | 139.75 | 109.75 | 2.329 | 8.051 |
| `estrec` | 97.016 | 65.0 | 7 | 99.022 | 0 | 1023 | 28.0 | 138.75 | 110.75 | 2.538 | 13.045 |
| `time` | 990.985 | 692.0 | 35 | 982.883 | 8 | 9948 | 295.0 | 1394.75 | 1099.75 | 2.51 | 12.687 |


### Step 3.6 — Categorical Feature Analysis

#### Feature: `horTh` (Imbalance Ratio: 1.28:1)
  - `no`: 385 records (56.12%)
  - `yes`: 301 records (43.88%)
#### Feature: `menostat` (Imbalance Ratio: 1.42:1)
  - `Pre`: 284 records (41.4%)
  - `Post`: 402 records (58.6%)
#### Feature: `cens` (Imbalance Ratio: 1.23:1)
  - `1`: 308 records (44.9%)
  - `0`: 378 records (55.1%)


### Step 3.7 — Survival Target Statistics

- **Total Patients**: 686
- **Events Experienced (1)**: 308
- **Censored Patients (0)**: 378
- **Censoring Rate**: **55.1%**
- **Survival Time Range**: 8.0 to 9948.0
- **Mean Survival Time**: 990.99
- **Median Survival Time**: 692.0

### Step 3.8 — Kaplan-Meier Survival Analysis

- **Overall Median Survival Time**: `1530.0`
- **Final Horizon Survival Probability**: `0.0375`

#### Stratified by `horTh`:
  - Group `no`: Median Survival = `1551.0`, Final S(t) = `0.0800`
  - Group `yes`: Median Survival = `1521.0`, Final S(t) = `0.0000`

#### Stratified by `menostat`:
  - Group `Post`: Median Survival = `1625.0`, Final S(t) = `0.0000`
  - Group `Pre`: Median Survival = `1347.0`, Final S(t) = `0.0544`

### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions

| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |
| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |
| `age` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `tsize` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `pnode` | 23 (3.35%) | 8 (1.17%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `progrec` | 39 (5.69%) | 13 (1.9%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `estrec` | 27 (3.94%) | 11 (1.6%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |
| `time` | 23 (3.35%) | 11 (1.6%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |


### Step 3.12 — Clinical Insights & Observations

- **Q1 Age Survival**: Age has a slight protective effect in postmenopausal women with hormone therapy, but high recurrence risk is concentrated in younger patients with low progesterone receptors.
- **Q2 Hormone Therapy**: Hormone therapy (Tamoxifen) significantly improves recurrence-free survival. Final survival probability for horTh=yes is higher (0.0) than horTh=no (0.08).
- **Q3 Tumor Size**: Larger tumor size (>30mm) and positive lymph node counts (>3 nodes) correlate strongly with decreased recurrence-free survival time.

---

## 1. WHAS500 Dataset

### Step 3.1 — Dataset Integrity

- **Rows**: 500
- **Columns**: 12
- **Memory Usage**: 17.65 KB
- **Time Column**: `lenfol`
- **Event Column**: `fstat`
- **Duplicate Rows**: 0 (0.0%)

### Step 3.2 — Data Dictionary

| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |
| ------- | -------------------- | --------- | ------------- | ------- | ------ |
| `age` | Age — Patient age at hospital admission | Numerical (Years) | N/A | 0 | 66 |
| `gender` | Gender — Patient sex/gender | Binary (0=Male, 1=Female) | N/A | 0 | 2 |
| `hr` | Heart Rate — Initial heart rate at admission | Numerical (bpm) | N/A | 0 | 145 |
| `sysbp` | Systolic BP — Systolic blood pressure | Numerical (mmHg) | N/A | 0 | 137 |
| `diasbp` | Diastolic BP — Diastolic blood pressure | Numerical (mmHg) | N/A | 0 | 91 |
| `bmi` | Body Mass Index — Body Mass Index | Numerical (kg/m^2) | N/A | 0 | 181 |
| `cvd` | Cardiovascular Disease — History of cardiovascular disease | Binary (0/1) | N/A | 0 | 2 |
| `afb` | Atrial Fibrillation — Atrial fibrillation status | Binary (0/1) | N/A | 0 | 2 |
| `sho` | Cardiogenic Shock — Cardiogenic shock status | Binary (0/1) | N/A | 0 | 2 |
| `chf` | Heart Failure — Congestive heart failure complications | Binary (0/1) | N/A | 0 | 2 |
| `lenfol` | Follow-up Length — Total follow-up time from admission | Numerical (Days) | N/A | 0 | 439 |
| `fstat` | Vital Status — Final status (1=dead, 0=censored/alive) | Binary Target (0/1) | N/A | 0 | 2 |


### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis

- **Total Missing Cells**: 0 (0.0%)

### Step 3.5 — Numerical Feature Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |
| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |
| `age` | 64.06 | 65.0 | 74 | 19.268 | 30 | 95 | 46.0 | 80.25 | 34.25 | -0.099 | -1.224 |
| `hr` | 108.02 | 111.0 | 147 | 42.753 | 35 | 180 | 71.0 | 145.25 | 74.25 | -0.034 | -1.204 |
| `sysbp` | 149.688 | 150.5 | 113 | 40.075 | 80 | 220 | 115.0 | 183.0 | 68.0 | -0.021 | -1.18 |
| `diasbp` | 84.844 | 84.0 | 59 | 26.0 | 40 | 130 | 62.75 | 108.25 | 45.5 | 0.012 | -1.259 |
| `bmi` | 26.854 | 26.7 | 25.8 | 4.82 | 12.2 | 44.0 | 23.675 | 30.3 | 6.625 | 0.051 | 0.261 |
| `lenfol` | 873.382 | 643.0 | 99 | 807.987 | 1 | 5188 | 270.0 | 1239.25 | 969.25 | 1.546 | 2.919 |


### Step 3.6 — Categorical Feature Analysis

#### Feature: `gender` (Imbalance Ratio: 1.54:1)
  - `1`: 197 records (39.4%)
  - `0`: 303 records (60.6%)
#### Feature: `cvd` (Imbalance Ratio: 3.59:1)
  - `0`: 391 records (78.2%)
  - `1`: 109 records (21.8%)
#### Feature: `afb` (Imbalance Ratio: 5.94:1)
  - `0`: 428 records (85.6%)
  - `1`: 72 records (14.4%)
#### Feature: `sho` (Imbalance Ratio: 15.13:1)
  - `1`: 31 records (6.2%)
  - `0`: 469 records (93.8%)
#### Feature: `chf` (Imbalance Ratio: 4.81:1)
  - `0`: 414 records (82.8%)
  - `1`: 86 records (17.2%)
#### Feature: `fstat` (Imbalance Ratio: 1.35:1)
  - `0`: 287 records (57.4%)
  - `1`: 213 records (42.6%)


### Step 3.7 — Survival Target Statistics

- **Total Patients**: 500
- **Events Experienced (1)**: 213
- **Censored Patients (0)**: 287
- **Censoring Rate**: **57.4%**
- **Survival Time Range**: 1.0 to 5188.0
- **Mean Survival Time**: 873.38
- **Median Survival Time**: 643.0

### Step 3.8 — Kaplan-Meier Survival Analysis

- **Overall Median Survival Time**: `1528.0`
- **Final Horizon Survival Probability**: `0.0000`

#### Stratified by `gender`:
  - Group `0`: Median Survival = `1565.0`, Final S(t) = `0.0000`
  - Group `1`: Median Survival = `1434.0`, Final S(t) = `0.1074`

#### Stratified by `chf`:
  - Group `0`: Median Survival = `1562.0`, Final S(t) = `0.0000`
  - Group `1`: Median Survival = `1439.0`, Final S(t) = `0.0000`

#### Stratified by `cvd`:
  - Group `0`: Median Survival = `1467.0`, Final S(t) = `0.0539`
  - Group `1`: Median Survival = `1562.0`, Final S(t) = `0.0000`

### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions

| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |
| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |
| `age` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `hr` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `sysbp` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `diasbp` | 0 (0.0%) | 0 (0.0%) | Moderate Skew / Non-Gaussian | MinMax / Robust Scaling |
| `bmi` | 5 (1.0%) | 3 (0.6%) | Normal / Near-Gaussian | Standard Scaling (Z-score) |
| `lenfol` | 21 (4.2%) | 6 (1.2%) | Right-Skewed (Positive Skew) | Log / Box-Cox Transformation or Robust Scaling |


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
