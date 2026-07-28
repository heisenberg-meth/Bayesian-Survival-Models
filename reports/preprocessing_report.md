# Phase 4 — Data Preprocessing & Pipeline Architecture Report

## Executive Summary

Phase 4 (**Data Preprocessing**) establishes the standardized data engineering pipeline for the Bayesian Survival Analysis project. By converting raw clinical data into machine-learning-ready datasets, this phase guarantees that all four downstream benchmark survival models (**Cox Proportional Hazards**, **Bayesian Cox**, **Random Survival Forests**, and **DeepSurv**) operate on identical, leak-free, standardized inputs.

The preprocessing pipeline was executed across **GBSG2** (686 patients), **WHAS500** (500 patients), and **METABRIC** (1,904 patients).

---

## 1. Step-by-Step Preprocessing Protocol

### Step 4.1 — Data Validation
* **Schema Integrity**: Validated target variables (`time`, `event`) across all datasets.
* **Range Verification**: Verified that survival times are strictly positive ($T > 0$) and event indicators are strictly binary ($E \in \{0, 1\}$).
* **Missing & Infinite Values**: Confirmed $0$ infinite or corrupted values.

| Dataset | Total Samples | Target Time Col | Target Event Col | Validation Result |
| :--- | :---: | :--- | :--- | :---: |
| **GBSG2** | 686 | `time` (Days) | `cens` (1=recurrence, 0=censored) | **PASSED** |
| **WHAS500** | 500 | `lenfol` (Days) | `fstat` (1=dead, 0=alive) | **PASSED** |
| **METABRIC** | 1,904 | `duration` (Months) | `event` (1=dead, 0=censored) | **PASSED** |

---

### Step 4.2 — Data Cleaning & Standardization
* Dropped exact duplicate rows ($0$ duplicates found).
* Standardized column names to lower case and uniform snake_case.
* Renamed raw dataset target column headers to `time` and `event` across all exported CSVs.

---

### Step 4.3 — Missing Value Imputation
* **Strategy**: Median imputation for numerical features, mode imputation for categorical features.
* **Leakage Prevention**: Imputation statistics were computed **strictly on the Training set** ($70\%$) and applied to Validation ($15\%$) and Test ($15\%$) sets.
* **Result**: All raw datasets were $100\%$ complete. Pipeline retains imputation parameters in `metadata.json` for future production deployment.

---

### Step 4.4 — Categorical Encoding
* **One-Hot Encoding**: Applied to nominal categorical variables with `drop_first=True` (or $K-1$ dummy variables) to eliminate dummy variable trap and multicollinearity:
  * **GBSG2**: `horTh` $\rightarrow$ `horTh_yes`, `menostat` $\rightarrow$ `menostat_Pre`.
  * **WHAS500**: Binary flags `gender_1`, `cvd_1`, `afb_1`, `sho_1`, `chf_1`.
  * **METABRIC**: `chemotherapy_1`, `hormone_therapy_1`, `PAM50Subtype_Her2`, `PAM50Subtype_LumA`, `PAM50Subtype_LumB`, `PAM50Subtype_Normal`.
* **Ordinal Encoding**:
  * **METABRIC**: `tumour_stage` mapped ordinally ($1 \rightarrow 0, 2 \rightarrow 1, 3 \rightarrow 2, 4 \rightarrow 3$).

---

### Step 4.5 — Numerical Feature Scaling
* **Method**: `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
* **Leakage Prevention**: Mean ($\mu$) and standard deviation ($\sigma$) were fitted **only on the Training set** and transformed across Validation and Test sets.
* **Parameters Exported**: Saved per-feature $(\mu, \sigma)$ to `metadata.json`.

---

### Step 4.6 — Feature Engineering & Outlier Transformation
* **Log-Transformations for Skewed Features**:
  To stabilize extreme right-skewed clinical counts identified in Phase 3 EDA:
  * **GBSG2**: $\text{log\_pnode} = \log(1 + \text{pnode})$, $\text{log\_progrec} = \log(1 + \text{progrec})$, $\text{log\_estrec} = \log(1 + \text{estrec})$.
  * **METABRIC**: $\text{log\_lymph\_nodes} = \log(1 + \text{lymph\_nodes\_positive})$.
* **Clinically Justified Interaction Terms**:
  * **GBSG2**: $\text{age\_x\_horTh} = \text{age} \times \mathbb{I}(\text{horTh}=\text{yes})$.
  * **WHAS500**: $\text{age\_x\_chf} = \text{age} \times \text{chf}$, $\text{age\_x\_gender} = \text{age} \times \text{gender}$.
  * **METABRIC**: $\text{age\_x\_stage} = \text{age} \times \text{tumour\_stage}$.

---

### Step 4.7 — Feature Selection & Outlier Management
* Medical outliers in clinical counts were retained to avoid removing valid high-risk patient records, but transformed via $\log(1+x)$ and standardized to control gradient scale.
* Filtered zero-variance features (all retained features exhibit non-zero variance).

---

### Step 4.8 & 4.9 — Survival Target Format
Target vectors across all datasets strictly adhere to:
* `time`: Continuous floating-point value ($T > 0$).
* `event`: Binary integer $E \in \{0, 1\}$.

---

### Step 4.10 — Stratified Train / Validation / Test Splitting
* **Split Ratios**: **70% Train** / **15% Validation** / **15% Test**.
* **Stratification Metric**: Stratified by `event` indicator to guarantee identical censoring proportions across all splits.
* **Random Seed**: Fixed `random_state = 42` for $100\%$ scientific reproducibility.

#### Dataset Split Summary
| Dataset | Total Rows | Train ($70\%$) | Val ($15\%$) | Test ($15\%$) | Train Censoring | Val Censoring | Test Censoring |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GBSG2** | 686 | 479 | 102 | 105 | **55.11%** | **54.90%** | **55.24%** |
| **WHAS500** | 500 | 349 | 74 | 77 | **57.31%** | **58.11%** | **57.14%** |
| **METABRIC** | 1,904 | 1,332 | 285 | 287 | **42.04%** | **42.11%** | **41.81%** |

---

### Step 4.11 — 5-Fold Stratified Cross-Validation Strategy
* Pre-generated 5-fold stratified cross-validation split indices on the Training set for hyperparameter tuning.
* Stored in `cv_folds.json` under each dataset directory.

---

### Step 4.12 — Saved Processed Data Structure

```text
data/processed/
├── gbsg2/
│   ├── train.csv         (479 rows x 13 cols)
│   ├── val.csv           (102 rows x 13 cols)
│   ├── test.csv          (105 rows x 13 cols)
│   ├── cv_folds.json     (5-fold stratified index split)
│   └── metadata.json     (Scaling params, features, validation report)
├── whas500/
│   ├── train.csv         (349 rows x 14 cols)
│   ├── val.csv           (74 rows x 14 cols)
│   ├── test.csv          (77 rows x 14 cols)
│   ├── cv_folds.json     (5-fold stratified index split)
│   └── metadata.json     (Scaling params, features, validation report)
└── metabric/
    ├── train.csv         (1332 rows x 13 cols)
    ├── val.csv           (285 rows x 13 cols)
    ├── test.csv          (287 rows x 13 cols)
    ├── cv_folds.json     (5-fold stratified index split)
    └── metadata.json     (Scaling params, features, validation report)
```

---

### Step 4.13 — Reusable Preprocessing Pipeline Architecture
Module location: `src/data/preprocessing.py`

```text
               Raw Dataset (CSV)
                      │
                      ▼
         DatasetValidator (src/data/validators.py)
                      │
                      ▼
       Cleaning & Target Standardization
                      │
                      ▼
    FeatureEngineer (src/data/feature_engineering.py)
                      │
                      ▼
  Stratified 70/15/15 Splitter (src/data/split.py)
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      Train Subset       Val & Test Subsets
            │                   │
  Fit Scaler & Encoders         │
            │                   │
            ├───────────────────┘
            ▼
    Transform Features
            │
            ▼
  Generate 5-Fold Stratified CV (cv_folds.json)
            │
            ▼
  Export Clean CSVs & Metadata (data/processed/)
```

---

## Conclusion & Next Phase Transition

With Phase 4 complete, all three datasets are fully preprocessed, validated, leak-free, and saved in `data/processed/`.

As agreed, the roadmap has been updated so that **Model Evaluation Framework** is placed before model implementations, ensuring all 4 models are evaluated using identical metrics ($C$-index, Integrated Brier Score, Time-dependent $C$-index) and protocols.

**Next Milestone:** **Phase 5 — Model Evaluation Framework** (`src/evaluation/metrics.py`, `src/evaluation/cross_validation.py`).
