# Comparative Bayesian Survival Analysis: Benchmark of Cox PH, RSF, DeepSurv, and Bayesian Cox Models

This repository contains a publication-grade research framework for benchmarking, validating, and explaining four major paradigms of survival analysis:

1. **Frequentist Baseline**: Cox Proportional Hazards (Cox PH) model.
2. **Non-parametric Ensemble**: Random Survival Forest (RSF).
3. **Deep Learning**: DeepSurv neural network (diffe			rentiable Cox partial likelihood).
4. **Bayesian Primary Contribution**: Bayesian Cox model with piecewise constant baseline hazards and Normal, Student-t, and Laplace prior parameterizations.

We benchmark all models on three classic clinical cohorts: **GBSG2** (breast cancer), **WHAS500** (heart attack), and **METABRIC** (breast cancer genomics).

---

## 🚀 Quick Start & Environment Setup

### Prerequisites

- **Operating System**: Linux (developed and tested on Linux)
- **Python Version**: `3.10` or higher (recommended: `3.10` or `3.11` for compatibility with `pymc` and `torchtuples`)

### Installation via Conda / Mamba (Recommended)

Create the isolated environment containing all dependencies (including PyTorch, PyMC, Scikit-Survival, and Torchtuples):

```bash
# Clone the repository
git clone https://github.com/heisenberg-meth/Bayesian-Survival-Models.git
cd Bayesian-Survival-Models

# Create the conda environment
conda env create -f environment.yml
conda activate survival-benchmark
```

Alternatively, you can install via `pip` and the `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 Ingestion & Preprocessing

To download and preprocess the benchmark datasets:

```bash
# 1. Download datasets (GBSG2, WHAS500, METABRIC)
python scripts/download_datasets.py

# 2. Preprocess, scale, and partition datasets (train/val/test splits, cross-validation folds)
python scripts/prepare_dataset.py
```

This generates the standardized survival format in `data/processed/`.

---

## 🏃 Run Experiments & Training

All models can be trained individually using the pre-configured hyperparameter search scripts.

### 1. Cox PH Baseline

```bash
python scripts/run_cox_ph.py
```

*Fits standard semi-parametric Cox PH models and logs C-index and Integrated Brier Score.*

### 2. Random Survival Forest

```bash
python scripts/run_rsf.py
```

*Executes randomized grid search over tree depth, split criteria, and min samples leaf.*

### 3. DeepSurv

```bash
python scripts/run_deepsurv.py
```

*Trains feedforward survival neural networks using the PyCox framework, with hyperparameter tuning for learning rate, dropout, and batch size.*

### 4. Bayesian Cox Model (Primary Contribution)

```bash
python scripts/run_bayesian_cox.py
```

*Estimates full posteriors using PyMC and Variational Inference (ADVI) / NUTS MCMC sampling.*

---

## 📈 Unified Evaluation & Statistical Validation

To run the unified evaluation pipeline, generate model comparison tables, and produce summary figures:

```bash
# Run comprehensive evaluation
python scripts/evaluate.py
```

This will compile metrics across 5-fold cross-validation and output comparative charts to `reports/figures/model_performance_comparison.png`.

To validate whether differences are statistically meaningful, run the bootstrap validation and hypothesis testing suite:

```bash
python scripts/run_statistical_validation.py
```

This performs:

- **100 Bootstrap Replicates** per model and dataset for 95% Confidence Interval calculations.
- **Pairwise Wilcoxon Signed-Rank Tests** with multiple comparison corrections (**Bonferroni, Holm, FDR-BH**).
- **Stability Analysis** (3 independent random seeds).
- **Sensitivity Analysis** under varying training sizes (50%, 75%, 100%).
- **Computational Complexity Profiling** (training vs. inference rurtimes).

---

## 📂 Repository Structure

```text
Bayesian-Survival-Models/
├── config/            # Yaml files defining hyperparameters and experimental setups
├── data/              # Datasets raw and processed splits
├── docs/              # System architecture, datasets description, and references
├── reports/           # Markdown report templates, generated figures, and JSON tables
│   ├── figures/       # Forest plots, survival curves, and model comparison charts
│   ├── paper/         # Draft manuscript for publication
│   └── tables/        # Metric summary results in JSON/CSV
├── scripts/           # Standalone execution pipelines (download, train, stats, explain)
├── src/               # Reusable core modules
│   ├── datasets/      # Data loaders and standardization logic
│   ├── evaluation/    # Metrics implementation (C-Index, IBS)
│   ├── models/        # Cox PH, RSF, DeepSurv, and Bayesian PyMC structures
│   ├── statistics/    # Wilcoxon testing, Holm corrections, bootstrap CI logic
│   └── visualization/ # Visual styling and plotting functions
└── tests/             # Pytest unit and integration test suite
```

---

## 📄 Publications & Reports

- **Manuscript**: The detailed comparative paper draft can be viewed at:`reports/paper/comparative_survival_analysis_paper.md`
- **Statistical Validation Report**: The automated statistical verification summary is available at:`reports/statistical_validation_report.md`
- **Explainability & SHAP Analysis**: SHAP and posterior hazard ratio analyses are compiled under:
  `reports/explainability_report.md`

---

## 🛠️ Reproducibility and Hardware Log

All experiments are fully deterministic, pinning the random seed to `42`.

- **Python version**: `3.10.12`
- **PyMC version**: `5.10.3`
- **PyTorch version**: `2.1.2`
- **Operating System**: Linux kernel 6.x
- **Hardware Profile**: x86_64, minimum 8GB RAM, CPU execution default (due to missing `g++` compilation fallback in PyTensor, ADVI variational parameters are optimized in pure Python space).
# Bayesian-COX
