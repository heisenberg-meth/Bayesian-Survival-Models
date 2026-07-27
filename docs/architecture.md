# System Architecture

## Overview
This repository provides a modular, reproducible framework for Bayesian Survival Modeling with Cox Proportional Hazards and Deep Survival extensions.

## Core Packages & Components

```
Bayesian-COX/
├── experiments/       # Yaml definitions for experiments
├── docs/              # Research documentation & references
├── src/
│   ├── datasets/      # Dataset abstractions (GBSG2, METABRIC, WHAS500)
│   ├── models/        # Survival models including Bayesian Cox & DeepSurv
│   │   └── bayesian/  # Modular PyMC/Stan Bayesian Cox specification
│   ├── losses/        # Custom loss functions (Cox Partial Likelihood, Ranking Loss)
│   ├── pipelines/     # Training, evaluation, preprocessing, inference pipelines
│   ├── statistics/    # Bootstrap, Permutation tests, CIs, Hypothesis testing
│   ├── explainability/# SHAP, posterior analysis, and feature importance
│   └── visualization/ # Kaplan-Meier, calibration, and posterior density plots
├── artifacts/         # Machine-generated outputs (checkpoints, posterior traces, shap)
└── results/           # Publication-ready tables and figures
```

## Component Responsibilities

1. **`src/datasets/`**: Standardizes dataset loading, validation, and standard survival dataset formats across GBSG2, METABRIC, and WHAS500.
2. **`src/models/bayesian/`**: Encapsulates model formulation, priors, log-likelihood, MCMC sampling, and convergence diagnostics.
3. **`src/pipelines/`**: Orchestrates reproducible execution flows for training, cross-validation, evaluation, and inference.
4. **`src/losses/`**: Implements differentiable loss functions (negative Cox log-likelihood, concordance loss) for neural survival architectures.
5. **`src/statistics/`**: Provides rigorous uncertainty quantification through bootstrapping and non-parametric hypothesis testing.
