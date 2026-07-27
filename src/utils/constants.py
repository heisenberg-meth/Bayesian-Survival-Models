"""
Project constants and default definitions.
"""

DATASETS = ["gbsg2", "whas500", "metabric"]
MODELS = ["cox_ph", "random_survival_forest", "deepsurv", "bayesian_cox"]
METRICS = ["c_index", "brier_score", "integrated_brier_score", "auc"]

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
