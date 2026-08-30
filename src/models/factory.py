from typing import Any

from src.models.base import BaseSurvivalModel
from src.models.bayesian.model import BayesianCoxModel
from src.models.cox import CoxPHModel
from src.models.deepsurv import DeepSurvModel
from src.models.random_survival_forest import RandomSurvivalForestModel


class ModelFactory:
    """Factory for creating survival models."""

    @staticmethod
    def create(model_type: str, config: dict[str, Any]) -> BaseSurvivalModel:
        """
        Creates a survival model instance based on the model_type and config.

        Supported model types:
        - bayesian_cox
        - cox_ph
        - rsf
        - deepsurv
        """
        if model_type == "bayesian_cox":
            return BayesianCoxModel(
                inference_method=config.get("method", "mcmc"),
                draws=config.get("draws", 2000),
                tune=config.get("tune", 1000),
                chains=config.get("chains", 4),
                target_accept=config.get("target_accept", 0.95),
                random_state=config.get("seed", 42),
                coefficient_prior=config.get("coefficient_prior", "normal"),
                prior_params=config.get("prior_params", {}),
                baseline_hazard_prior=config.get("baseline_hazard_prior", "gamma"),
                baseline_hazard_params=config.get("baseline_hazard_params", {}),
                n_intervals=config.get("n_intervals", 6),
            )
        elif model_type == "cox_ph":
            return CoxPHModel(l2_reg=config.get("l2_reg", 1e-4))
        elif model_type == "rsf":
            return RandomSurvivalForestModel(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                min_samples_split=config.get("min_samples_split", 10),
                min_samples_leaf=config.get("min_samples_leaf", 3),
                max_features=config.get("max_features", "sqrt"),
                bootstrap=config.get("bootstrap", True),
                random_state=config.get("seed", 42),
            )
        elif model_type == "deepsurv":
            return DeepSurvModel(
                hidden_dims=config.get("hidden_dims", [32, 16]),
                l2_reg=config.get("l2_reg", 1e-3),
                random_state=config.get("seed", 42),
                max_iter=config.get("max_iter", 300),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
