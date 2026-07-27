"""
Prior specifications for Bayesian Survival Models.
"""
from typing import Dict, Any, Optional

class PriorSpecification:
    """Configures prior distributions for regression coefficients and baseline hazard."""
    
    def __init__(
        self,
        beta_mean: float = 0.0,
        beta_sd: float = 10.0,
        baseline_prior_type: str = "gamma",
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.baseline_prior_type = baseline_prior_type
        self.alpha = alpha
        self.beta = beta

    def get_config(self) -> Dict[str, Any]:
        return {
            "beta_mean": self.beta_mean,
            "beta_sd": self.beta_sd,
            "baseline_prior_type": self.baseline_prior_type,
            "alpha": self.alpha,
            "beta": self.beta,
        }
