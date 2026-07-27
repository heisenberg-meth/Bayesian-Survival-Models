"""
MCMC Sampler interface for Bayesian Survival Models.
"""
from typing import Dict, Any

class MCMCSampler:
    """Manages MCMC execution and trace generation."""

    def __init__(self, draws: int = 2000, tune: int = 1000, chains: int = 4, target_accept: float = 0.95):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept

    def sample(self, model_spec: Any) -> Dict[str, Any]:
        """Runs sampler over model specification."""
        return {
            "draws": self.draws,
            "chains": self.chains,
            "status": "ready",
        }
