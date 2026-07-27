"""
Log-likelihood computations for Cox Proportional Hazards model under right censoring.
"""
import numpy as np

class CoxPartialLikelihood:
    """Computes Cox partial log-likelihood and risk sets."""

    @staticmethod
    def log_likelihood(beta: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
        """Compute partial log likelihood for given parameters beta."""
        order = np.argsort(-time)
        X_ordered = X[order]
        event_ordered = event[order]
        
        eta = X_ordered @ beta
        exp_eta = np.exp(eta)
        cum_sum_risk = np.cumsum(exp_eta)
        
        log_like = np.sum(event_ordered * (eta - np.log(cum_sum_risk)))
        return float(log_like)
