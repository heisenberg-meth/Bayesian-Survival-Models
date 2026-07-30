"""
Base Survival Model Interface for Bayesian Survival Analysis Project.
Defines unified API contract for all 4 survival models:
- Cox Proportional Hazards (Baseline)
- Random Survival Forests (RSF)
- DeepSurv (Deep Learning Survival)
- Bayesian Cox (PyMC MCMC)
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseSurvivalModel(ABC):
    """Abstract Base Class for Survival Models."""

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray
    ) -> "BaseSurvivalModel":
        """Fits model parameters on training dataset."""

    @abstractmethod
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts continuous risk scores (higher risk = shorter survival time)."""

    @abstractmethod
    def predict_survival(self, X: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
        """
        Predicts survival probability matrix S(t | X_i).
        Shape: (N_samples, M_times)
        """

    def get_summary(self) -> pd.DataFrame:
        """Returns summary dataframe of model parameters/coefficients if applicable."""
        return pd.DataFrame()


# Alias for backwards compatibility
BaseModel = BaseSurvivalModel
