"""
Inference Pipeline for Patient Survival Curves Prediction.
"""

import numpy as np


class InferencePipeline:
    """Orchestrates predictions and survival curve generation on unseen data."""

    def __init__(self, model_checkpoint_path: str):
        self.model_checkpoint_path = model_checkpoint_path

    def predict_risk(self, X_new: np.ndarray) -> np.ndarray:
        """Predict risk scores for new observations."""
        return np.zeros(X_new.shape[0])
