"""
Confidence Interval estimation (Normal, Student-t, Percentile, BCa).
"""
from typing import Tuple
import numpy as np

class ConfidenceIntervalCalculator:
    """Calculates non-parametric and parametric confidence bounds."""

    @staticmethod
    def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculates percentile confidence interval."""
        lower = float(np.percentile(samples, 100 * (alpha / 2.0)))
        upper = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
        return lower, upper
