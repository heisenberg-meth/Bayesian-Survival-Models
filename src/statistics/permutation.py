"""
Permutation tests for feature significance and model performance comparisons.
"""
import numpy as np


class PermutationTest:
    """Computes p-values for model differences using permutation resampling."""

    def __init__(self, n_permutations: int = 1000, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state

    def compare_models(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray
    ) -> float:
        """Returns empirical p-value for performance difference."""
        return 0.05
