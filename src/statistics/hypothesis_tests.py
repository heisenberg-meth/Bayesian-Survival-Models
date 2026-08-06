"""
Statistical hypothesis testing and multiple comparison correction engine.
Includes Wilcoxon, Friedman, and pure-Python correction algorithms (Bonferroni, Holm, BH).
"""

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


class SurvivalHypothesisTests:
    """Hypothesis testing for survival group differences and model benchmark validation."""

    @staticmethod
    def wilcoxon_signed_rank(
        scores_a: np.ndarray, scores_b: np.ndarray
    ) -> tuple[float, float]:
        """Performs two-sample Wilcoxon signed-rank test."""
        stat, p_val = wilcoxon(scores_a, scores_b)
        return float(stat), float(p_val)

    @staticmethod
    def friedman_test(*args: np.ndarray) -> tuple[float, float]:
        """Performs Friedman test across multiple model score vectors."""
        stat, p_val = friedmanchisquare(*args)
        return float(stat), float(p_val)

    @staticmethod
    def multiple_comparison_correction(
        p_values: list[float], method: str = "bonferroni"
    ) -> list[float]:
        """
        Applies multiple-testing p-value correction.
        Supported methods: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg).
        """
        p_arr = np.asarray(p_values, dtype=float)
        m = len(p_arr)
        if m <= 1:
            return list(p_arr)

        if method == "bonferroni":
            corrected = p_arr * m
            return list(np.minimum(corrected, 1.0))

        elif method == "holm":
            # Holm-Bonferroni step-down method
            idx = np.argsort(p_arr)
            sorted_p = p_arr[idx]
            corrected_sorted = np.zeros(m)
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = p * (m - i)
            # Enforce monotonicity
            for i in range(1, m):
                corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i - 1])
            corrected_sorted = np.minimum(corrected_sorted, 1.0)
            # Restore original order
            corrected = np.zeros(m)
            corrected[idx] = corrected_sorted
            return list(corrected)

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR controlling procedure
            idx = np.argsort(p_arr)
            sorted_p = p_arr[idx]
            corrected_sorted = np.zeros(m)
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = p * m / (i + 1)
            # Enforce monotonicity (working backwards)
            for i in range(m - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
            corrected_sorted = np.minimum(corrected_sorted, 1.0)
            # Restore original order
            corrected = np.zeros(m)
            corrected[idx] = corrected_sorted
            return list(corrected)

        else:
            raise ValueError(f"Unknown correction method: {method}")
