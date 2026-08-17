"""
Confidence Interval estimation (Normal, Student-t, Percentile, BCa).
"""

import numpy as np
from scipy.stats import sem, t


class ConfidenceIntervalCalculator:
    """Calculates non-parametric and parametric confidence bounds."""

    @staticmethod
    def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
        """Calculates percentile confidence interval."""
        if len(samples) == 0:
            return 0.0, 0.0
        lower = float(np.percentile(samples, 100 * (alpha / 2.0)))
        upper = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
        return lower, upper

    @staticmethod
    def nadeau_bengio_ci(
        diffs: np.ndarray, n_train: int, n_test: int, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Computes Nadeau-Bengio corrected confidence interval for repeated CV."""
        from scipy.stats import t

        n_folds = len(diffs)
        mean_diff = np.mean(diffs)
        var_diff = np.var(diffs, ddof=1)

        correction = (1.0 / n_folds) + (n_test / n_train)
        corrected_var = correction * var_diff

        if corrected_var == 0:
            return float(mean_diff), float(mean_diff)

        t_crit = t.ppf(1 - alpha / 2, df=n_folds - 1)
        margin = t_crit * np.sqrt(corrected_var)

        return float(mean_diff - margin), float(mean_diff + margin)

    @staticmethod
    def normal_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
        """Calculates normal approximation confidence interval."""
        if len(samples) < 2:
            return float(np.mean(samples)) if len(samples) == 1 else 0.0, float(
                np.mean(samples)
            ) if len(samples) == 1 else 0.0
        mean = np.mean(samples)
        se = sem(samples)
        h = se * t.ppf(1.0 - alpha / 2.0, len(samples) - 1)
        return float(mean - h), float(mean + h)

    @staticmethod
    def get_summary_statistics(
        samples: np.ndarray, alpha: float = 0.05
    ) -> dict[str, float]:
        """Returns mean, std, se, and 95% CI bounds."""
        if len(samples) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "se": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }
        samples = np.asarray(samples)
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        se_val = float(sem(samples)) if len(samples) > 1 else 0.0
        ci_lower, ci_upper = ConfidenceIntervalCalculator.percentile_ci(samples, alpha)
        return {
            "mean": mean_val,
            "std": std_val,
            "se": se_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
