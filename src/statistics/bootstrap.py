"""
Bootstrap Resampling Engine for Survival Models.
Allows estimating uncertainty (SD, SE, 95% Confidence Intervals) for C-index, IBS, and AUC.
"""

import logging
from collections.abc import Callable

import numpy as np

from src.evaluation.metrics import evaluate_survival_model

logger = logging.getLogger(__name__)


class BootstrapResampler:
    """Performs repeated bootstrap resampling to estimate uncertainty in survival metrics."""

    def __init__(self, n_bootstraps: int = 100, random_state: int = 42):
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

    def bootstrap_metrics(
        self,
        y_time: np.ndarray,
        y_event: np.ndarray,
        risk_scores: np.ndarray,
        surv_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        eval_times: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """
        Samples with replacement and evaluates metrics repeatedly.
        Returns a dict mapping metric names to lists of bootstrap scores.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = len(y_time)

        bootstrap_c_indices = []
        bootstrap_ibs = []
        bootstrap_aucs = []

        # Precompute the full survival matrix if survival function is provided
        if surv_prob_fn is not None and eval_times is not None and len(eval_times) > 0:
            S_mat_full = surv_prob_fn(eval_times)
        else:
            S_mat_full = None

        for _ in range(self.n_bootstraps):
            # Sample indices with replacement
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)

            # Slice data
            boot_time = y_time[boot_idx]
            boot_event = y_event[boot_idx]
            boot_risk = risk_scores[boot_idx]

            # Slice survival probability matrix
            if S_mat_full is not None:
                S_mat_boot = S_mat_full[boot_idx, :]

                def boot_surv_fn(times, S_mat_b=S_mat_boot):
                    return S_mat_b
            else:
                boot_surv_fn = None

            # Check if we have at least one event in the bootstrap sample to avoid division by zero/exceptions
            if np.sum(boot_event == 1) < 2 or len(np.unique(boot_time)) < 2:
                continue

            try:
                eval_res = evaluate_survival_model(
                    y_time=boot_time,
                    y_event=boot_event,
                    risk_scores=boot_risk,
                    surv_prob_fn=boot_surv_fn,
                    eval_times=eval_times,
                )

                bootstrap_c_indices.append(eval_res["c_index"])
                if "integrated_brier_score" in eval_res:
                    bootstrap_ibs.append(eval_res["integrated_brier_score"])
                if "time_dependent_auc" in eval_res:
                    auc_vals = list(eval_res["time_dependent_auc"].values())
                    if len(auc_vals) > 0:
                        bootstrap_aucs.append(float(np.mean(auc_vals)))
            except (ValueError, ZeroDivisionError, RuntimeError) as exc:
                logger.debug(
                    "Skipping bootstrap iteration because metric evaluation failed: %s",
                    exc,
                )
                continue

        results = {
            "c_index": bootstrap_c_indices,
        }
        if bootstrap_ibs:
            results["integrated_brier_score"] = bootstrap_ibs
        if bootstrap_aucs:
            results["auc"] = bootstrap_aucs

        return results
