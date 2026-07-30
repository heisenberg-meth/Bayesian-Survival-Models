"""
Survival Model Evaluation Metrics Engine for Bayesian Survival Models.
Provides publication-grade vectorized implementations for:
- Harrell's Concordance Index (C-index) with standard error
- Kaplan-Meier IPCW (Inverse Probability of Censoring Weighting) Estimator
- Time-Dependent Brier Score BS(t)
- Integrated Brier Score (IBS)
- Comprehensive Model Evaluation Suite
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def concordance_index(
    y_time: np.ndarray, y_event: np.ndarray, risk_scores: np.ndarray
) -> tuple[float, float]:
    """
    Computes Harrell's Concordance Index (C-index) for right-censored survival data.

    Parameters:
        y_time (np.ndarray): Survival/follow-up times (N,)
        y_event (np.ndarray): Event status indicators (1=event, 0=censored) (N,)
        risk_scores (np.ndarray): Predicted risk scores (higher value = higher risk / shorter survival) (N,)

    Returns:
        c_index (float): Harrell's C-index in [0, 1]
        se (float): Asymptotic standard error approximation
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    n = len(y_time)
    if n == 0:
        return 0.5, 0.0

    concordant = 0.0
    tied_risk = 0.0
    total_pairs = 0.0

    # Identify all valid comparable pairs (i, j) where event_i == 1 and time_i < time_j
    # Or both event_i == 1 and event_j == 1 and time_i != time_j
    event_indices = np.where(y_event == 1)[0]

    for i in event_indices:
        t_i = y_time[i]
        r_i = risk_scores[i]

        for j in range(n):
            if i == j:
                continue

            t_j = y_time[j]
            e_j = y_event[j]

            # Pair is comparable if t_i < t_j, or if t_i == t_j and e_j == 0
            if t_i < t_j:
                r_j = risk_scores[j]
                total_pairs += 1.0

                if r_i > r_j:
                    concordant += 1.0
                elif r_i == r_j:
                    tied_risk += 1.0
            elif t_i == t_j and e_j == 1 and i < j:
                # Both experienced event at exact same time
                r_j = risk_scores[j]
                total_pairs += 1.0

                if r_i > r_j:
                    concordant += 1.0
                elif r_i == r_j:
                    tied_risk += 1.0

    if total_pairs == 0:
        return 0.5, 0.0

    c_idx = (concordant + 0.5 * tied_risk) / total_pairs

    # Asymptotic standard error estimation
    se = np.sqrt((c_idx * (1.0 - c_idx)) / max(total_pairs, 1.0))

    return float(c_idx), float(se)


class KaplanMeierCensoringEstimator:
    """Computes Kaplan-Meier survival curve for censoring distribution G(t) = P(C > t)."""

    def __init__(self):
        self.times = np.array([])
        self.survival_probs = np.array([])

    def fit(self, y_time: np.ndarray, y_event: np.ndarray):
        """Fits censoring distribution where event indicator is inverted (1 - event)."""
        y_time = np.asarray(y_time, dtype=float)
        y_cens = 1 - np.asarray(y_event, dtype=int)

        order = np.argsort(y_time)
        times = y_time[order]
        cens = y_cens[order]

        unique_times, _counts = np.unique(times, return_counts=True)
        n_at_risk = len(times)

        surv_probs = []
        current_surv = 1.0

        for t in unique_times:
            idx = times == t
            d_i = cens[idx].sum()  # number of censings at time t
            n_i = n_at_risk

            if n_i > 0:
                current_surv *= 1.0 - d_i / n_i
            surv_probs.append(current_surv)
            n_at_risk -= len(idx)

        self.times = unique_times
        self.survival_probs = np.array(surv_probs)
        return self

    def predict(self, eval_times: np.ndarray) -> np.ndarray:
        """Predicts G(t) for given evaluation times."""
        eval_times = np.asarray(eval_times, dtype=float)
        if len(self.times) == 0:
            return np.ones_like(eval_times)

        # Step function interpolation (left-continuous or step-right)
        indices = np.searchsorted(self.times, eval_times, side="right") - 1
        g_vals = np.ones_like(eval_times, dtype=float)

        valid_mask = indices >= 0
        g_vals[valid_mask] = self.survival_probs[
            np.minimum(indices[valid_mask], len(self.survival_probs) - 1)
        ]
        # Clip to avoid division by zero in IPCW
        return np.maximum(g_vals, 1e-4)


def brier_score_at_time(
    y_time: np.ndarray,
    y_event: np.ndarray,
    surv_probs: np.ndarray,
    eval_time: float,
    km_cens: KaplanMeierCensoringEstimator,
) -> float:
    """
    Computes Inverse Probability of Censoring Weighted (IPCW) Brier Score at time t.

    Parameters:
        y_time: Observed survival times (N,)
        y_event: Event indicators (N,)
        surv_probs: Predicted survival probabilities S(eval_time | X_i) for each subject i
        eval_time: Evaluation time milestone t
        km_cens: Fitted KaplanMeierCensoringEstimator
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    surv_probs = np.asarray(surv_probs, dtype=float)

    n = len(y_time)
    if n == 0:
        return 0.0

    g_t = km_cens.predict(np.array([eval_time]))[0]
    g_ti = km_cens.predict(y_time)

    weights = np.zeros(n, dtype=float)
    bs_terms = np.zeros(n, dtype=float)

    # Term 1: Subject experienced event prior to eval_time (T_i <= t and E_i == 1)
    mask1 = (y_time <= eval_time) & (y_event == 1)
    weights[mask1] = 1.0 / g_ti[mask1]
    bs_terms[mask1] = (0.0 - surv_probs[mask1]) ** 2

    # Term 2: Subject survived past eval_time (T_i > t)
    mask2 = y_time > eval_time
    weights[mask2] = 1.0 / g_t
    bs_terms[mask2] = (1.0 - surv_probs[mask2]) ** 2

    # Term 3: Subject censored before eval_time (T_i <= t and E_i == 0) -> weight = 0

    valid_mask = mask1 | mask2
    if not np.any(valid_mask):
        return 0.0

    bs = float(np.sum(weights[valid_mask] * bs_terms[valid_mask]) / n)
    return bs


def integrated_brier_score(
    y_time: np.ndarray,
    y_event: np.ndarray,
    surv_prob_fn: Callable[[np.ndarray], np.ndarray],
    eval_times: np.ndarray,
) -> float:
    """
    Computes Integrated Brier Score (IBS) by integrating BS(t) across time range.

    Parameters:
        y_time: Survival times
        y_event: Event indicators
        surv_prob_fn: Callable function taking eval_times (M,) and returning (N, M) matrix of S(t_j | X_i)
        eval_times: Grid of evaluation times (M,)
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    eval_times = np.sort(np.asarray(eval_times, dtype=float))

    km_cens = KaplanMeierCensoringEstimator().fit(y_time, y_event)

    # Predict survival probability matrix: shape (N, M)
    S_mat = surv_prob_fn(eval_times)

    bs_list = []
    for idx, t in enumerate(eval_times):
        probs_t = S_mat[:, idx]
        bs_t = brier_score_at_time(y_time, y_event, probs_t, t, km_cens)
        bs_list.append(bs_t)

    bs_arr = np.array(bs_list)

    # Integrate using trapezoidal rule
    t_min, t_max = eval_times[0], eval_times[-1]
    if t_max == t_min:
        return float(bs_arr[0])

    ibs = float(np.trapz(bs_arr, eval_times) / (t_max - t_min))
    return ibs


def evaluate_survival_model(
    y_time: np.ndarray,
    y_event: np.ndarray,
    risk_scores: np.ndarray,
    surv_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    eval_times: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Master Evaluation Suite for any survival model.
    Returns C-index, SE, Brier scores, Integrated Brier Score (IBS), and time-dependent AUC.
    """
    c_idx, se = concordance_index(y_time, y_event, risk_scores)

    results = {
        "c_index": float(round(c_idx, 4)),
        "c_index_se": float(round(se, 4)),
    }

    if surv_prob_fn is not None and eval_times is not None and len(eval_times) > 0:
        km_cens = KaplanMeierCensoringEstimator().fit(y_time, y_event)

        S_mat = surv_prob_fn(eval_times)
        brier_scores = {}
        td_aucs = {}

        from src.evaluation.ranking import time_dependent_auc

        for idx, t in enumerate(eval_times):
            bs_t = brier_score_at_time(y_time, y_event, S_mat[:, idx], t, km_cens)
            brier_scores[f"bs_time_{round(t, 1)}"] = float(round(bs_t, 4))

            auc_t = time_dependent_auc(y_time, y_event, risk_scores, t, km_cens)
            td_aucs[f"auc_time_{round(t, 1)}"] = float(round(auc_t, 4))

        ibs = integrated_brier_score(y_time, y_event, surv_prob_fn, eval_times)

        results["brier_scores"] = brier_scores
        results["integrated_brier_score"] = float(round(ibs, 4))
        results["time_dependent_auc"] = td_aucs

    return results
