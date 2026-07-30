"""
Ranking and Discrimination Metrics for Survival Models.
"""

import numpy as np

from src.evaluation.metrics import KaplanMeierCensoringEstimator


def time_dependent_auc(
    y_time: np.ndarray,
    y_event: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: float,
    km_cens: KaplanMeierCensoringEstimator,
) -> float:
    """Computes IPCW-weighted cumulative/dynamic time-dependent AUC at time t (Uno's estimator)."""
    len(y_time)

    # Predict censoring probabilities
    g_ti = km_cens.predict(y_time)
    g_t = km_cens.predict(np.array([eval_time]))[0]

    cases_mask = (y_time <= eval_time) & (y_event == 1)
    controls_mask = y_time > eval_time

    cases_idx = np.where(cases_mask)[0]
    controls_idx = np.where(controls_mask)[0]

    if len(cases_idx) == 0 or len(controls_idx) == 0:
        return 0.5

    concordant = 0.0
    tied = 0.0
    total_weight = 0.0

    for i in cases_idx:
        w_i = 1.0 / max(g_ti[i], 1e-4)
        r_i = risk_scores[i]

        for j in controls_idx:
            w_j = 1.0 / max(g_t, 1e-4)
            r_j = risk_scores[j]

            pair_weight = w_i * w_j
            total_weight += pair_weight

            if r_i > r_j:
                concordant += pair_weight
            elif r_i == r_j:
                tied += pair_weight

    if total_weight == 0:
        return 0.5

    return float((concordant + 0.5 * tied) / total_weight)
