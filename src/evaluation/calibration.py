"""
Calibration Curve Computation for Survival Models.
"""

import numpy as np


class KaplanMeierSurvivalEstimator:
    """Computes Kaplan-Meier survival curve S(t) = P(T > t)."""

    def __init__(self):
        self.times = np.array([])
        self.survival_probs = np.array([])

    def fit(self, y_time: np.ndarray, y_event: np.ndarray):
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        order = np.argsort(y_time)
        times = y_time[order]
        events = y_event[order]

        unique_times = np.unique(times)
        n_at_risk = len(times)

        surv_probs = []
        current_surv = 1.0

        for t in unique_times:
            idx = times == t
            d_i = events[idx].sum()  # number of events at time t
            n_i = n_at_risk

            if n_i > 0:
                current_surv *= 1.0 - d_i / n_i
            surv_probs.append(current_surv)
            n_at_risk -= len(idx)

        self.times = unique_times
        self.survival_probs = np.array(surv_probs)
        return self

    def predict(self, eval_times: np.ndarray) -> np.ndarray:
        eval_times = np.asarray(eval_times, dtype=float)
        if len(self.times) == 0:
            return np.ones_like(eval_times)

        indices = np.searchsorted(self.times, eval_times, side="right") - 1
        s_vals = np.ones_like(eval_times, dtype=float)

        valid_mask = indices >= 0
        s_vals[valid_mask] = self.survival_probs[
            np.minimum(indices[valid_mask], len(self.survival_probs) - 1)
        ]
        return s_vals


def compute_calibration_curve(
    y_time: np.ndarray,
    y_event: np.ndarray,
    pred_surv: np.ndarray,
    eval_time: float,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes calibration curve at a specific time milestone t.
    Groups predictions into bins and estimates actual survival using Kaplan-Meier.
    """
    order = np.argsort(pred_surv)
    y_time_sorted = y_time[order]
    y_event_sorted = y_event[order]
    pred_surv_sorted = pred_surv[order]

    bin_edges = np.linspace(0, len(pred_surv), n_bins + 1, dtype=int)

    mean_pred = []
    observed_surv = []

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if start == end:
            continue

        bin_time = y_time_sorted[start:end]
        bin_event = y_event_sorted[start:end]
        bin_pred = pred_surv_sorted[start:end]

        # Mean predicted survival probability
        mean_pred.append(float(np.mean(bin_pred)))

        # Estimate observed survival probability at eval_time using KM on this bin
        km = KaplanMeierSurvivalEstimator().fit(bin_time, bin_event)
        obs_s = km.predict(np.array([eval_time]))[0]
        observed_surv.append(float(obs_s))

    return np.array(mean_pred), np.array(observed_surv)
