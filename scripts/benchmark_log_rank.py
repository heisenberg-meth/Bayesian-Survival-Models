import time

import numpy as np


def original_log_rank_split_score(
    x_feat: np.ndarray, times: np.ndarray, events: np.ndarray, threshold: float
) -> float:
    left_mask = x_feat <= threshold
    right_mask = ~left_mask

    n_left = left_mask.sum()
    n_right = right_mask.sum()

    if n_left < 3 or n_right < 3:
        return -1.0

    # Distinct event times
    event_mask = events == 1
    if not np.any(event_mask):
        return -1.0

    # Log-rank test statistic computation
    unique_event_times = np.unique(times[event_mask])

    num_sum = 0.0
    den_sum = 0.0

    for t in unique_event_times:
        # Risk sets at time t
        at_risk = times >= t
        Y_total = at_risk.sum()
        if Y_total <= 1:
            continue

        Y_L = (at_risk & left_mask).sum()
        if Y_L == 0 or Y_L == Y_total:
            continue

        d_total = (at_risk & (times == t) & event_mask).sum()
        d_L = (at_risk & left_mask & (times == t) & event_mask).sum()

        E_L = d_total * (Y_L / Y_total)
        V_L = (
            (Y_L / Y_total)
            * (1.0 - Y_L / Y_total)
            * ((Y_total - d_total) / (Y_total - 1.0))
            * d_total
        )

        num_sum += d_L - E_L
        den_sum += V_L

    if den_sum <= 1e-8:
        return -1.0

    log_rank_stat = (num_sum**2) / den_sum
    return float(log_rank_stat)


def optimized_log_rank_split_score(
    x_feat: np.ndarray, times: np.ndarray, events: np.ndarray, threshold: float
) -> float:
    left_mask = x_feat <= threshold
    right_mask = ~left_mask

    n_left = left_mask.sum()
    n_right = right_mask.sum()

    if n_left < 3 or n_right < 3:
        return -1.0

    n = len(times)

    # Sort times and events
    sort_idx = np.argsort(times)
    t_sorted = times[sort_idx]
    e_sorted = events[sort_idx]
    L_sorted = left_mask[sort_idx]

    # Suffix sums of L_sorted to get Y_L at each index
    Y_L_all = np.zeros(n + 1, dtype=float)
    Y_L_all[:-1] = np.cumsum(L_sorted[::-1])[::-1]

    # Find unique times and their first occurrence index
    unique_times, first_indices, counts = np.unique(
        t_sorted, return_index=True, return_counts=True
    )

    num_sum = 0.0
    den_sum = 0.0

    e_L_sorted = e_sorted & L_sorted

    for k, t in enumerate(unique_times):
        start = first_indices[k]
        end = start + counts[k]

        # Number of events at time t
        d_total = e_sorted[start:end].sum()
        if d_total == 0:
            continue

        # Number of events in left group at time t
        d_L = e_L_sorted[start:end].sum()

        # Risk sets at time t (suffix starting at 'start')
        Y_total = n - start
        Y_L = Y_L_all[start]

        if Y_total <= 1 or Y_L == 0 or Y_L == Y_total:
            continue

        E_L = d_total * (Y_L / Y_total)
        V_L = (
            (Y_L / Y_total)
            * (1.0 - Y_L / Y_total)
            * ((Y_total - d_total) / (Y_total - 1.0))
            * d_total
        )

        num_sum += d_L - E_L
        den_sum += V_L

    if den_sum <= 1e-8:
        return -1.0

    return float((num_sum**2) / den_sum)


# Generate synthetic survival data
np.random.seed(42)
n_samples = 500
x_feat = np.random.randn(n_samples)
times = np.random.exponential(scale=10.0, size=n_samples)
events = np.random.binomial(n=1, p=0.8, size=n_samples)
threshold = 0.0

# Correctness test
res_orig = original_log_rank_split_score(x_feat, times, events, threshold)
res_opt = optimized_log_rank_split_score(x_feat, times, events, threshold)
print(f"Original score:  {res_orig}")
print(f"Optimized score: {res_opt}")
assert np.allclose(res_orig, res_opt), "Scores do not match!"
print("Correctness verified!")

# Speed benchmark
start = time.time()
for _ in range(200):
    original_log_rank_split_score(x_feat, times, events, threshold)
orig_time = time.time() - start

start = time.time()
for _ in range(200):
    optimized_log_rank_split_score(x_feat, times, events, threshold)
opt_time = time.time() - start

print(f"Original execution time (200 runs):  {orig_time:.4f} seconds")
print(f"Optimized execution time (200 runs): {opt_time:.4f} seconds")
print(f"Speedup factor: {orig_time / opt_time:.2f}x")
