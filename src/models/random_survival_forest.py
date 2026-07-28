"""
Random Survival Forest (RSF) Implementation for Bayesian Survival Models.
Non-parametric ensemble algorithm using Log-Rank splitting criterion and Nelson-Aalen
leaf cumulative hazard estimation. Handles non-linear feature interactions and non-proportional hazards.
"""

import numpy as np
import pandas as pd

from src.models.base import BaseSurvivalModel


def _log_rank_split_score(x_feat: np.ndarray, times: np.ndarray, events: np.ndarray, threshold: float) -> float:
    """Computes two-sample Log-Rank statistic for a candidate split (threshold)."""
    left_mask = (x_feat <= threshold)
    right_mask = ~left_mask

    n_left = left_mask.sum()
    n_right = right_mask.sum()

    if n_left < 3 or n_right < 3:
        return -1.0

    # Distinct event times
    event_mask = (events == 1)
    if not np.any(event_mask):
        return -1.0

    # Log-rank test statistic computation
    unique_event_times = np.unique(times[event_mask])

    num_sum = 0.0
    den_sum = 0.0

    for t in unique_event_times:
        # Risk sets at time t
        at_risk = (times >= t)
        Y_total = at_risk.sum()
        if Y_total <= 1:
            continue

        Y_L = (at_risk & left_mask).sum()
        if Y_L == 0 or Y_L == Y_total:
            continue

        d_total = (at_risk & (times == t) & event_mask).sum()
        d_L = (at_risk & left_mask & (times == t) & event_mask).sum()

        E_L = d_total * (Y_L / Y_total)
        V_L = (Y_L / Y_total) * (1.0 - Y_L / Y_total) * ((Y_total - d_total) / (Y_total - 1.0)) * d_total

        num_sum += (d_L - E_L)
        den_sum += V_L

    if den_sum <= 1e-8:
        return -1.0

    log_rank_stat = (num_sum ** 2) / den_sum
    return float(log_rank_stat)


class SurvivalTreeNode:
    """Node in a Survival Tree."""

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False

        # Leaf parameters
        self.unique_times = np.array([])
        self.nelson_aalen_chaz = np.array([])


class SurvivalTree:
    """Single Survival Tree built using Log-Rank splitting."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 10,
        max_features: str = "sqrt",
        random_state: int | None = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.all_unique_event_times = np.array([])

    def fit(self, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray, all_event_times: np.ndarray):
        self.all_unique_event_times = all_event_times
        rng = np.random.RandomState(self.random_state)
        self.root = self._build_tree(X, y_time, y_event, depth=0, rng=rng)
        return self

    def _build_tree(self, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray, depth: int, rng: np.random.RandomState) -> SurvivalTreeNode:
        node = SurvivalTreeNode(depth=depth)
        n_samples, n_features = X.shape

        # Leaf termination check
        if depth >= self.max_depth or n_samples < self.min_samples_split or y_event.sum() == 0:
            return self._make_leaf(node, y_time, y_event)

        # Select random subset of features
        if self.max_features == "sqrt":
            k_feats = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, float):
            k_feats = max(1, int(self.max_features * n_features))
        else:
            k_feats = n_features

        feat_indices = rng.choice(n_features, size=k_feats, replace=False)

        best_score = -1.0
        best_feat = None
        best_thresh = None

        for feat in feat_indices:
            x_col = X[:, feat]
            unique_vals = np.unique(x_col)

            if len(unique_vals) <= 1:
                continue

            # Sample candidate split percentiles to speed up search
            if len(unique_vals) > 10:
                percentiles = np.linspace(10, 90, num=8)
                candidate_thresholds = np.percentile(x_col, percentiles)
            else:
                candidate_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in candidate_thresholds:
                score = _log_rank_split_score(x_col, y_time, y_event, thresh)
                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh

        if best_score <= 0.0 or best_feat is None:
            return self._make_leaf(node, y_time, y_event)

        node.feature_idx = best_feat
        node.threshold = best_thresh

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y_time[left_mask], y_event[left_mask], depth + 1, rng)
        node.right = self._build_tree(X[right_mask], y_time[right_mask], y_event[right_mask], depth + 1, rng)

        return node

    def _make_leaf(self, node: SurvivalTreeNode, y_time: np.ndarray, y_event: np.ndarray) -> SurvivalTreeNode:
        node.is_leaf = True
        node.unique_times = self.all_unique_event_times

        # Compute Nelson-Aalen Cumulative Hazard for leaf
        chaz = np.zeros(len(self.all_unique_event_times), dtype=float)
        cum_h = 0.0

        for idx, t in enumerate(self.all_unique_event_times):
            at_risk = (y_time >= t).sum()
            events_at_t = ((y_time == t) & (y_event == 1)).sum()

            if at_risk > 0:
                cum_h += (events_at_t / at_risk)
            chaz[idx] = cum_h

        node.nelson_aalen_chaz = chaz
        return node

    def predict_chaz(self, X: np.ndarray) -> np.ndarray:
        """Predicts cumulative hazard curve for each sample in X. Shape: (N, M)."""
        n_samples = X.shape[0]
        m_times = len(self.all_unique_event_times)
        chaz_matrix = np.zeros((n_samples, m_times), dtype=float)

        for i in range(n_samples):
            curr = self.root
            while not curr.is_leaf:
                if X[i, curr.feature_idx] <= curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            chaz_matrix[i, :] = curr.nelson_aalen_chaz

        return chaz_matrix


class RandomSurvivalForestModel(BaseSurvivalModel):
    """Random Survival Forest Ensemble Model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_samples_split: int = 10,
        max_features: str = "sqrt",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees: list[SurvivalTree] = []
        self.feature_names = []
        self.all_unique_event_times = np.array([])
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray) -> "RandomSurvivalForestModel":
        """Fits Random Survival Forest ensemble."""
        self.feature_names = list(X.columns)
        X_mat = X.values.astype(float)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        n_samples, _n_features = X_mat.shape
        self.all_unique_event_times = np.sort(np.unique(y_time[y_event == 1]))

        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_mat[boot_idx]
            t_boot = y_time[boot_idx]
            e_boot = y_event[boot_idx]

            tree_seed = rng.randint(0, 1000000)
            tree = SurvivalTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_boot, t_boot, e_boot, self.all_unique_event_times)
            self.trees.append(tree)

        # Compute Permutation Feature Importances (VIMP)
        self._compute_feature_importance(X, y_time, y_event)

        return self

    def predict_cumulative_hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble cumulative hazard curves. Shape: (N_samples, M_times)."""
        X_mat = X[self.feature_names].values.astype(float)
        n_samples = len(X_mat)
        m_times = len(self.all_unique_event_times)

        chaz_sum = np.zeros((n_samples, m_times), dtype=float)

        for tree in self.trees:
            chaz_sum += tree.predict_chaz(X_mat)

        return chaz_sum / len(self.trees)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts risk score (Mortality score = integrated cumulative hazard)."""
        chaz = self.predict_cumulative_hazard(X)
        # Sum cumulative hazard across event times as mortality score
        return np.sum(chaz, axis=1)

    def predict_survival(self, X: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
        """Predicts survival probability matrix S(t | X_i) = exp(-H_RSF(t | X_i))."""
        chaz_forest = self.predict_cumulative_hazard(X)
        
        # Interpolate cumulative hazard to eval_times
        indices = np.searchsorted(self.all_unique_event_times, eval_times, side='right') - 1
        h0_eval = np.zeros((len(X), len(eval_times)), dtype=float)

        valid_mask = indices >= 0
        valid_indices = np.minimum(indices[valid_mask], len(self.all_unique_event_times) - 1)

        for i in range(len(X)):
            h0_eval[i, valid_mask] = chaz_forest[i, valid_indices]

        surv_matrix = np.exp(-h0_eval)
        return np.clip(surv_matrix, 1e-6, 1.0)

    def _compute_feature_importance(self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray):
        """Computes Permutation Feature Importance (VIMP) based on drop in C-index."""
        from src.evaluation.metrics import concordance_index

        baseline_risk = self.predict_risk(X)
        baseline_c, _ = concordance_index(y_time, y_event, baseline_risk)

        importances = {}
        rng = np.random.RandomState(self.random_state)

        for col in self.feature_names:
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_risk = self.predict_risk(X_perm)
            perm_c, _ = concordance_index(y_time, y_event, perm_risk)
            importances[col] = float(max(0.0, baseline_c - perm_c))

        self.feature_importances_ = importances

    def get_summary(self) -> pd.DataFrame:
        """Returns feature importances dataframe."""
        if self.feature_importances_ is None:
            return pd.DataFrame()

        df_imp = pd.DataFrame([
            {"feature": k, "importance (VIMP)": v} for k, v in self.feature_importances_.items()
        ]).sort_values(by="importance (VIMP)", ascending=False).reset_index(drop=True)
        return df_imp
