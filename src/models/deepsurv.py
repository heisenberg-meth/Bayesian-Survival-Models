"""
DeepSurv Model Implementation for Bayesian Survival Models Project.
Deep Feedforward Neural Network parameterizing non-linear Cox proportional hazards risk functions.
Optimizes negative Cox partial log-likelihood with SELU/ReLU activations, L2 regularization,
and Breslow baseline hazard estimation.
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt

from src.models.base import BaseSurvivalModel


def _selu(x: np.ndarray) -> np.ndarray:
    """Scaled Exponential Linear Unit (SELU) activation function."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def _selu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of SELU activation function."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, 1.0, alpha * np.exp(x))


class DeepSurvModel(BaseSurvivalModel):
    """DeepSurv (Deep Learning Cox Survival Model)."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        l2_reg: float = 1e-3,
        random_state: int = 42,
        max_iter: int = 300,
    ):
        if hidden_dims is None:
            hidden_dims = [32, 16]
        self.hidden_dims = hidden_dims
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.max_iter = max_iter

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.feature_names = []

        self.unique_event_times = np.array([])
        self.baseline_cum_hazard = np.array([])
        self.training_loss_history = []
        self.feature_importances_ = None

    def _init_weights(self, in_dim: int, rng: np.random.RandomState):
        """Initializes network weights using LeCun normal initialization for SELU."""
        dims = [in_dim] + self.hidden_dims + [1]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            std = np.sqrt(1.0 / dims[i])
            W = rng.normal(loc=0.0, scale=std, size=(dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    def _pack_params(self) -> np.ndarray:
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.ravel())
            params.append(b.ravel())
        return np.concatenate(params)

    def _unpack_params(self, param_vec: np.ndarray, in_dim: int):
        dims = [in_dim] + self.hidden_dims + [1]
        idx = 0
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            w_shape = (dims[i], dims[i + 1])
            w_size = dims[i] * dims[i + 1]
            W = param_vec[idx : idx + w_size].reshape(w_shape)
            idx += w_size

            b_shape = (dims[i + 1],)
            b_size = dims[i + 1]
            b = param_vec[idx : idx + b_size].reshape(b_shape)
            idx += b_size

            self.weights.append(W)
            self.biases.append(b)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass through deep neural network."""
        activations = [X]
        linear_outputs = []

        curr = X
        for i in range(len(self.weights) - 1):
            z = np.dot(curr, self.weights[i]) + self.biases[i]
            linear_outputs.append(z)
            curr = _selu(z)
            activations.append(curr)

        # Output layer (linear log-risk)
        g_risk = np.dot(curr, self.weights[-1]) + self.biases[-1]  # (N, 1)
        linear_outputs.append(g_risk)
        activations.append(g_risk)

        return g_risk.ravel(), activations, linear_outputs

    def fit(
        self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray
    ) -> "DeepSurvModel":
        """Fits DeepSurv neural network parameters by minimizing Cox partial log-likelihood."""
        self.feature_names = list(X.columns)
        X_mat = X.values.astype(float)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        n_samples, in_dim = X_mat.shape

        # Sort by survival time ascending
        order = np.argsort(y_time)
        X_mat = X_mat[order]
        y_time = y_time[order]
        y_event = y_event[order]

        rng = np.random.RandomState(self.random_state)
        self._init_weights(in_dim, rng)

        self.training_loss_history = []

        def _objective_and_grad(param_vec):
            self._unpack_params(param_vec, in_dim)
            g_risk, activations, linear_outputs = self._forward(X_mat)

            # Cox Partial Log-Likelihood
            exp_risk = np.exp(g_risk - np.max(g_risk))
            risk_sums = np.cumsum(exp_risk[::-1])[::-1]
            event_mask = y_event == 1

            log_lik = np.sum(
                g_risk[event_mask] - (np.max(g_risk) + np.log(risk_sums[event_mask]))
            )

            # L2 weight regularization
            l2_loss = 0.5 * self.l2_reg * sum(np.sum(W**2) for W in self.weights)
            neg_loss = -(log_lik - l2_loss)

            self.training_loss_history.append(float(neg_loss))

            # Derivative wrt risk scores g_risk
            dL_dg = np.zeros(n_samples)

            for i in range(n_samples):
                if y_event[i] == 1:
                    dL_dg[i] -= 1.0

            for k in range(n_samples):
                if y_event[k] == 1:
                    at_risk_mask = y_time >= y_time[k]
                    denom = np.sum(exp_risk[at_risk_mask])
                    if denom > 0:
                        dL_dg[at_risk_mask] += exp_risk[at_risk_mask] / denom

            dL_dg += self.l2_reg * 0.0  # regularize weights only

            # Backpropagation
            d_out = dL_dg[:, None]  # (N, 1)

            dW_list = []
            db_list = []

            # Output layer gradients
            dW_out = np.dot(activations[-2].T, d_out) + self.l2_reg * self.weights[-1]
            db_out = np.sum(d_out, axis=0)

            dW_list.insert(0, dW_out)
            db_list.insert(0, db_out)

            curr_delta = np.dot(d_out, self.weights[-1].T)

            # Hidden layers backwards
            for layer_idx in range(len(self.weights) - 2, -1, -1):
                curr_delta = curr_delta * _selu_grad(linear_outputs[layer_idx])
                dW = (
                    np.dot(activations[layer_idx].T, curr_delta)
                    + self.l2_reg * self.weights[layer_idx]
                )
                db = np.sum(curr_delta, axis=0)

                dW_list.insert(0, dW)
                db_list.insert(0, db)

                if layer_idx > 0:
                    curr_delta = np.dot(curr_delta, self.weights[layer_idx].T)

            # Pack gradients
            grad_vec = []
            for dW, db in zip(dW_list, db_list):
                grad_vec.append(dW.ravel())
                grad_vec.append(db.ravel())

            return neg_loss, np.concatenate(grad_vec)

        # Initial param vector
        init_vec = self._pack_params()

        res = opt.minimize(
            _objective_and_grad,
            init_vec,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-7},
        )

        self._unpack_params(res.x, in_dim)

        # Fit Breslow baseline cumulative hazard on trained neural risk predictions
        g_risk_fit, _, _ = self._forward(X_mat)
        exp_risk_fit = np.exp(g_risk_fit)

        event_times = y_time[y_event == 1]
        unique_event_times = np.unique(event_times)

        cum_hazard = []
        h0_cum = 0.0

        for t in unique_event_times:
            d_t = np.sum((y_time == t) & (y_event == 1))
            at_risk_mask = y_time >= t
            sum_risk = np.sum(exp_risk_fit[at_risk_mask])
            if sum_risk > 0:
                h0_cum += d_t / sum_risk
            cum_hazard.append(h0_cum)

        self.unique_event_times = unique_event_times
        self.baseline_cum_hazard = np.array(cum_hazard)

        # Compute feature importances
        self._compute_feature_importance(X, y_time, y_event)

        return self

    def _compute_feature_importance(
        self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray
    ):
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

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns feature importances dataframe."""
        if self.feature_importances_ is None:
            return pd.DataFrame()

        df_imp = (
            pd.DataFrame(
                [
                    {"feature": k, "importance (VIMP)": v}
                    for k, v in self.feature_importances_.items()
                ]
            )
            .sort_values(by="importance (VIMP)", ascending=False)
            .reset_index(drop=True)
        )
        return df_imp

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts log-risk score g_theta(X)."""
        if len(self.weights) == 0:
            raise ValueError("Model must be fitted before calling predict_risk().")
        X_mat = X[self.feature_names].values.astype(float)
        g_risk, _, _ = self._forward(X_mat)
        return g_risk

    def predict_survival(self, X: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
        """Predicts survival matrix S(t | X_i) = exp(-H0(t) * exp(g_theta(X_i)))."""
        if len(self.weights) == 0 or len(self.unique_event_times) == 0:
            raise ValueError("Model must be fitted before calling predict_survival().")

        risk_scores = self.predict_risk(X)
        exp_risk = np.exp(risk_scores)

        indices = np.searchsorted(self.unique_event_times, eval_times, side="right") - 1
        h0_eval = np.zeros_like(eval_times, dtype=float)

        valid_mask = indices >= 0
        h0_eval[valid_mask] = self.baseline_cum_hazard[
            np.minimum(indices[valid_mask], len(self.baseline_cum_hazard) - 1)
        ]

        cum_haz_matrix = np.outer(exp_risk, h0_eval)
        surv_matrix = np.exp(-cum_haz_matrix)
        return np.clip(surv_matrix, 1e-6, 1.0)

    def get_summary(self) -> pd.DataFrame:
        """Returns neural network architecture summary."""
        records = [
            {
                "layer": "Input Layer",
                "shape": f"({len(self.feature_names)}, {self.hidden_dims[0]})",
            },
            {
                "layer": "Hidden Layer 1 (SELU)",
                "shape": f"({self.hidden_dims[0]}, {self.hidden_dims[1]})",
            },
            {
                "layer": "Output Layer (Log-Risk)",
                "shape": f"({self.hidden_dims[1]}, 1)",
            },
            {
                "layer": "Total Loss Iterations",
                "shape": str(len(self.training_loss_history)),
            },
        ]
        return pd.DataFrame(records)
