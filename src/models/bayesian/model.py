"""
Full Bayesian Cox Proportional Hazards Model Implementation in PyMC.
Uses Vectorized Piecewise Exponential Baseline Hazard formulation with Poisson likelihood equivalence.
Supports ADVI (Automatic Differentiation Variational Inference) and MCMC (NUTS/Metropolis) sampling.
Provides posterior distributions of Hazard Ratios, epistemic parameter uncertainty, and credible interval survival predictions.
"""

import numpy as np
import pandas as pd
import pymc as pm

from src.models.base import BaseSurvivalModel


class BayesianCoxModel(BaseSurvivalModel):
    """Bayesian Cox Proportional Hazards Model with PyMC Variational & MCMC Inference."""

    def __init__(
        self,
        n_intervals: int = 6,
        inference_method: str = "advi",  # 'advi' or 'mcmc'
        n_advi_iterations: int = 10000,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        random_state: int = 42,
        coefficient_prior: str = "normal",
        prior_params: dict | None = None,
        baseline_hazard_prior: str = "gamma",
        baseline_hazard_params: dict | None = None,
    ):
        self.n_intervals = n_intervals
        self.inference_method = inference_method.lower()
        self.n_advi_iterations = n_advi_iterations
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_state = random_state
        self.coefficient_prior = coefficient_prior.lower()
        self.prior_params = prior_params if prior_params is not None else {}
        self.baseline_hazard_prior = baseline_hazard_prior.lower()
        self.baseline_hazard_params = (
            baseline_hazard_params if baseline_hazard_params is not None else {}
        )

        self.feature_names = []
        self.cutoffs = np.array([])
        self.pymc_model = None
        self.idata = None  # InferenceData object
        self.summary_df = None

        self.beta_samples = None  # (S_draws, p_features)
        self.lambda_samples = None  # (S_draws, M_intervals)

    def _prepare_piecewise_data(
        self, X_mat: np.ndarray, y_time: np.ndarray, y_event: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts survival data to sample-interval exposures and event counts using fast 2D NumPy operations.
        """
        _n_samples, _n_features = X_mat.shape

        event_times = y_time[y_event == 1]
        if len(event_times) == 0:
            event_times = y_time

        percentiles = np.linspace(0, 100, self.n_intervals + 1)
        cutoffs = np.percentile(event_times, percentiles)
        cutoffs[0] = 0.0
        cutoffs[-1] = max(y_time.max() + 1.0, cutoffs[-1])
        cutoffs = np.unique(cutoffs)
        if len(cutoffs) - 1 < self.n_intervals:
            cutoffs = np.linspace(0, y_time.max() + 1.0, self.n_intervals + 1)

        len(cutoffs) - 1

        t_starts = cutoffs[:-1]  # (M,)
        t_ends = cutoffs[1:]  # (M,)

        t_i = y_time[:, None]  # (N, 1)
        e_i = y_event[:, None]  # (N, 1)

        # Vectorized exposure calculation: max(0, min(t_i, t_end) - t_start)
        exposures = np.clip(np.minimum(t_i, t_ends) - t_starts, 0, None)

        # Vectorized event matrix: 1 if event and t_start < t_i <= t_end
        events_matrix = ((e_i == 1) & (t_i > t_starts) & (t_i <= t_ends)).astype(float)

        return cutoffs, exposures, events_matrix

    def fit(
        self, X: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray
    ) -> "BayesianCoxModel":
        """Fits Bayesian Cox Model using PyMC ADVI Variational Inference or MCMC Sampling."""
        self.feature_names = list(X.columns)
        X_mat = X.values.astype(float)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        _n_samples, n_features = X_mat.shape

        cutoffs, exposures, events_matrix = self._prepare_piecewise_data(
            X_mat, y_time, y_event
        )
        self.cutoffs = cutoffs
        M = len(cutoffs) - 1

        valid_mask = exposures > 0
        obs_events = events_matrix[valid_mask]
        obs_exposures = exposures[valid_mask]

        sample_idx, interval_idx = np.where(valid_mask)
        obs_X = X_mat[sample_idx]
        log_exposure = np.log(obs_exposures)

        with pm.Model() as model:
            # Priors for regression coefficients.
            #
            # All prior families expose the same final latent coefficient
            # vector `beta` so that the likelihood and posterior extraction
            # remain unchanged.

            prior_mu = self.prior_params.get("mu", 0.0)
            prior_sigma = self.prior_params.get("sigma", 1.0)

            if self.coefficient_prior == "normal":
                beta = pm.Normal(
                    "beta",
                    mu=prior_mu,
                    sigma=prior_sigma,
                    shape=n_features,
                )

            elif self.coefficient_prior == "student-t":
                prior_nu = self.prior_params.get("nu", 3.0)

                beta = pm.StudentT(
                    "beta",
                    nu=prior_nu,
                    mu=prior_mu,
                    sigma=prior_sigma,
                    shape=n_features,
                )

            elif self.coefficient_prior == "laplace":
                prior_b = self.prior_params.get("b", 1.0)

                beta = pm.Laplace(
                    "beta",
                    mu=prior_mu,
                    b=prior_b,
                    shape=n_features,
                )

            elif self.coefficient_prior == "regularized_horseshoe":
                # Regularised horseshoe:
                # beta_j = tau * lambda_j * c / sqrt(c^2 + tau^2*lambda_j^2)
                #          * z_j
                #
                # tau      : global shrinkage
                # lambda_j : local shrinkage
                # c        : slab scale
                # z_j      : standard normal coefficient.

                tau_scale = self.prior_params.get("tau_scale", 0.1)
                slab_scale = self.prior_params.get("slab_scale", 2.0)
                slab_df = self.prior_params.get("slab_df", 4.0)

                tau = pm.HalfNormal(
                    "tau",
                    sigma=tau_scale,
                )

                lambda_local = pm.HalfStudentT(
                    "lambda_local",
                    nu=4.0,
                    sigma=1.0,
                    shape=n_features,
                )

                c2 = pm.InverseGamma(
                    "slab_scale_squared",
                    alpha=slab_df / 2.0,
                    beta=(slab_df * slab_scale**2) / 2.0,
                )

                shrinkage = tau * lambda_local

                regularized_shrinkage = shrinkage * pm.math.sqrt(
                    c2 / (c2 + shrinkage**2)
                )

                z = pm.Normal(
                    "z",
                    mu=0.0,
                    sigma=1.0,
                    shape=n_features,
                )

                beta = pm.Deterministic(
                    "beta",
                    prior_mu + prior_sigma * regularized_shrinkage * z,
                )

            elif self.coefficient_prior == "continuous_spike_slab":
                # Continuous spike-and-slab prior.
                #
                # pi_j is a continuous inclusion probability. The coefficient
                # scale smoothly interpolates between a narrow spike and a
                # wider slab without using discrete Bernoulli indicators.

                spike_sd = self.prior_params.get("spike_sd", 0.1)
                slab_sd = self.prior_params.get(
                    "slab_sd",
                    prior_sigma,
                )
                inclusion_alpha = self.prior_params.get(
                    "inclusion_alpha",
                    1.0,
                )
                inclusion_beta = self.prior_params.get(
                    "inclusion_beta",
                    1.0,
                )

                inclusion_prob = pm.Beta(
                    "inclusion_prob",
                    alpha=inclusion_alpha,
                    beta=inclusion_beta,
                    shape=n_features,
                )

                coefficient_sd = pm.math.sqrt(
                    inclusion_prob * slab_sd**2 + (1.0 - inclusion_prob) * spike_sd**2
                )

                beta = pm.Normal(
                    "beta",
                    mu=prior_mu,
                    sigma=coefficient_sd,
                    shape=n_features,
                )

            else:
                raise ValueError(f"Unknown coefficient prior: {self.coefficient_prior}")

            # Baseline hazard prior centered on the observed event rate.
            #
            # The piecewise-exponential model uses:
            #   mu = exposure * lambda_m * exp(X beta)
            #
            # Parameterising lambda on the log scale keeps positivity while
            # centering the prior near the empirical event rate.
            total_exposure = float(np.sum(obs_exposures))
            total_events = float(np.sum(obs_events))

            empirical_rate = max(
                total_events / total_exposure,
                np.finfo(float).tiny,
            )

            log_lambda_center = float(np.log(empirical_rate))

            baseline_prior = self.baseline_hazard_prior

            if baseline_prior == "spline":
                from src.models.bayesian.splines import SplineBasis

                df = self.baseline_hazard_params.get("df", 8)
                spline = SplineBasis(df=df)
                spline.fit(y_time)
                self.spline_basis_ = spline

                B_exact = spline.transform(y_time)
                n_grid = self.baseline_hazard_params.get("n_grid", 200)
                B_grid, weights, mask = spline.get_integration_matrix(
                    y_time, n_grid=n_grid
                )

                gamma = pm.Normal(
                    "gamma",
                    mu=log_lambda_center,
                    sigma=self.baseline_hazard_params.get("sigma", 1.0),
                    shape=df,
                )

                log_lambda_exact = pm.math.dot(B_exact, gamma)
                lambda_grid = pm.math.exp(pm.math.dot(B_grid, gamma))
                H0_t = pm.math.dot(mask, lambda_grid * weights)

                eta = pm.math.dot(X_mat, beta)

                log_lik = pm.math.sum(
                    y_event * (log_lambda_exact + eta) - H0_t * pm.math.exp(eta)
                )
                pm.Potential("spline_likelihood", log_lik)

            else:
                if baseline_prior == "gamma":
                    alpha = self.baseline_hazard_params.get("alpha", 1.0)
                    rate = alpha / empirical_rate

                    lambda_m = pm.Gamma(
                        "lambda",
                        alpha=alpha,
                        beta=rate,
                        shape=M,
                    )

                elif baseline_prior == "lognormal":
                    log_lambda = pm.Normal(
                        "log_lambda",
                        mu=log_lambda_center,
                        sigma=1.0,
                        shape=M,
                    )
                    lambda_m = pm.math.exp(log_lambda)

                else:
                    raise ValueError(
                        f"Unknown baseline hazard prior: {self.baseline_hazard_prior}"
                    )

                eta = pm.math.dot(obs_X, beta)
                lambda_obs = lambda_m[interval_idx]

                log_mu = log_exposure + pm.math.log(lambda_obs) + eta

                pm.Poisson(
                    "obs",
                    mu=pm.math.exp(log_mu),
                    observed=obs_events,
                )

            if self.inference_method == "advi":
                approx = pm.fit(
                    method="advi",
                    n=self.n_advi_iterations,
                    progressbar=False,
                    random_seed=self.random_state,
                )
                self.idata = approx.sample(
                    draws=self.draws, random_seed=self.random_state
                )
            else:
                self.idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    cores=min(self.chains, 4),
                    target_accept=self.target_accept,
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True,
                )

        self.pymc_model = model

        # Extract posterior samples
        posterior = self.idata.posterior
        beta_raw = posterior["beta"].values  # (chains, draws, p)

        if self.baseline_hazard_prior == "spline":
            self.gamma_samples = posterior["gamma"].values.reshape(
                -1, self.spline_basis_.df
            )
            self.lambda_samples = None
        elif self.baseline_hazard_prior == "gamma":
            lambda_raw = posterior["lambda"].values
            self.lambda_samples = lambda_raw.reshape(-1, M)
        else:
            log_lambda_raw = posterior["log_lambda"].values
            self.lambda_samples = np.exp(log_lambda_raw.reshape(-1, M))

        self.beta_samples = beta_raw.reshape(-1, n_features)

        # Build Summary DataFrame for Hazard Ratios
        summary_records = []
        for j, feat in enumerate(self.feature_names):
            b_j = self.beta_samples[:, j]
            hr_j = np.exp(b_j)

            summary_records.append(
                {
                    "feature": feat,
                    "coef (beta_mean)": float(np.mean(b_j)),
                    "exp(coef) HR Mean": float(np.mean(hr_j)),
                    "exp(coef) HR Median": float(np.median(hr_j)),
                    "HR SD": float(np.std(hr_j)),
                    "95% Credible Lower": float(np.percentile(hr_j, 2.5)),
                    "95% Credible Upper": float(np.percentile(hr_j, 97.5)),
                    "Prob(HR > 1)": float(np.mean(hr_j > 1.0)),
                }
            )

        self.summary_df = (
            pd.DataFrame(summary_records)
            .sort_values(by="exp(coef) HR Mean", ascending=False)
            .reset_index(drop=True)
        )
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts posterior mean linear risk score eta = X * mean(beta)."""
        if self.beta_samples is None:
            raise ValueError("Model must be fitted before calling predict_risk().")
        X_mat = X[self.feature_names].values.astype(float)
        mean_beta = np.mean(self.beta_samples, axis=0)
        return np.dot(X_mat, mean_beta)

    def predict_survival(self, X: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
        """Predicts posterior mean survival probability matrix S(t | X_i)."""
        surv_mean, _, _ = self.predict_survival_with_credible_intervals(X, eval_times)
        return surv_mean

    def predict_survival_with_credible_intervals(
        self, X: pd.DataFrame, eval_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Posterior Mean Survival Probability Matrix along with 95% Credible Interval Bounds.
        Returns: (surv_mean, surv_lower_95, surv_upper_95)
        """
        if self.beta_samples is None:
            raise ValueError("Model must be fitted before calling predict_survival().")

        if self.baseline_hazard_prior != "spline" and len(self.cutoffs) == 0:
            raise ValueError("Piecewise model must have cutoffs fitted.")

        X_mat = X[self.feature_names].values.astype(float)
        n_samples = len(X_mat)
        m_times = len(eval_times)
        S_draws = len(self.beta_samples)

        H0_draws = np.zeros((S_draws, m_times), dtype=float)

        if self.baseline_hazard_prior == "spline":
            n_grid = self.baseline_hazard_params.get("n_grid", 200)
            B_grid, weights, mask = self.spline_basis_.get_integration_matrix(
                eval_times, n_grid=n_grid
            )

            for s in range(S_draws):
                gamma_s = self.gamma_samples[s]
                lambda_grid = np.exp(np.dot(B_grid, gamma_s))
                H0_draws[s] = np.dot(mask, lambda_grid * weights)

        else:
            M = len(self.cutoffs) - 1
            durations = np.diff(self.cutoffs)

            for s in range(S_draws):
                lambdas = self.lambda_samples[s]
                cum_h_endpoints = np.cumsum(lambdas * durations)
                cum_h_endpoints = np.insert(cum_h_endpoints, 0, 0.0)

                for j, t in enumerate(eval_times):
                    idx = np.searchsorted(self.cutoffs, t, side="right") - 1
                    if idx < 0:
                        H0_draws[s, j] = 0.0
                    elif idx >= M:
                        H0_draws[s, j] = cum_h_endpoints[M] + lambdas[-1] * (
                            t - self.cutoffs[-1]
                        )
                    else:
                        H0_draws[s, j] = cum_h_endpoints[idx] + lambdas[idx] * (
                            t - self.cutoffs[idx]
                        )

        risk_draws = np.dot(self.beta_samples, X_mat.T)  # (S_draws, N_samples)
        exp_risk_draws = np.exp(risk_draws)

        surv_draws = np.zeros((S_draws, n_samples, m_times), dtype=float)

        for s in range(S_draws):
            cum_h_si = np.outer(exp_risk_draws[s], H0_draws[s])
            surv_draws[s] = np.exp(-cum_h_si)

        surv_mean = np.clip(np.mean(surv_draws, axis=0), 1e-6, 1.0)
        surv_lower = np.clip(np.percentile(surv_draws, 2.5, axis=0), 1e-6, 1.0)
        surv_upper = np.clip(np.percentile(surv_draws, 97.5, axis=0), 1e-6, 1.0)

        return surv_mean, surv_lower, surv_upper

    def get_mcmc_diagnostics(self) -> pd.DataFrame:
        """
        Computes Gelman-Rubin (R-hat) and Effective Sample Size (ESS) diagnostics for regression parameters.
        Only valid if inference_method == 'mcmc' or if MCMC chains exist.
        """
        import re

        import arviz as az

        if self.idata is None:
            return pd.DataFrame()
        if "posterior" not in self.idata:
            return pd.DataFrame()

        summary = az.summary(self.idata, var_names=["beta"])
        summary = summary.reset_index()
        summary.columns = ["parameter"] + list(summary.columns[1:])

        def map_param_name(param_str):
            match = re.search(r"beta\[(\d+)\]", param_str)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(self.feature_names):
                    return self.feature_names[idx]
            return param_str

        summary["feature"] = summary["parameter"].apply(map_param_name)
        return summary

    def get_summary(self) -> pd.DataFrame:
        """Returns Posterior Summary DataFrame for Hazard Ratios and Coefficients."""
        return self.summary_df if self.summary_df is not None else pd.DataFrame()
