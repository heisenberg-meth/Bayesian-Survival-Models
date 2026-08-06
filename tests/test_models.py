import numpy as np
import pandas as pd

from src.models.deepsurv import DeepSurvModel
from src.models.random_survival_forest import RandomSurvivalForestModel, SurvivalTree


def test_survival_tree_basic():
    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y_time = np.random.exponential(scale=10.0, size=50)
    y_event = np.random.binomial(n=1, p=0.8, size=50)
    all_event_times = np.sort(np.unique(y_time[y_event == 1]))

    # Train tree
    tree = SurvivalTree(max_depth=3, min_samples_split=5, min_samples_leaf=2)
    tree.fit(X, y_time, y_event, all_event_times)

    # Predict cumulative hazard
    chaz = tree.predict_chaz(X)
    assert chaz.shape == (50, len(all_event_times))
    assert np.all(chaz >= 0.0)


def test_rsf_model():
    # Create simple synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["x1", "x2", "x3", "x4"])
    y_time = np.random.exponential(scale=10.0, size=60)
    y_event = np.random.binomial(n=1, p=0.8, size=60)

    # Train RSF
    model = RandomSurvivalForestModel(
        n_estimators=10,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=42,
    )
    model.fit(X, y_time, y_event)

    # Risk predictions
    risk = model.predict_risk(X)
    assert risk.shape == (60,)
    assert not np.any(np.isnan(risk))

    # Survival predictions
    eval_times = np.percentile(y_time[y_event == 1], [25, 50, 75])
    surv = model.predict_survival(X, eval_times)
    assert surv.shape == (60, len(eval_times))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0)

    # Check importance
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 4


def test_deepsurv_model():
    # Create simple synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["x1", "x2", "x3", "x4"])
    y_time = np.random.exponential(scale=10.0, size=60) + 1.0  # times > 0
    y_event = np.random.binomial(n=1, p=0.8, size=60)

    # Train DeepSurv
    model = DeepSurvModel(
        hidden_dims=[8, 4],
        l2_reg=1e-3,
        random_state=42,
        max_iter=50,
    )
    model.fit(X, y_time, y_event)

    # Risk predictions
    risk = model.predict_risk(X)
    assert risk.shape == (60,)
    assert not np.any(np.isnan(risk))

    # Survival predictions
    eval_times = np.percentile(y_time[y_event == 1], [25, 50, 75])
    surv = model.predict_survival(X, eval_times)
    assert surv.shape == (60, len(eval_times))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0)

    # Check feature importances
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 4
    imp_df = model.get_feature_importances()
    assert imp_df.shape == (4, 2)
    assert list(imp_df.columns) == ["feature", "importance (VIMP)"]


def test_bayesian_cox_model():
    from src.models.bayesian.model import BayesianCoxModel

    # Create simple synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["x1", "x2", "x3", "x4"])
    y_time = np.random.exponential(scale=10.0, size=60) + 1.0  # times > 0
    y_event = np.random.binomial(n=1, p=0.8, size=60)

    # Train Bayesian Cox with ADVI
    model = BayesianCoxModel(
        n_intervals=4,
        inference_method="advi",
        n_advi_iterations=200,
        draws=100,
        random_state=42,
        coefficient_prior="student-t",
        prior_params={"nu": 3.0, "sigma": 2.0},
    )
    model.fit(X, y_time, y_event)

    # Risk predictions
    risk = model.predict_risk(X)
    assert risk.shape == (60,)
    assert not np.any(np.isnan(risk))

    # Survival predictions with credible intervals
    eval_times = np.percentile(y_time[y_event == 1], [25, 50, 75])
    surv, surv_low, surv_up = model.predict_survival_with_credible_intervals(
        X, eval_times
    )
    assert surv.shape == (60, len(eval_times))
    assert surv_low.shape == (60, len(eval_times))
    assert surv_up.shape == (60, len(eval_times))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0)
    assert np.all(surv_low <= surv) and np.all(surv <= surv_up)

    # Check summary
    summary_df = model.get_summary()
    assert summary_df.shape == (4, 8)
    assert "exp(coef) HR Mean" in summary_df.columns
    assert "95% Credible Lower" in summary_df.columns
    assert "95% Credible Upper" in summary_df.columns

    # Train Bayesian Cox with MCMC (Metropolis/NUTS) to check diagnostics
    model_mcmc = BayesianCoxModel(
        n_intervals=4,
        inference_method="mcmc",
        draws=50,
        tune=50,
        chains=2,
        random_state=42,
        coefficient_prior="laplace",
        prior_params={"b": 1.5},
    )
    model_mcmc.fit(X, y_time, y_event)

    diag_df = model_mcmc.get_mcmc_diagnostics()
    assert "r_hat" in diag_df.columns or "rhat" in diag_df.columns.str.lower()
    assert "ess_bulk" in diag_df.columns or "ess" in diag_df.columns.str.lower()
    assert "feature" in diag_df.columns
