import numpy as np
import pandas as pd
import pytest

from src.data.loader import DatasetLoader
from src.data.preprocessing import SurvivalDataPipeline
from src.models.bayesian.model import BayesianCoxModel


# Fixture to load and preprocess a small chunk of data for each dataset
@pytest.fixture(scope="module", params=["gbsg2", "whas500", "metabric"])
def dataset_data(request, tmp_path_factory):
    dataset_name = request.param
    raw = DatasetLoader.load_raw("data", dataset_name).head(100)  # fast

    cat_cols = []
    ord_cols = None
    if dataset_name == "gbsg2":
        time_col, event_col = "time", "cens"
        cat_cols = ["horTh", "menostat"]
    elif dataset_name == "whas500":
        time_col, event_col = "lenfol", "fstat"
        cat_cols = ["gender"]
    elif dataset_name == "metabric":
        time_col, event_col = "duration", "event"
        cat_cols = ["PAM50Subtype"]

    tmp = tmp_path_factory.mktemp("data")
    pipeline = SurvivalDataPipeline(
        dataset_name=dataset_name,
        time_col=time_col,
        event_col=event_col,
        categorical_cols=cat_cols,
        ordinal_cols=ord_cols,
        random_state=42,
    )
    pipeline.run(raw, output_dir=tmp)

    df = pd.read_csv(tmp / "processed" / dataset_name / "train.csv")
    X = df.drop(columns=["time", "event", "subject_id"], errors="ignore")
    y_time = df["time"].to_numpy(dtype=float)
    y_event = df["event"].to_numpy(dtype=int)

    return X, y_time, y_event


@pytest.mark.parametrize("baseline_prior", ["gamma", "lognormal", "spline"])
def test_baseline_hazard_matrix(dataset_data, baseline_prior):
    X, y_time, y_event = dataset_data

    model = BayesianCoxModel(
        inference_method="mcmc",
        draws=5,
        tune=5,
        chains=2,
        random_state=42,
        baseline_hazard_prior=baseline_prior,
        baseline_hazard_params={"df": 6, "n_grid": 50, "sigma": 1.0}
        if baseline_prior == "spline"
        else {},
        n_intervals=2,
    )

    # 1. Fit
    model.fit(X, y_time, y_event)

    # 2. Posterior Check
    assert model.beta_samples is not None
    if baseline_prior == "spline":
        assert model.gamma_samples is not None
    else:
        assert model.lambda_samples is not None

    # 3. Risk Prediction
    risk = model.predict_risk(X)
    assert risk.shape == (len(X),)

    # 4. Survival and Credible Intervals
    eval_times = np.array([np.median(y_time), np.max(y_time)])
    surv_mean, surv_lower, surv_upper = model.predict_survival_with_credible_intervals(
        X, eval_times
    )

    assert surv_mean.shape == (len(X), 2)
    assert surv_lower.shape == (len(X), 2)
    assert surv_upper.shape == (len(X), 2)

    assert np.all((surv_mean >= 0) & (surv_mean <= 1))
    assert np.all((surv_lower >= 0) & (surv_lower <= 1))
    assert np.all((surv_upper >= 0) & (surv_upper <= 1))
    assert np.all(surv_lower <= surv_mean)
    assert np.all(surv_mean <= surv_upper)
