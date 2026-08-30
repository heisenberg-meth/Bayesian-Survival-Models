import numpy as np
import pandas as pd
import pytest

from src.models.cox import CoxPHModel
from src.models.deepsurv import DeepSurvModel
from src.models.random_survival_forest import RandomSurvivalForestModel


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "age": np.random.normal(50, 10, n_samples),
            "biomarker": np.random.normal(0, 1, n_samples),
        }
    )

    # Generate survival times
    risk = X["age"] * 0.05 + X["biomarker"] * 0.5
    scale = np.exp(-risk)
    y_time = np.random.exponential(scale * 10)

    # Generate censoring
    censor_time = np.random.exponential(20, n_samples)

    y_event = (y_time <= censor_time).astype(int)
    y_time = np.minimum(y_time, censor_time)

    return X, y_time, y_event


@pytest.mark.parametrize(
    "model_class", [CoxPHModel, RandomSurvivalForestModel, DeepSurvModel]
)
def test_frequentist_model_interface(dummy_data, model_class):
    X, y_time, y_event = dummy_data

    if model_class == RandomSurvivalForestModel:
        model = model_class(n_estimators=10, random_state=42)
    elif model_class == DeepSurvModel:
        model = model_class(max_iter=50, random_state=42)
    else:
        model = model_class()

    # Test fit
    model.fit(X, y_time, y_event)

    # Test predict_risk
    risk = model.predict_risk(X)
    assert risk.shape == (len(X),)
    assert not np.isnan(risk).any()

    # Test predict_survival
    eval_times = np.array([5.0, 10.0, 15.0])
    surv = model.predict_survival(X, eval_times)

    assert surv.shape == (len(X), len(eval_times))
    assert np.all((surv >= 0) & (surv <= 1))

    # Survival should be non-increasing over time
    assert np.all(np.diff(surv, axis=1) <= 1e-6)

    # Test get_summary
    summary = model.get_summary()
    assert isinstance(summary, pd.DataFrame)
