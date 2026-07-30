import numpy as np
import pandas as pd

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
