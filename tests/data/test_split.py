import numpy as np
import pandas as pd
import pytest

from src.data.split import StratifiedSurvivalSplitter


@pytest.fixture
def dummy_survival_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "subject_id": [f"subj_{i}" for i in range(n)],
            "time": np.random.uniform(10, 100, size=n),
            "event": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "feature1": np.random.randn(n),
        }
    )
    return df


def test_deterministic_folds(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    folds1 = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    splitter2 = StratifiedSurvivalSplitter(random_state=42)
    folds2 = splitter2.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    for f1, f2 in zip(folds1, folds2):
        assert f1["train_indices"] == f2["train_indices"]
        assert f1["val_indices"] == f2["val_indices"]


def test_different_seeds_produce_different_folds(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    folds1 = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    splitter2 = StratifiedSurvivalSplitter(random_state=99)
    folds2 = splitter2.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    # Check that at least the first fold is different
    assert folds1[0]["train_indices"] != folds2[0]["train_indices"]


def test_validation_folds_mutually_exclusive_and_exhaustive(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    n_splits = 25
    folds = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=n_splits)

    all_val_indices = []
    for f in folds:
        all_val_indices.extend(f["val_indices"])

    # mutually exclusive: length of list should equal length of set
    assert len(all_val_indices) == len(set(all_val_indices))

    # exhaustive: every row appears exactly once
    assert set(all_val_indices) == set(dummy_survival_data.index.values)


def test_train_val_disjoint(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    folds = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    for f in folds:
        train_set = set(f["train_indices"])
        val_set = set(f["val_indices"])
        assert train_set.isdisjoint(val_set)


def test_event_stratification_preserved(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    n_splits = 25
    folds = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=n_splits)

    global_event_rate = dummy_survival_data["event"].mean()

    for f in folds:
        val_idx = f["val_indices"]
        val_df = dummy_survival_data.loc[val_idx]
        val_event_rate = val_df["event"].mean()

        # approximate preservation (within 0.1 for a reasonably sized dataset)
        assert abs(global_event_rate - val_event_rate) < 0.1


def test_select_fold_invalid_numbers_raise(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    folds = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    with pytest.raises(ValueError, match="fold must be between 0 and 4"):
        StratifiedSurvivalSplitter.select_fold(dummy_survival_data, folds, 5)

    with pytest.raises(ValueError):
        StratifiedSurvivalSplitter.select_fold(dummy_survival_data, folds, -1)

    with pytest.raises(TypeError):
        StratifiedSurvivalSplitter.select_fold(dummy_survival_data, folds, "0")


def test_select_fold_returns_correct_dataframes(dummy_survival_data):
    splitter = StratifiedSurvivalSplitter(random_state=42)
    folds = splitter.create_cv_folds(dummy_survival_data, "event", n_splits=5)

    train_df, val_df = StratifiedSurvivalSplitter.select_fold(
        dummy_survival_data, folds, 0
    )

    assert len(train_df) == len(folds[0]["train_indices"])
    assert len(val_df) == len(folds[0]["val_indices"])
    assert "subject_id" in train_df.columns
    assert "subject_id" in val_df.columns
