import pytest

from src.experiments.config_resolver import resolve_bayesian_config


def test_resolve_gbsg2_normal():
    cfg = resolve_bayesian_config("GBSG2", "Normal")
    assert cfg["coefficient_prior"] == "normal"
    assert cfg["beta_prior_sd"] == 10.0
    assert cfg["baseline_hazard_prior"] == "gamma"
    assert cfg["draws"] == 2000
    assert cfg["target_accept"] == 0.95


def test_resolve_whas500_horseshoe():
    cfg = resolve_bayesian_config("WHAS500", "Regularised Horseshoe")
    assert cfg["coefficient_prior"] == "regularized_horseshoe"
    assert cfg["beta_prior_sd"] == 5.0
    assert cfg["baseline_hazard_prior"] == "gamma"
    assert cfg["draws"] == 2000
    assert cfg["target_accept"] == 0.95


def test_resolve_metabric_spike_slab():
    cfg = resolve_bayesian_config("METABRIC", "Continuous Spike-and-Slab")
    assert cfg["coefficient_prior"] == "continuous_spike_slab"
    assert cfg["beta_prior_sd"] == 2.5
    assert cfg["baseline_hazard_prior"] == "spline"
    assert cfg["draws"] == 3000
    assert cfg["tune"] == 1500
    assert cfg["target_accept"] == 0.98


def test_resolve_invalid_dataset():
    with pytest.raises(FileNotFoundError):
        resolve_bayesian_config("INVALID", "Normal")
