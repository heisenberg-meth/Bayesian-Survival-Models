import os
from typing import Any

from src.utils.config import load_yaml


def resolve_bayesian_config(
    dataset: str, prior_name: str, experiments_dir: str = "experiments"
) -> dict[str, Any]:
    """
    Resolve dataset-specific Bayesian configuration.
    Accepts dataset (e.g., 'GBSG2') and prior_name (e.g., 'Normal').
    """
    yaml_path = os.path.join(experiments_dir, f"{dataset.lower()}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Experiment config not found for dataset {dataset} at {yaml_path}"
        )

    cfg = load_yaml(yaml_path)

    model_cfg = cfg.get("model", {})
    priors = model_cfg.get("priors", {})
    sampler = model_cfg.get("sampler", {})

    prior_map = {
        "Normal": "normal",
        "Regularised Horseshoe": "regularized_horseshoe",
        "Continuous Spike-and-Slab": "continuous_spike_slab",
    }

    resolved = {
        "coefficient_prior": prior_map.get(prior_name, "normal"),
        "beta_prior_mean": float(priors.get("beta_prior_mean", 0.0)),
        "beta_prior_sd": float(priors.get("beta_prior_sd", 10.0)),
        "baseline_hazard_prior": priors.get("baseline_hazard_prior", "gamma"),
        "baseline_hazard_params": priors.get("baseline_hazard_params", {}),
        "draws": int(sampler.get("draws", 2000)),
        "tune": int(sampler.get("tune", 1000)),
        "chains": int(sampler.get("chains", 4)),
        "target_accept": float(sampler.get("target_accept", 0.95)),
        "n_intervals": int(model_cfg.get("n_intervals", 6)),
    }

    return resolved
