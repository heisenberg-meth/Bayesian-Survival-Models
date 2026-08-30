import pytest
from src.experiments.config_resolver import resolve_bayesian_config
from src.experiments.manifest import ExperimentCell
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager

def test_integrity():
    resolved = resolve_bayesian_config("METABRIC", "Continuous Spike-and-Slab")
    # assert the config has baseline_hazard_params
    assert "baseline_hazard_params" in resolved
    print("Resolved baseline params:", resolved["baseline_hazard_params"])
    
    cell = ExperimentCell(
        experiment_id="test",
        dataset="METABRIC",
        prior="Continuous Spike-and-Slab",
        fold=0,
        seed=42,
        **resolved
    )
    assert cell.baseline_hazard_params is not None
    print("Cell baseline params:", cell.baseline_hazard_params)
    print("ALL OK")

test_integrity()
