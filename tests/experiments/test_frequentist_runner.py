from src.experiments.frequentist_manifest import FrequentistCell
from src.experiments.frequentist_runner import FrequentistRunner


def test_frequentist_runner_cox_ph():
    cell = FrequentistCell(
        experiment_id="test_freq",
        dataset="GBSG2",
        model_type="cox_ph",
        fold=0,
        seed=42,
        model_params={"l2_reg": 1e-4},
    )
    runner = FrequentistRunner()
    result = runner.run_cell(cell)

    assert result["status"] == "PASS"
    assert "c_index" in result["metrics"]
    assert len(result["predictions"]) == result["n_validation"]
    assert len(result["survival"]) == result["n_validation"]
