from src.experiments.manifest import ExperimentCell
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager


def make_cell() -> ExperimentCell:
    return ExperimentCell(
        experiment_id="runner_test",
        dataset="GBSG2",
        prior="Normal",
        fold=0,
        seed=42,
    )


def test_runner_executes_and_completes(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    expected = {
        "cell_id": make_cell().cell_id,
        "status": "PASS",
    }

    runner._execute = lambda cell: expected

    result = runner.run(make_cell())

    assert result == expected
    assert manager.is_complete(make_cell())

    checkpoint = manager.load_checkpoint(make_cell())

    assert checkpoint["status"] == "complete"
    assert "result" in checkpoint["artifacts"]


def test_completed_cell_is_not_executed_again(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    calls = []

    def execute(cell):
        calls.append(cell.cell_id)
        return {
            "cell_id": cell.cell_id,
            "status": "PASS",
        }

    runner._execute = execute

    cell = make_cell()

    first = runner.run(cell)
    second = runner.run(cell)

    assert first == second
    assert calls == [cell.cell_id]


def test_failed_cell_is_marked_failed(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    def execute(_cell):
        raise RuntimeError("synthetic failure")

    runner._execute = execute

    cell = make_cell()

    try:
        runner.run(cell)
    except RuntimeError as exc:
        assert str(exc) == "synthetic failure"
    else:
        raise AssertionError("Expected RuntimeError")

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "failed"


def test_runner_uses_cell_identity(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    runner._execute = lambda cell: {
        "cell_id": cell.cell_id,
        "status": "PASS",
    }

    cell_a = make_cell()

    cell_b = ExperimentCell(
        experiment_id="runner_test",
        dataset="GBSG2",
        prior="Normal",
        fold=0,
        seed=43,
    )

    runner.run(cell_a)
    runner.run(cell_b)

    assert manager.is_complete(cell_a)
    assert manager.is_complete(cell_b)

    assert manager._get_checkpoint_path(cell_a) != manager._get_checkpoint_path(cell_b)


def test_failed_diagnostics_are_not_marked_complete(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    cell = make_cell()

    runner._execute = lambda _cell: {
        "cell_id": cell.cell_id,
        "status": "FAIL",
        "diagnostics": {
            "status": "FAIL",
            "rhat_max": 1.25,
            "ess_bulk_min": 100.0,
            "divergences": 5,
            "bfmi_min": 0.1,
            "tree_depth_max": 10,
        },
    }

    result = runner.run(cell)

    assert result["status"] == "FAIL"
    assert not manager.is_complete(cell)

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "failed_diagnostics"


def test_passing_diagnostics_are_marked_complete(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    cell = make_cell()

    runner._execute = lambda _cell: {
        "cell_id": cell.cell_id,
        "status": "PASS",
        "diagnostics": {
            "status": "PASS",
            "rhat_max": 1.001,
            "ess_bulk_min": 1000.0,
            "divergences": 0,
            "bfmi_min": 0.8,
            "tree_depth_max": 8,
        },
    }

    result = runner.run(cell)

    assert result["status"] == "PASS"
    assert manager.is_complete(cell)

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "complete"


def test_different_folds_use_different_validation_rows(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    df = __import__("pandas").DataFrame(
        {
            "value": range(20),
            "time": range(20),
            "event": [0, 1] * 10,
        }
    )

    folds = [
        {
            "fold": 1,
            "train_indices": list(range(10, 20)),
            "val_indices": list(range(10)),
        },
        {
            "fold": 2,
            "train_indices": list(range(10)),
            "val_indices": list(range(10, 20)),
        },
    ]

    train_a, val_a = runner._select_fold(df, folds, 0)
    train_b, val_b = runner._select_fold(df, folds, 1)

    assert set(val_a["value"]) != set(val_b["value"])
    assert set(train_a["value"]) != set(train_b["value"])

    assert set(val_a["value"]).isdisjoint(set(train_a["value"]))
    assert set(val_b["value"]).isdisjoint(set(train_b["value"]))


def test_invalid_fold_is_rejected(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    df = __import__("pandas").DataFrame({"value": range(10)})

    folds = [
        {
            "fold": 1,
            "train_indices": list(range(5)),
            "val_indices": list(range(5, 10)),
        }
    ]

    try:
        runner._select_fold(df, folds, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid fold must raise ValueError")


def test_runner_preserves_fold_identity_and_checkpoint(
    tmp_path,
):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    executed = []

    def execute(cell):
        executed.append(cell)

        return {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "dataset": cell.dataset,
            "prior": cell.prior,
            "fold": cell.fold,
            "seed": cell.seed,
            "status": "PASS",
        }

    runner._execute = execute

    cells = [
        ExperimentCell(
            experiment_id="integration_v1",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=42,
        ),
        ExperimentCell(
            experiment_id="integration_v1",
            dataset="GBSG2",
            prior="Normal",
            fold=1,
            seed=42,
        ),
        ExperimentCell(
            experiment_id="integration_v1",
            dataset="GBSG2",
            prior="Normal",
            fold=2,
            seed=42,
        ),
    ]

    results = [runner.run(cell) for cell in cells]

    assert len(executed) == 3

    assert [result["fold"] for result in results] == [0, 1, 2]

    for cell in cells:
        checkpoint = manager.load_checkpoint(cell)

        assert checkpoint["cell_id"] == cell.cell_id
        assert checkpoint["fold"] == cell.fold
        assert checkpoint["seed"] == cell.seed
        assert checkpoint["status"] == "complete"
        assert "result" in checkpoint["artifacts"]

    assert manager._get_checkpoint_path(cells[0]) != manager._get_checkpoint_path(
        cells[1]
    )

    assert manager._get_checkpoint_path(cells[1]) != manager._get_checkpoint_path(
        cells[2]
    )


def test_runner_resume_does_not_reexecute_completed_cell(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    calls = []

    def execute(cell):
        calls.append(cell.cell_id)

        return {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "fold": cell.fold,
            "status": "PASS",
        }

    runner._execute = execute

    cell = ExperimentCell(
        experiment_id="resume_v1",
        dataset="GBSG2",
        prior="Normal",
        fold=2,
        seed=42,
    )

    first = runner.run(cell)
    second = runner.run(cell)

    assert first == second
    assert calls == [cell.cell_id]

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "complete"
    assert checkpoint["completed_at"] is not None


def test_failed_cell_can_be_retried(tmp_path):
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    runner = ExperimentRunner(manager)

    calls = []

    def fail_once(cell):
        calls.append(cell.cell_id)

        if len(calls) == 1:
            raise RuntimeError("first attempt failed")

        return {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "fold": cell.fold,
            "status": "PASS",
        }

    runner._execute = fail_once

    cell = ExperimentCell(
        experiment_id="retry_v1",
        dataset="GBSG2",
        prior="Normal",
        fold=0,
        seed=42,
    )

    try:
        runner.run(cell)
    except RuntimeError as exc:
        assert str(exc) == "first attempt failed"
    else:
        raise AssertionError("Expected first execution to fail")

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "failed"

    result = runner.run(cell)

    assert result["status"] == "PASS"

    checkpoint = manager.load_checkpoint(cell)

    assert checkpoint["status"] == "complete"
    assert len(calls) == 2
