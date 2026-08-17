from src.experiments.manifest import ExperimentCell, build_manifest


def make_cell(seed: int = 42) -> ExperimentCell:
    return ExperimentCell(
        experiment_id="robustness_v1",
        dataset="GBSG2",
        prior="Regularised Horseshoe",
        fold=0,
        seed=seed,
        method="mcmc",
        draws=1000,
        tune=1000,
        chains=4,
    )


def test_cell_identity_is_deterministic():
    cell_a = make_cell()
    cell_b = make_cell()

    assert cell_a.cell_id == cell_b.cell_id
    assert cell_a.canonical_json() == cell_b.canonical_json()


def test_changing_seed_changes_identity():
    cell_a = make_cell(seed=42)
    cell_b = make_cell(seed=43)

    assert cell_a.cell_id != cell_b.cell_id


def test_changing_fold_changes_identity():
    cell_a = make_cell()

    cell_b = ExperimentCell(
        experiment_id=cell_a.experiment_id,
        dataset=cell_a.dataset,
        prior=cell_a.prior,
        fold=1,
        seed=cell_a.seed,
        method=cell_a.method,
        draws=cell_a.draws,
        tune=cell_a.tune,
        chains=cell_a.chains,
    )

    assert cell_a.cell_id != cell_b.cell_id


def test_manifest_is_deterministic():
    cells_a = [make_cell(42), make_cell(43)]
    cells_b = [make_cell(42), make_cell(43)]

    manifest_a = build_manifest(cells_a)
    manifest_b = build_manifest(cells_b)

    assert manifest_a["manifest_hash"] == manifest_b["manifest_hash"]
    assert manifest_a["n_cells"] == 2


def test_manifest_contains_cell_ids():
    manifest = build_manifest([make_cell()])

    assert manifest["n_cells"] == 1
    assert len(manifest["cells"]) == 1
    assert manifest["cells"][0]["cell_id"] == make_cell().cell_id


def test_cell_is_immutable():
    cell = make_cell()

    try:
        cell.seed = 999
    except AttributeError:
        pass
    else:
        raise AssertionError("ExperimentCell must be immutable")
