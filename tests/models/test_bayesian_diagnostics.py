import arviz as az
import numpy as np
import pytest

from src.models.bayesian.diagnostics import MCMCDiagnostics


def make_trace(
    chains: int = 4,
    draws: int = 500,
    divergences: int = 0,
):
    rng = np.random.default_rng(42)

    posterior = {
        "beta": rng.normal(
            size=(chains, draws, 2),
        ),
    }

    diverging = np.zeros(
        (chains, draws),
        dtype=bool,
    )

    if divergences:
        diverging.flat[:divergences] = True

    sample_stats = {
        "diverging": diverging,
        "tree_depth": np.full(
            (chains, draws),
            5,
            dtype=int,
        ),
        "energy": rng.normal(
            size=(chains, draws),
        ),
    }

    return az.from_dict(
        posterior=posterior,
        sample_stats=sample_stats,
    )


def test_rhat_is_computed():
    trace = make_trace()

    result = MCMCDiagnostics.compute_rhat(trace)

    assert "rhat_max" in result
    assert np.isfinite(result["rhat_max"])
    assert result["rhat_max"] != 1.01


def test_ess_is_computed():
    trace = make_trace()

    result = MCMCDiagnostics.compute_ess(trace)

    assert np.isfinite(result["ess_bulk_min"])
    assert np.isfinite(result["ess_tail_min"])
    assert result["ess_bulk_min"] > 0
    assert result["ess_tail_min"] > 0


def test_divergences_are_computed():
    trace = make_trace(divergences=3)

    result = MCMCDiagnostics.compute_divergences(trace)

    assert result["divergences"] == 3


def test_bfmi_is_computed():
    trace = make_trace()

    result = MCMCDiagnostics.compute_bfmi(trace)

    assert "bfmi_min" in result
    assert np.isfinite(result["bfmi_min"])


def test_tree_depth_is_computed():
    trace = make_trace()

    result = MCMCDiagnostics.compute_tree_depth(trace)

    assert result["tree_depth_max"] == 5


def test_diagnostic_gate_passes_valid_trace():
    trace = make_trace()

    result = MCMCDiagnostics.compute_all(trace)

    assert result["status"] == "PASS"
    assert all(result["gates"].values())


def test_diagnostic_gate_fails_divergence():
    trace = make_trace(divergences=1)

    result = MCMCDiagnostics.compute_all(trace)

    assert result["status"] == "FAIL"
    assert result["gates"]["divergences"] is False


def test_diagnostic_gate_fails_tree_depth():
    trace = make_trace()

    trace.sample_stats["tree_depth"].values[:] = 10

    result = MCMCDiagnostics.compute_all(trace, max_tree_depth=10)

    assert result["status"] == "FAIL"
    assert result["gates"]["tree_depth"] is False


def test_diagnostic_gate_does_not_use_hardcoded_values():
    trace_a = make_trace(chains=4, draws=200)
    trace_b = make_trace(chains=4, draws=1000)

    ess_a = MCMCDiagnostics.compute_ess(trace_a)
    ess_b = MCMCDiagnostics.compute_ess(trace_b)

    assert ess_a["ess_bulk_min"] != ess_b["ess_bulk_min"]


def test_invalid_trace_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        MCMCDiagnostics.compute_rhat(None)
