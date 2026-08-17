"""
MCMC convergence diagnostics computed from an ArviZ InferenceData trace.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np


class MCMCDiagnostics:
    """Computes convergence and sampling-quality diagnostics."""

    @staticmethod
    def _validate_trace(trace: Any) -> None:
        """Validate that the supplied object contains posterior samples."""
        if trace is None:
            raise ValueError("Inference trace cannot be None.")

        if not hasattr(trace, "posterior"):
            raise TypeError("Trace must provide an ArviZ posterior group.")

    @staticmethod
    def compute_rhat(trace: Any) -> dict[str, float]:
        """
        Compute maximum rank-normalised R-hat across posterior parameters.
        """
        MCMCDiagnostics._validate_trace(trace)

        summary = az.rhat(trace, method="rank")

        values = np.asarray(summary.to_array().values, dtype=float)

        if values.size == 0:
            raise ValueError("No posterior parameters available for R-hat.")

        if not np.all(np.isfinite(values)):
            raise ValueError("R-hat contains non-finite values.")

        return {
            "rhat_max": float(np.max(values)),
        }

    @staticmethod
    def compute_ess(trace: Any) -> dict[str, float]:
        """
        Compute minimum bulk and tail effective sample sizes.
        """
        MCMCDiagnostics._validate_trace(trace)

        ess_bulk = az.ess(trace, method="bulk")
        ess_tail = az.ess(trace, method="tail")

        bulk_values = np.asarray(
            ess_bulk.to_array().values,
            dtype=float,
        )
        tail_values = np.asarray(
            ess_tail.to_array().values,
            dtype=float,
        )

        if bulk_values.size == 0:
            raise ValueError("No posterior parameters available for ESS.")

        if not np.all(np.isfinite(bulk_values)):
            raise ValueError("Bulk ESS contains non-finite values.")

        if not np.all(np.isfinite(tail_values)):
            raise ValueError("Tail ESS contains non-finite values.")

        return {
            "ess_bulk_min": float(np.min(bulk_values)),
            "ess_tail_min": float(np.min(tail_values)),
        }

    @staticmethod
    def compute_divergences(trace: Any) -> dict[str, int]:
        """
        Count divergent NUTS transitions from sampler statistics.
        """
        MCMCDiagnostics._validate_trace(trace)

        if "sample_stats" not in trace:
            raise ValueError("Trace does not contain sampler statistics.")

        sample_stats = trace.sample_stats

        if "diverging" not in sample_stats:
            raise ValueError("Sampler statistics do not contain 'diverging'.")

        divergent = np.asarray(
            sample_stats["diverging"].values,
            dtype=bool,
        )

        return {
            "divergences": int(np.sum(divergent)),
        }

    @staticmethod
    def compute_bfmi(trace: Any) -> dict[str, float]:
        """
        Compute the minimum Bayesian fraction of missing information.
        """
        MCMCDiagnostics._validate_trace(trace)

        if "sample_stats" not in trace:
            raise ValueError("Trace does not contain sampler statistics.")

        values = np.asarray(
            az.bfmi(trace),
            dtype=float,
        )

        if values.size == 0:
            raise ValueError("BFMI could not be computed.")

        if not np.all(np.isfinite(values)):
            raise ValueError("BFMI contains non-finite values.")

        return {
            "bfmi_min": float(np.min(values)),
        }

    @staticmethod
    def compute_tree_depth(trace: Any) -> dict[str, int]:
        """
        Report the maximum observed NUTS tree depth.
        """
        MCMCDiagnostics._validate_trace(trace)

        if "sample_stats" not in trace:
            raise ValueError("Trace does not contain sampler statistics.")

        sample_stats = trace.sample_stats

        if "tree_depth" not in sample_stats:
            raise ValueError("Sampler statistics do not contain 'tree_depth'.")

        tree_depth = np.asarray(
            sample_stats["tree_depth"].values,
            dtype=int,
        )

        if tree_depth.size == 0:
            raise ValueError("No tree-depth samples available.")

        return {
            "tree_depth_max": int(np.max(tree_depth)),
        }

    @staticmethod
    def compute_all(
        trace: Any,
        max_tree_depth: int = 10,
    ) -> dict[str, Any]:
        """
        Compute all mandatory Bayesian sampling diagnostics.

        The returned status is PASS only when all mandatory quality gates pass.
        """
        rhat = MCMCDiagnostics.compute_rhat(trace)
        ess = MCMCDiagnostics.compute_ess(trace)
        divergences = MCMCDiagnostics.compute_divergences(trace)
        bfmi = MCMCDiagnostics.compute_bfmi(trace)
        tree_depth = MCMCDiagnostics.compute_tree_depth(trace)

        rhat_pass = rhat["rhat_max"] < 1.01
        ess_pass = ess["ess_bulk_min"] > 400
        divergences_pass = divergences["divergences"] == 0
        bfmi_pass = bfmi["bfmi_min"] >= 0.3
        tree_depth_pass = tree_depth["tree_depth_max"] < max_tree_depth

        passed = (
            rhat_pass
            and ess_pass
            and divergences_pass
            and bfmi_pass
            and tree_depth_pass
        )

        return {
            **rhat,
            **ess,
            **divergences,
            **bfmi,
            **tree_depth,
            "gates": {
                "rhat": rhat_pass,
                "ess": ess_pass,
                "divergences": divergences_pass,
                "bfmi": bfmi_pass,
                "tree_depth": tree_depth_pass,
            },
            "status": "PASS" if passed else "FAIL",
        }
