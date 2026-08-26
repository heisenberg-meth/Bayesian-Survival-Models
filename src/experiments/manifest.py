"""
Deterministic experiment manifest and cell identity.

Every robustness experiment must be represented by an immutable,
serialisable cell specification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentCell:
    """
    Unique description of one experiment cell.

    The combination of all fields defines the computational experiment.
    """

    experiment_id: str
    dataset: str
    prior: str
    fold: int
    seed: int
    method: str = "mcmc"
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    n_intervals: int = 6
    coefficient_prior: str = "normal"
    beta_prior_mean: float = 0.0
    beta_prior_sd: float = 10.0
    baseline_hazard_prior: str = "gamma"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def canonical_json(self) -> str:
        """
        Return a deterministic JSON representation.

        Sorting keys is critical: the same logical cell must always
        produce the same identity regardless of dictionary ordering.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def cell_id(self) -> str:
        """
        Return a deterministic SHA-256 identity for this experiment cell.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def short_id(self) -> str:
        """Human-readable shortened cell identifier."""
        return self.cell_id[:16]


def build_manifest(cells: list[ExperimentCell]) -> dict[str, Any]:
    """
    Build a complete experiment manifest.

    The manifest itself is deterministic and contains the identity of
    every experiment cell.
    """
    serialized = [cell.to_dict() | {"cell_id": cell.cell_id} for cell in cells]

    manifest_payload = {
        "cells": serialized,
        "n_cells": len(serialized),
    }

    manifest_json = json.dumps(
        manifest_payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    return {
        **manifest_payload,
        "manifest_hash": manifest_hash,
    }
