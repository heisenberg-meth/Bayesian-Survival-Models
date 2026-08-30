import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrequentistCell:
    """Unique description of one frequentist experiment cell."""

    experiment_id: str
    dataset: str
    model_type: str
    fold: int
    seed: int
    model_params: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def cell_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def short_id(self) -> str:
        return self.cell_id[:16]
