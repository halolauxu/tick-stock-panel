"""Storage isolated from the legacy mining feature."""
from __future__ import annotations

from pathlib import Path

from app.services.mining_jobs import MiningRunStore


class AlphaRunStore(MiningRunStore):
    """Reuse battle-tested atomic manifests under an independent root."""

    def __init__(self, data_dir: Path | str) -> None:
        self.runs_root = (Path(data_dir).resolve() / "alpha_mining" / "runs").resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
