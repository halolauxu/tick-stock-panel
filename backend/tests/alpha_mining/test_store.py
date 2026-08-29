from __future__ import annotations

from pathlib import Path

from app.alpha_mining.store import AlphaRunStore


def test_alpha_store_is_physically_separate_from_legacy_mining(tmp_path: Path) -> None:
    store = AlphaRunStore(tmp_path)
    run = store.create({"engine_ids": ["cross_sectional"]}, {"generation": "g1"})

    assert store.runs_root == (tmp_path / "alpha_mining" / "runs").resolve()
    assert (store.runs_root / run["run_id"] / "manifest.json").is_file()
    assert not (tmp_path / "research" / "mining" / "runs").exists()
