from __future__ import annotations

# Requirements: AM-S1-003 through AM-S1-005 and AM-S4-006 through AM-S4-010.
import threading
import time
from pathlib import Path

import pytest

from app.alpha_mining.config_store import AlphaConfigStore
from app.services.alpha_mining_manager import AlphaMiningJobManager


def test_alpha_manager_uses_alpha_task_kind_and_prefixed_run_id(tmp_path: Path) -> None:
    observed = []

    def task_factory(kind, data_dir, payload):
        return {"kind": kind, "data_dir": str(data_dir), "payload": payload}

    def runner(task, progress_cb, cancel_event):
        del progress_cb, cancel_event
        observed.append(task)
        return {"status": "succeeded", "research_state": "outer_evaluated"}

    AlphaConfigStore(tmp_path).update({"enabled": True})
    manager = AlphaMiningJobManager(tmp_path, worker_runner=runner, task_factory=task_factory)
    try:
        run = manager.start({"engine_ids": ["cross_sectional_rank"]}, {"generation": "g1"})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stored = manager.store.get(run["run_id"])
            if stored and stored["status"] == "succeeded":
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("Alpha run did not complete")

        assert run["run_id"].startswith("alpha-")
        assert observed[0]["kind"] == "alpha_mining"
        assert manager.store.runs_root == (tmp_path / "alpha_mining" / "runs").resolve()
        evidence = manager.evidence.read_experiment(run["run_id"])
        assert evidence["contract"]["engine_manifests"]
        assert evidence["result"]["result"]["status"] == "succeeded"
    finally:
        manager.shutdown()


def test_alpha_manager_enforces_manual_and_automatic_permissions(tmp_path: Path) -> None:
    manager = AlphaMiningJobManager(tmp_path)
    try:
        with pytest.raises(ValueError, match="功能开关"):
            manager.start({}, {})
        AlphaConfigStore(tmp_path).update({"enabled": True})
        with pytest.raises(ValueError, match="自动研究权限"):
            manager.start({}, {}, source="scheduled")
    finally:
        manager.shutdown()


def test_failed_alpha_run_keeps_immutable_contract_and_failure_result(tmp_path: Path) -> None:
    AlphaConfigStore(tmp_path).update({"enabled": True})

    def fail(*_args):
        raise RuntimeError("synthetic worker failure")

    manager = AlphaMiningJobManager(tmp_path, worker_runner=fail)
    try:
        run = manager.start({"engine_ids": ["cross_sectional_rank"]}, {"generation": "g1"})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stored = manager.store.get(run["run_id"])
            if stored and stored["status"] == "failed":
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("Alpha failure was not persisted")
        evidence = manager.evidence.read_experiment(run["run_id"])
        assert evidence["result"]["result"]["status"] == "failed"
        assert evidence["result"]["result"]["error"] == "synthetic worker failure"
    finally:
        manager.shutdown()
