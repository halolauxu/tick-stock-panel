from __future__ import annotations

# Requirements: AM-S1-003 through AM-S1-005 and AM-S4-006 through AM-S4-010.
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from app.alpha_mining.config_store import AlphaConfigStore
from app.alpha_mining.hypotheses import AlphaHypothesisStore
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


def test_alpha_manager_persists_engine_progress_and_final_evidence_phase(tmp_path: Path) -> None:
    AlphaConfigStore(tmp_path).update({"enabled": True})
    progress_written = threading.Event()
    release_worker = threading.Event()

    def runner(_task, progress_cb, _cancel_event):
        progress_cb({
            "phase": "validation",
            "label": "截面排序发现 · 外层窗口 1/2",
            "done": 1,
            "total": 2,
            "percent": 50.0,
            "current_engine_id": "cross_sectional_rank",
            "trials_used": 7,
            "trial_limit": 48,
            "frozen_candidates": 1,
            "candidate_limit": 4,
            "backtests": 1,
            "engine_errors": 0,
            "engines": [{
                "engine_id": "cross_sectional_rank",
                "status": "running",
                "folds_done": 1,
                "folds_total": 2,
                "trials": 7,
                "selected": 1,
                "backtests": 1,
                "errors": 0,
                "message": "正在处理外层窗口 2/2",
            }],
        })
        progress_written.set()
        assert release_worker.wait(2)
        return {
            "status": "succeeded",
            "research_state": "outer_evaluated",
            "candidates": [],
            "champion": {},
            "progress": {"phase": "evidence", "label": "正在冻结证据"},
        }

    manager = AlphaMiningJobManager(tmp_path, worker_runner=runner)
    try:
        run = manager.start({"engine_ids": ["cross_sectional_rank"]}, {"generation": "g1"})
        assert progress_written.wait(2)
        summary = manager.store.read_summary(run["run_id"])
        assert summary["progress"]["engines"][0]["trials"] == 7
        assert summary["progress"]["backtests"] == 1
        release_worker.set()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stored = manager.store.get(run["run_id"])
            if stored and stored["status"] == "succeeded":
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("Alpha run did not finish")
        summary = manager.store.read_summary(run["run_id"])
        assert summary["progress"]["phase"] == "completed"
        assert summary["progress"]["label"] == "研究计算与候选证据冻结完成"
        assert summary["progress"]["percent"] == 100.0
    finally:
        release_worker.set()
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


def test_rejected_hypothesis_creates_durable_next_hypothesis(tmp_path: Path) -> None:
    AlphaConfigStore(tmp_path).update({"enabled": True})
    hypothesis = AlphaHypothesisStore(tmp_path).get("ah-system-selling-exhaustion-v1")

    def runner(*_args):
        return {
            "status": "succeeded",
            "research_state": "outer_evaluated",
            "champion": {},
            "candidates": [],
            "failure_analysis": {
                "zero_pass": True,
                "conclusion": "预注册组合没有通过样本外硬门槛。",
            },
            "next_research_suggestions": [{
                "suggestion_id": "next-oos-1",
                "title": "加入流动性约束后重新检验",
                "why": "原假设在高摩擦股票中衰减。",
                "request_patch": {
                    "factor_names": [
                        *hypothesis["test_spec"]["factor_names"],
                        "amihud_20d",
                    ],
                },
            }],
        }

    manager = AlphaMiningJobManager(tmp_path, worker_runner=runner)
    try:
        run = manager.start({
            "engine_ids": hypothesis["test_spec"]["engine_ids"],
            "factor_names": hypothesis["test_spec"]["factor_names"],
            "asset_type": "stock",
            "budget_profile": "exploratory",
            "forward_horizon": hypothesis["forward_horizon"],
            "hypothesis_id": hypothesis["hypothesis_id"],
            "hypothesis_contract": hypothesis,
        }, {"generation": "g1"})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stored = manager.store.get(run["run_id"])
            if stored and stored["status"] == "succeeded":
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("Alpha hypothesis run did not finish")

        summary = manager.store.read_summary(run["run_id"])
        assert summary["hypothesis"]["verdict"] == "rejected"
        assert len(summary["next_hypotheses"]) == 1
        next_hypothesis = summary["next_hypotheses"][0]
        assert next_hypothesis["source_kind"] == "failure"
        assert next_hypothesis["parent_hypothesis_id"] == hypothesis["hypothesis_id"]
        assert "amihud_20d" in next_hypothesis["test_spec"]["factor_names"]
        persisted = manager.hypotheses.get(hypothesis["hypothesis_id"])
        assert persisted["status"] == "rejected"
        assert persisted["results"][-1]["next_hypothesis_ids"] == [next_hypothesis["hypothesis_id"]]
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    ("profile", "expected_state"),
    [("balanced", "validation_candidate"), ("strict", "research_candidate")],
)
def test_only_full_history_strict_run_can_create_research_candidate(
    tmp_path: Path,
    monkeypatch,
    profile: str,
    expected_state: str,
) -> None:
    monkeypatch.setattr(
        "app.alpha_mining.lifecycle.enriched_partition_dates",
        lambda *_args: [date(2013, 1, 4), date(2026, 1, 1)],
    )
    AlphaConfigStore(tmp_path).update({"enabled": True})

    def runner(*_args):
        return {
            "status": "succeeded",
            "research_state": "research_candidate",
            "champion": {},
            "candidates": [{
                "engine_id": "cross_sectional_rank",
                "engine_name": "截面排序发现",
                "state": "research_candidate",
                "frozen_candidate": {
                    "recipe_id": f"strict-gate-{profile}",
                    "engine_id": "cross_sectional_rank",
                    "engine_version": "1.0.0",
                    "name": "Strict gate",
                    "thesis": "lifecycle test",
                    "signal_kind": "factor_rank",
                    "features": ["momentum_5d"],
                    "directions": [1],
                    "weights": [1.0],
                    "parameters": {"top_rank": 20},
                    "train_evidence": {},
                },
                "metrics": {"stitched_oos_return": 0.1},
                "gates": [],
                "folds": [],
            }],
        }

    manager = AlphaMiningJobManager(tmp_path, worker_runner=runner)
    try:
        run = manager.start({
            "engine_ids": ["cross_sectional_rank"],
            "factor_names": ["momentum_5d"],
            "asset_type": "stock",
            "budget_profile": profile,
            "start": "2013-01-04",
            "end": "2026-01-01",
        }, {"generation": "g1"})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stored = manager.store.get(run["run_id"])
            if stored and stored["status"] == "succeeded":
                break
            threading.Event().wait(0.01)
        summary = manager.store.read_summary(run["run_id"])
        assert summary["candidates"][0]["state"] == expected_state
        assert summary["research_state"] == expected_state
    finally:
        manager.shutdown()
