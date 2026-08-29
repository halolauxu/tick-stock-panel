"""Independent job manager for the Alpha mining subsystem."""
# ruff: noqa: RUF001
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from app.alpha_mining.config_store import AlphaConfigStore
from app.alpha_mining.contracts import CandidateSpec, FrozenSignalSpec
from app.alpha_mining.evidence import AlphaEvidenceStore
from app.alpha_mining.hypotheses import AlphaHypothesisStore
from app.alpha_mining.lifecycle import is_strict_full_history_request
from app.alpha_mining.policy import ALPHA_ALGORITHM_VERSION
from app.alpha_mining.providers import DeclarativeCandidateRenderer
from app.alpha_mining.registry import load_builtin_registry
from app.alpha_mining.store import AlphaRunStore
from app.backtest.worker import make_worker_task, run_worker_task
from app.services.mining_jobs import (
    ACTIVE_RUN_STATUSES,
    SUCCESS_RUN_STATUSES,
    compute_run_signature,
)
from app.services.mining_manager import MiningJobManager, TaskFactory, WorkerRunner


class AlphaMiningJobManager(MiningJobManager):
    def __init__(
        self,
        data_dir: Path | str,
        worker_runner: WorkerRunner = run_worker_task,
        task_factory: TaskFactory = make_worker_task,
    ) -> None:
        super().__init__(
            data_dir,
            worker_runner=worker_runner,
            task_factory=task_factory,
            store_factory=AlphaRunStore,
            task_kind="alpha_mining",
            thread_name_prefix="alpha-mining",
        )
        self.evidence = AlphaEvidenceStore(data_dir)
        self.hypotheses = AlphaHypothesisStore(data_dir)
        self.renderer = DeclarativeCandidateRenderer()

    def start(
        self,
        request: dict[str, Any],
        data_fingerprint: Any,
        force: bool = False,
        source: str = "manual",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        settings = AlphaConfigStore(self._data_dir).get()
        if not settings["enabled"]:
            raise ValueError("Alpha挖掘功能开关已关闭")
        if source != "manual" and not settings["auto_run_enabled"]:
            raise ValueError("Alpha自动研究权限未开启")
        resolved_run_id = run_id or f"alpha-{uuid.uuid4().hex[:24]}"
        with self._lock:
            if not force:
                signature = compute_run_signature(request, data_fingerprint)
                existing = self.store.find_by_signature(
                    signature,
                    statuses=ACTIVE_RUN_STATUSES | SUCCESS_RUN_STATUSES,
                )
                if existing is not None:
                    return existing
            registry, failures = load_builtin_registry()
            self.evidence.create_experiment(resolved_run_id, {
                "contract_version": "alpha-experiment-v1",
                "algorithm_version": ALPHA_ALGORITHM_VERSION,
                "request": request,
                "data_fingerprint": data_fingerprint,
                "code_fingerprint": _code_fingerprint(),
                "engine_manifests": [engine.manifest.to_dict() for engine in registry.list()],
                "engine_load_failures": failures,
                "budget": {
                    "max_candidates_per_engine": request.get("max_candidates_per_engine"),
                    "max_trials_per_engine": request.get("max_trials_per_engine"),
                },
                "labels": {
                    "horizons": [1, 3, 5, 10, 20, 60],
                    "selected_horizon": request.get("forward_horizon"),
                    "net_of_costs": True,
                },
                "execution": {
                    "entry": "open_t_plus_one",
                    "exit": "open_t_plus_one",
                    "commission_pct": request.get("commission_pct"),
                    "stamp_tax_pct": request.get("stamp_tax_pct"),
                    "slippage_bps": request.get("slippage_bps"),
                },
            })
            manifest = super().start(
                request,
                data_fingerprint,
                force=force,
                source=source,
                run_id=resolved_run_id,
            )
            hypothesis_id = request.get("hypothesis_id")
            if hypothesis_id:
                self.hypotheses.attach_run(str(hypothesis_id), str(manifest["run_id"]))
            return manifest

    def _finish_success(self, run_id, result, cancel_event) -> None:
        if cancel_event.is_set():
            super()._finish_success(run_id, result, cancel_event)
            return
        champion = dict(result.get("champion") or {})
        experiment = self.evidence.read_experiment(run_id)
        experiment_request = dict(experiment.get("contract", {}).get("request") or {})
        strict_full_history = is_strict_full_history_request(self._data_dir, experiment_request)
        for row in result.get("candidates") or []:
            frozen_row = row.get("frozen_candidate")
            if not isinstance(frozen_row, dict):
                row["state"] = "rejected"
                row["evidence_reason"] = "no_frozen_candidate"
                continue
            candidate = CandidateSpec(
                recipe_id=str(frozen_row["recipe_id"]),
                engine_id=str(frozen_row["engine_id"]),
                engine_version=str(frozen_row["engine_version"]),
                name=str(frozen_row["name"]),
                thesis=str(frozen_row["thesis"]),
                signal_kind=str(frozen_row["signal_kind"]),
                features=tuple(str(value) for value in frozen_row["features"]),
                directions=tuple(int(value) for value in frozen_row["directions"]),
                weights=tuple(float(value) for value in frozen_row["weights"]),
                parameters=dict(frozen_row["parameters"]),
                train_evidence=dict(frozen_row["train_evidence"]),
            )
            frozen = FrozenSignalSpec.from_candidate(candidate)
            rendered = self.renderer.render(frozen)
            candidate_evidence = frozen.to_dict()
            candidate_evidence["research"] = candidate.to_dict()
            evidence = self.evidence.freeze_candidate(
                run_id=run_id,
                engine_id=candidate.engine_id,
                candidate=candidate_evidence,
                renderer=dict(rendered),
            )
            if row.get("state") == "research_candidate":
                target = "research_candidate" if strict_full_history else "validation_candidate"
            else:
                target = "rejected"
            outer_evaluation = {
                "metrics": row.get("metrics"),
                "gates": row.get("gates"),
                "folds": row.get("folds"),
            }
            if champion.get("strategy_id"):
                outer_evaluation["champion"] = champion
            evidence = self.evidence.record_outer_evaluation(
                evidence["candidate_id"],
                outer_evaluation,
                target,
            )
            row["candidate_id"] = evidence["candidate_id"]
            row["state"] = evidence["state"]["state"]
        candidate_states = {str(row.get("state") or "") for row in result.get("candidates") or []}
        if "research_candidate" in candidate_states:
            result["research_state"] = "research_candidate"
        elif "validation_candidate" in candidate_states:
            result["research_state"] = "validation_candidate"
        progress = dict(result.get("progress") or {})
        progress.update({
            "phase": "completed",
            "label": "研究计算与候选证据冻结完成",
            "done": 1,
            "total": 1,
            "percent": 100.0,
            "current_engine_id": None,
        })
        result["progress"] = progress
        hypothesis_id = experiment_request.get("hypothesis_id")
        if hypothesis_id:
            hypothesis = self.hypotheses.get(str(hypothesis_id))
            next_hypotheses = []
            if not ({"research_candidate", "validation_candidate"} & candidate_states):
                for suggestion in result.get("next_research_suggestions") or []:
                    if not isinstance(suggestion, dict):
                        continue
                    next_hypotheses.append(self.hypotheses.create_from_failure(
                        hypothesis,
                        run_id,
                        suggestion,
                        experiment_request,
                    ))
            result["hypothesis"] = {
                "hypothesis_id": hypothesis_id,
                "title": hypothesis["title"],
                "source_kind": hypothesis["source_kind"],
                "verdict": (
                    "supported"
                    if {"research_candidate", "validation_candidate"} & candidate_states
                    else "rejected"
                ),
            }
            result["next_hypotheses"] = next_hypotheses
            self.hypotheses.record_result(
                str(hypothesis_id),
                run_id,
                verdict=(
                    "supported"
                    if {"research_candidate", "validation_candidate"} & candidate_states
                    else "rejected"
                ),
                conclusion=str(
                    (result.get("failure_analysis") or {}).get("conclusion")
                    or "本轮假设产生了通过历史门槛的候选"
                ),
                next_hypothesis_ids=[str(row["hypothesis_id"]) for row in next_hypotheses],
            )
        self.evidence.record_experiment_result(run_id, result)
        super()._finish_success(run_id, result, cancel_event)

    def _finish_failed(self, run_id: str, exc: Exception) -> None:
        experiment = self.evidence.read_experiment(run_id)
        request = dict(experiment.get("contract", {}).get("request") or {})
        if request.get("hypothesis_id"):
            self.hypotheses.record_result(
                str(request["hypothesis_id"]),
                run_id,
                verdict="blocked",
                conclusion=f"执行链路失败，尚不能判断假设真伪：{str(exc)[:500]}",
            )
        self.evidence.record_experiment_result(run_id, {
            "status": "failed",
            "error": str(exc)[:2000],
            "trial_ledger": [],
        })
        super()._finish_failed(run_id, exc)

    def _finish_cancelled_locked(self, run_id: str) -> None:
        experiment = self.evidence.read_experiment(run_id)
        request = dict(experiment.get("contract", {}).get("request") or {})
        if request.get("hypothesis_id"):
            self.hypotheses.record_result(
                str(request["hypothesis_id"]),
                run_id,
                verdict="cancelled",
                conclusion="研究被人工取消，假设未被支持或证伪。",
            )
        self.evidence.record_experiment_result(run_id, {
            "status": "cancelled",
            "trial_ledger": [],
        })
        super()._finish_cancelled_locked(run_id)


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "alpha_mining").rglob("*.py"))
    paths.extend([
        root / "backtest" / "mining_runtime.py",
        root / "strategy" / "builtin" / "factor_rank_research.py",
    ])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
