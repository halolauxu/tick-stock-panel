"""Dynamic Alpha champion ledger with explicit, non-overwriting promotion."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.alpha_mining.evidence import AlphaEvidenceStore
from app.alpha_mining.policy import HARD_GATES

_LOCK = threading.RLock()


class AlphaChampionStore:
    def __init__(self, data_dir: Path | str, evidence: AlphaEvidenceStore | None = None) -> None:
        self.path = Path(data_dir).resolve() / "alpha_mining" / "champion.json"
        self.evidence = evidence or AlphaEvidenceStore(data_dir)

    def get(self) -> dict[str, Any]:
        with _LOCK:
            if self.path.is_file():
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            return {
                "schema_version": "alpha-champion-v1",
                "current": {
                    "kind": "none",
                    "strategy_id": None,
                    "candidate_id": None,
                    "effective_at": None,
                    "reason": "尚无通过Alpha系统全部门槛的正式冠军",
                },
                "history": [],
            }

    def leaderboard(self) -> dict[str, Any]:
        candidates = []
        for item in self.evidence.list_candidates():
            evaluation = item.get("outer_evaluation") or {}
            evidence = evaluation.get("evaluation") or {}
            metrics = evidence.get("metrics") or {}
            gates = evidence.get("gates") or []
            candidates.append({
                "candidate_id": item["candidate_id"],
                "engine_id": item["engine_id"],
                "state": item["state"]["state"],
                "return": metrics.get("stitched_oos_return"),
                "sharpe": metrics.get("stitched_oos_sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "gates_passed": sum(gate.get("status") == "passed" for gate in gates),
                "gates_failed": sum(gate.get("status") == "failed" for gate in gates),
                "gates_pending": sum(gate.get("status") == "pending" for gate in gates),
            })
        candidates.sort(key=lambda item: _return_key(item.get("return")), reverse=True)
        return {"champion": self.get()["current"], "challengers": candidates}

    def promote(self, candidate_id: str, strategy_id: str) -> dict[str, Any]:
        with _LOCK:
            candidate = self.evidence.get_candidate(candidate_id)
            if candidate["state"]["state"] != "challenger":
                raise ValueError("只有前向通过的挑战者可以晋级冠军")
            evaluation = (candidate.get("outer_evaluation") or {}).get("evaluation") or {}
            gates = evaluation.get("gates") or []
            historical = [gate for gate in gates if gate.get("id") != "forward_shadow"]
            required_ids = {
                gate["id"] for gate in HARD_GATES if gate["id"] != "forward_shadow"
            }
            forward = (candidate.get("forward_evaluation") or {}).get("evaluation") or {}
            if (
                {gate.get("id") for gate in historical} != required_ids
                or any(gate.get("status") != "passed" for gate in historical)
                or forward.get("verdict") != "passed"
            ):
                raise ValueError("候选仍有未通过或待验证门槛")
            current = self.get()
            history = list(current.get("history") or [])
            history.append(dict(current["current"]))
            next_value = {
                "schema_version": "alpha-champion-v1",
                "current": {
                    "kind": "alpha_candidate",
                    "strategy_id": strategy_id,
                    "candidate_id": candidate_id,
                    "effective_at": datetime.now(UTC).isoformat(),
                    "reason": "all historical, stress and forward gates passed",
                },
                "history": history,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.path, next_value)
            self.evidence.transition(candidate_id, "champion", {"strategy_id": strategy_id})
            return next_value


def _return_key(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
