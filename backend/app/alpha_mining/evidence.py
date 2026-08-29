"""Content-addressed immutable Alpha experiments and hash-chained state evidence."""
# Requirements: AM-S4-006 through AM-S4-011 and AM-S6-011.
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.alpha_mining.state import AlphaExperimentState, transition_alpha_state

_LOCK = threading.RLock()


class AlphaEvidenceError(RuntimeError):
    pass


class AlphaEvidenceStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir).resolve() / "alpha_mining" / "evidence"
        self.experiments_root = self.root / "experiments"
        self.candidates_root = self.root / "candidates"
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        self.candidates_root.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, run_id: str, contract: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": "alpha-experiment-v1",
            "run_id": run_id,
            "contract": _canonical(contract),
        }
        payload["content_sha256"] = _digest(payload)
        path = self._experiment_path(run_id)
        with _LOCK:
            _write_once_or_verify(path, payload)
        return payload

    def record_experiment_result(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        experiment = self.read_experiment(run_id)
        payload = {
            "schema_version": "alpha-experiment-result-v1",
            "run_id": run_id,
            "experiment_sha256": experiment["content_sha256"],
            "result": _canonical(result),
        }
        payload["content_sha256"] = _digest(payload)
        path = self._experiment_result_path(run_id)
        with _LOCK:
            _write_once_or_verify(path, payload)
        return payload

    def freeze_candidate(
        self,
        *,
        run_id: str,
        engine_id: str,
        candidate: dict[str, Any],
        renderer: dict[str, Any],
    ) -> dict[str, Any]:
        contract = self.read_experiment(run_id)
        identity = {
            "run_id": run_id,
            "engine_id": engine_id,
            "candidate": _canonical(candidate),
            "experiment_sha256": contract["content_sha256"],
        }
        candidate_id = f"ac-{_digest(identity)[:32]}"
        payload = {
            "schema_version": "alpha-candidate-v1",
            "candidate_id": candidate_id,
            **identity,
            "renderer": _canonical(renderer),
        }
        payload["content_sha256"] = _digest(payload)
        directory = self._candidate_dir(candidate_id)
        with _LOCK:
            directory.mkdir(parents=False, exist_ok=True)
            _write_once_or_verify(directory / "candidate.json", payload)
            if not (directory / "state.json").exists():
                _atomic_json(directory / "state.json", {
                    "candidate_id": candidate_id,
                    "state": AlphaExperimentState.DRAFT.value,
                    "revision": 0,
                    "updated_at": _now(),
                })
                _atomic_text(directory / "events.jsonl", "")
                for state in (
                    AlphaExperimentState.REGISTERED,
                    AlphaExperimentState.DATA_READY,
                    AlphaExperimentState.DISCOVERY,
                    AlphaExperimentState.FROZEN,
                ):
                    self.transition(candidate_id, state, {"run_id": run_id})
        return self.get_candidate(candidate_id)

    def record_outer_evaluation(
        self,
        candidate_id: str,
        evaluation: dict[str, Any],
        target: str,
    ) -> dict[str, Any]:
        directory = self._candidate_dir(candidate_id)
        payload = {
            "schema_version": "alpha-outer-evidence-v1",
            "candidate_id": candidate_id,
            "evaluation": _canonical(evaluation),
        }
        payload["content_sha256"] = _digest(payload)
        with _LOCK:
            _write_once_or_verify(directory / "outer_evaluation.json", payload)
            current = self.get_state(candidate_id)["state"]
            if current == AlphaExperimentState.FROZEN.value:
                self.transition(
                    candidate_id,
                    AlphaExperimentState.OUTER_EVALUATED,
                    {"evidence_sha256": payload["content_sha256"]},
                )
            current = self.get_state(candidate_id)["state"]
            if current == AlphaExperimentState.OUTER_EVALUATED.value:
                self.transition(candidate_id, target, {"gate_verdict": target})
        return self.get_candidate(candidate_id)

    def transition(
        self,
        candidate_id: str,
        target: AlphaExperimentState | str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with _LOCK:
            state = self.get_state(candidate_id)
            destination = transition_alpha_state(state["state"], target)
            events = self.events(candidate_id)
            previous_hash = events[-1]["event_sha256"] if events else None
            event = {
                "revision": int(state["revision"]) + 1,
                "candidate_id": candidate_id,
                "from": state["state"],
                "to": destination.value,
                "timestamp": _now(),
                "previous_sha256": previous_hash,
                "evidence": _canonical(evidence),
            }
            event["event_sha256"] = _digest(event)
            _append_fsync(self._candidate_dir(candidate_id) / "events.jsonl", event)
            next_state = {
                "candidate_id": candidate_id,
                "state": destination.value,
                "revision": event["revision"],
                "updated_at": event["timestamp"],
                "last_event_sha256": event["event_sha256"],
            }
            _atomic_json(self._candidate_dir(candidate_id) / "state.json", next_state)
            return next_state

    def read_experiment(self, run_id: str) -> dict[str, Any]:
        payload = _read_object(self._experiment_path(run_id))
        result = self._experiment_result_path(run_id)
        payload["result"] = _read_object(result) if result.is_file() else None
        return payload

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        output = []
        for path in sorted(self.experiments_root.glob("alpha-*.json"), reverse=True):
            if path.name.endswith(".result.json"):
                continue
            try:
                output.append(self.read_experiment(path.stem))
            except (OSError, AlphaEvidenceError):
                continue
            if len(output) >= limit:
                break
        return output

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        payload = _read_object(self._candidate_dir(candidate_id) / "candidate.json")
        payload["state"] = self.get_state(candidate_id)
        outer = self._candidate_dir(candidate_id) / "outer_evaluation.json"
        payload["outer_evaluation"] = _read_object(outer) if outer.is_file() else None
        shadow = self._candidate_dir(candidate_id) / "shadow.json"
        payload["shadow"] = _read_object(shadow) if shadow.is_file() else None
        forward = self._candidate_dir(candidate_id) / "forward_evaluation.json"
        payload["forward_evaluation"] = _read_object(forward) if forward.is_file() else None
        publication = self._candidate_dir(candidate_id) / "publication.json"
        payload["publication"] = _read_object(publication) if publication.is_file() else None
        return payload

    def get_state(self, candidate_id: str) -> dict[str, Any]:
        return _read_object(self._candidate_dir(candidate_id) / "state.json")

    def list_candidates(self) -> list[dict[str, Any]]:
        output = []
        for path in sorted(self.candidates_root.glob("ac-*")):
            try:
                output.append(self.get_candidate(path.name))
            except (OSError, AlphaEvidenceError):
                continue
        return sorted(
            output,
            key=lambda item: str(item.get("state", {}).get("updated_at") or ""),
            reverse=True,
        )

    def events(self, candidate_id: str) -> list[dict[str, Any]]:
        path = self._candidate_dir(candidate_id) / "events.jsonl"
        if not path.is_file():
            return []
        output = []
        previous = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            expected = item.pop("event_sha256", None)
            actual = _digest(item)
            item["event_sha256"] = expected
            if expected != actual or item.get("previous_sha256") != previous:
                raise AlphaEvidenceError("Alpha候选事件哈希链损坏")
            previous = expected
            output.append(item)
        return output

    def write_shadow_receipt(self, candidate_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": "alpha-shadow-v1",
            "candidate_id": candidate_id,
            "receipt": _canonical(receipt),
        }
        payload["content_sha256"] = _digest(payload)
        with _LOCK:
            _write_once_or_verify(self._candidate_dir(candidate_id) / "shadow.json", payload)
        return payload

    def write_forward_evaluation(
        self,
        candidate_id: str,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "alpha-forward-evidence-v1",
            "candidate_id": candidate_id,
            "evaluation": _canonical(evaluation),
        }
        payload["content_sha256"] = _digest(payload)
        with _LOCK:
            _write_once_or_verify(
                self._candidate_dir(candidate_id) / "forward_evaluation.json",
                payload,
            )
        return payload

    def write_publication_receipt(
        self,
        candidate_id: str,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "alpha-publication-v1",
            "candidate_id": candidate_id,
            "receipt": _canonical(receipt),
        }
        payload["content_sha256"] = _digest(payload)
        with _LOCK:
            _write_once_or_verify(
                self._candidate_dir(candidate_id) / "publication.json",
                payload,
            )
        return payload

    def _experiment_path(self, run_id: str) -> Path:
        if not run_id.startswith("alpha-") or "/" in run_id or ".." in run_id:
            raise AlphaEvidenceError("无效Alpha运行ID")
        return self.experiments_root / f"{run_id}.json"

    def _experiment_result_path(self, run_id: str) -> Path:
        self._experiment_path(run_id)
        return self.experiments_root / f"{run_id}.result.json"

    def _candidate_dir(self, candidate_id: str) -> Path:
        if not candidate_id.startswith("ac-") or "/" in candidate_id or ".." in candidate_id:
            raise AlphaEvidenceError("无效Alpha候选ID")
        return self.candidates_root / candidate_id


def _canonical(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
    return json.loads(encoded)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeyError(path.stem) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AlphaEvidenceError(f"Alpha证据不可读: {path.name}") from exc
    if not isinstance(value, dict):
        raise AlphaEvidenceError(f"Alpha证据格式错误: {path.name}")
    return value


def _write_once_or_verify(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _read_object(path) != payload:
            raise AlphaEvidenceError(f"不可变Alpha证据冲突: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if _read_object(path) != payload:
                raise AlphaEvidenceError(f"不可变Alpha证据冲突: {path.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_fsync(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _now() -> str:
    return datetime.now(UTC).isoformat()
