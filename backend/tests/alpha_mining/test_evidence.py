from __future__ import annotations

# Requirements: AM-S4-006 through AM-S4-011 and AM-S6-011.
import json

import pytest

from app.alpha_mining.champion import AlphaChampionStore
from app.alpha_mining.contracts import CandidateSpec, FrozenSignalSpec
from app.alpha_mining.evidence import AlphaEvidenceError, AlphaEvidenceStore
from app.alpha_mining.policy import HARD_GATES
from app.alpha_mining.providers import DeclarativeCandidateRenderer


def _candidate() -> FrozenSignalSpec:
    return FrozenSignalSpec.from_candidate(CandidateSpec(
        recipe_id="test.factor",
        engine_id="test_engine",
        engine_version="1.0.0",
        name="Test",
        thesis="Synthetic contract",
        signal_kind="factor_rank",
        features=("momentum_5d",),
        directions=(1,),
        weights=(1.0,),
        parameters={"top_rank": 20},
        train_evidence={"ic": 0.1},
    ))


def _freeze(store: AlphaEvidenceStore):
    store.create_experiment("alpha-test", {"request": {"forward_horizon": 5}})
    frozen = _candidate()
    return store.freeze_candidate(
        run_id="alpha-test",
        engine_id="test_engine",
        candidate=frozen.to_dict(),
        renderer=dict(DeclarativeCandidateRenderer().render(frozen)),
    )


def test_candidate_evidence_is_immutable_and_state_cannot_skip(tmp_path) -> None:
    store = AlphaEvidenceStore(tmp_path)
    candidate = _freeze(store)
    candidate_id = candidate["candidate_id"]
    with pytest.raises(ValueError, match="invalid Alpha state transition"):
        store.transition(candidate_id, "champion", {})
    store.record_outer_evaluation(candidate_id, {"metrics": {}, "gates": []}, "rejected")
    with pytest.raises(AlphaEvidenceError, match="冲突"):
        store.record_outer_evaluation(candidate_id, {"metrics": {"return": 1}}, "rejected")


def test_candidate_state_event_hash_chain_detects_tampering(tmp_path) -> None:
    store = AlphaEvidenceStore(tmp_path)
    candidate = _freeze(store)
    path = store._candidate_dir(candidate["candidate_id"]) / "events.jsonl"
    events = path.read_text(encoding="utf-8").splitlines()
    item = json.loads(events[0])
    item["to"] = "champion"
    events[0] = json.dumps(item)
    path.write_text("\n".join(events) + "\n", encoding="utf-8")
    with pytest.raises(AlphaEvidenceError, match="哈希链"):
        store.events(candidate["candidate_id"])


def test_dynamic_champion_requires_every_historical_gate_and_forward_pass(tmp_path) -> None:
    store = AlphaEvidenceStore(tmp_path)
    candidate = _freeze(store)
    candidate_id = candidate["candidate_id"]
    gates = [
        {"id": gate["id"], "status": "passed"}
        for gate in HARD_GATES
        if gate["id"] != "forward_shadow"
    ]
    store.record_outer_evaluation(candidate_id, {"metrics": {}, "gates": gates}, "research_candidate")
    store.transition(candidate_id, "shadow", {})
    store.write_forward_evaluation(candidate_id, {"verdict": "passed"})
    store.transition(candidate_id, "challenger", {})
    champion = AlphaChampionStore(tmp_path, store).promote(candidate_id, "alpha_factor_test")
    assert champion["current"]["candidate_id"] == candidate_id
    assert store.get_state(candidate_id)["state"] == "champion"


def test_champion_ledger_starts_empty_instead_of_using_an_existing_strategy(tmp_path) -> None:
    current = AlphaChampionStore(tmp_path).get()["current"]

    assert current["kind"] == "none"
    assert current["strategy_id"] is None
    assert current["candidate_id"] is None


def test_experiment_result_is_write_once_and_lists_failed_trials(tmp_path) -> None:
    store = AlphaEvidenceStore(tmp_path)
    store.create_experiment("alpha-ledger", {"budget": 3})
    result = {"trial_ledger": [{"trial": 1, "status": "failed"}]}
    store.record_experiment_result("alpha-ledger", result)
    assert store.list_experiments()[0]["result"]["result"] == result
    with pytest.raises(AlphaEvidenceError, match="冲突"):
        store.record_experiment_result("alpha-ledger", {"trial_ledger": []})
