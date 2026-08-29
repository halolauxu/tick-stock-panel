from __future__ import annotations

# Requirements: AM-S6-012 and AM-S9-001 through AM-S9-012.
from types import SimpleNamespace

from app.alpha_mining.champion import AlphaChampionStore
from app.alpha_mining.contracts import CandidateSpec, FrozenSignalSpec
from app.alpha_mining.evidence import AlphaEvidenceStore
from app.alpha_mining.policy import HARD_GATES
from app.alpha_mining.providers import DeclarativeCandidateRenderer
from app.alpha_mining.publication import AlphaPublicationService


class _PublishedStrategyEngine:
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.reload_count = 0

    def reload(self) -> None:
        self.reload_count += 1

    def get(self, strategy_id: str):
        return SimpleNamespace(
            meta={
                "id": strategy_id,
                "research_only": False,
                "alpha_candidate_id": self.candidate_id,
            },
            source="custom",
        )


def _challenger(tmp_path):
    evidence = AlphaEvidenceStore(tmp_path)
    evidence.create_experiment("alpha-publish", {"request": {"forward_horizon": 5}})
    candidate = CandidateSpec(
        recipe_id="publish.factor",
        engine_id="cross_sectional_rank",
        engine_version="1.0.0",
        name="Publish",
        thesis="immutable",
        signal_kind="factor_rank",
        features=("momentum_5d",),
        directions=(1,),
        weights=(1.0,),
        parameters={"entry_score": 70.0, "exit_score": 40.0, "top_rank": 20},
        train_evidence={"ic_mean": 0.1},
    )
    frozen = FrozenSignalSpec.from_candidate(candidate)
    stored = evidence.freeze_candidate(
        run_id="alpha-publish",
        engine_id=candidate.engine_id,
        candidate=frozen.to_dict(),
        renderer=dict(DeclarativeCandidateRenderer().render(frozen)),
    )
    candidate_id = stored["candidate_id"]
    gates = [
        {"id": gate["id"], "status": "passed"}
        for gate in HARD_GATES
        if gate["id"] != "forward_shadow"
    ]
    evidence.record_outer_evaluation(
        candidate_id,
        {"metrics": {"stitched_oos_return": 0.2}, "gates": gates},
        "research_candidate",
    )
    evidence.transition(candidate_id, "shadow", {})
    evidence.write_forward_evaluation(candidate_id, {"verdict": "passed"})
    evidence.transition(candidate_id, "challenger", {})
    return evidence, candidate_id


def test_challenger_publication_is_create_only_validated_and_promotable(tmp_path) -> None:
    evidence, candidate_id = _challenger(tmp_path)
    engine = _PublishedStrategyEngine(candidate_id)
    service = AlphaPublicationService(tmp_path, engine)
    publication = service.publish(candidate_id)
    strategy_id = publication["strategy_id"]
    target = tmp_path / "strategies" / "custom" / f"{strategy_id}.py"
    assert target.is_file()
    assert "alpha_candidate_sha256" in target.read_text(encoding="utf-8")
    assert evidence.get_candidate(candidate_id)["publication"] is not None

    repeated = service.publish(candidate_id)
    assert repeated["strategy_id"] == strategy_id
    assert target.read_text(encoding="utf-8").count("META =") == 1

    champion = AlphaChampionStore(tmp_path, evidence).promote(candidate_id, strategy_id)
    assert champion["current"]["candidate_id"] == candidate_id
    assert evidence.get_state(candidate_id)["state"] == "champion"
