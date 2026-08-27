# ruff: noqa: RUF001
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

import app.services.serenity_pdf_scoring as scoring
from app.services.serenity_pdf_scoring import (
    DIMENSION_WEIGHTS,
    PENALTY_MULTIPLIER,
    CachedModelResult,
    ModelCallSpec,
    compute_full_score,
    execute_cached_call,
    initialize_semantic_tables,
    validate_score_output,
)
from app.services.serenity_pilot import PilotStore


def _score_output() -> dict:
    dimensions = []
    for dimension_id in DIMENSION_WEIGHTS:
        dimensions.append(
            {
                "dimension_id": dimension_id,
                "status": "EVIDENCED",
                "rating": 3,
                "reason": "存在可核验的原文依据",
                "evidence": [
                    {
                        "document_id": "doc-1",
                        "page_number": 1,
                        "quote": "唯一供应商，扩产建设周期为24个月。",
                        "direction": "SUPPORT",
                    }
                ],
            }
        )
    return {
        "schema_version": "1.0.0",
        "entity_id": "ENTITY-1",
        "cutoff": "T0",
        "dimensions": dimensions,
        "penalties": [
            {
                "penalty_id": penalty_id,
                "status": "UNKNOWN",
                "rating": None,
                "reason": "PDF未提供足够信息",
                "evidence": [],
            }
            for penalty_id in (
                "dilution_financing",
                "governance",
                "geopolitics",
                "liquidity",
                "hype_risk",
                "accounting_quality",
                "cyclicality",
                "alternative_design_risk",
            )
        ],
        "kill_switches": [],
    }


def test_cached_model_call_is_persisted_and_reused_without_runner(tmp_path: Path) -> None:
    store = PilotStore(tmp_path / "pilot")
    initialize_semantic_tables(store.connection)
    calls = 0

    def runner(_spec: ModelCallSpec) -> CachedModelResult:
        nonlocal calls
        calls += 1
        raw = json.dumps(_score_output(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        return CachedModelResult(
            raw_output=raw,
            response_sha256=hashlib.sha256(b"provider-response").hexdigest(),
            output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            context_id="DEEPSEEK-test-context",
            api_request_id="req-test",
            system_fingerprint=None,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            cost_micros_cny=1234,
        )

    spec = ModelCallSpec(
        replay_id="serenity-full64-test",
        stage="SERENITY_PDF_SCORE",
        slot_id="SLOT-SERENITY-ENTITY-1",
        trade_date=date(2026, 8, 26),
        prompt="frozen prompt",
        schema_path=tmp_path / "score.schema.json",
    )
    spec.schema_path.write_text("{}", encoding="utf-8")
    first = execute_cached_call(store.connection, spec, runner)
    second = execute_cached_call(store.connection, spec, runner)
    row = store.connection.execute(
        "SELECT status, raw_output_json, cost_micros_cny FROM semantic_model_calls"
    ).fetchone()
    store.close()

    assert calls == 1
    assert first.raw_output == second.raw_output
    assert row == ("PASS", first.raw_output, 1234)


def test_cache_key_includes_request_hash(tmp_path: Path) -> None:
    store = PilotStore(tmp_path / "pilot")
    initialize_semantic_tables(store.connection)
    calls = 0

    def runner(spec: ModelCallSpec) -> CachedModelResult:
        nonlocal calls
        calls += 1
        raw = json.dumps({**_score_output(), "cutoff": spec.prompt}, ensure_ascii=False)
        return CachedModelResult(
            raw_output=raw,
            response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            context_id=f"DEEPSEEK-{calls}",
            api_request_id=f"req-{calls}",
            system_fingerprint=None,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            cost_micros_cny=1,
        )

    base = dict(
        replay_id="serenity-full64-test",
        stage="SERENITY_PDF_SCORE",
        slot_id="SLOT-SERENITY-ENTITY-1",
        trade_date=date(2026, 8, 26),
        schema_path=tmp_path / "score.schema.json",
    )
    base["schema_path"].write_text("{}", encoding="utf-8")
    execute_cached_call(store.connection, ModelCallSpec(prompt="one", **base), runner)
    execute_cached_call(store.connection, ModelCallSpec(prompt="two", **base), runner)
    rows = store.connection.execute("SELECT count(*) FROM semantic_model_calls").fetchone()[0]
    store.close()

    assert calls == 2
    assert rows == 2


def test_cached_model_call_fails_closed_if_persisted_raw_json_is_corrupted(
    tmp_path: Path,
) -> None:
    store = PilotStore(tmp_path / "pilot")
    initialize_semantic_tables(store.connection)
    schema_path = tmp_path / "score.schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    spec = ModelCallSpec(
        replay_id="serenity-full64-test",
        stage="SERENITY_PDF_SCORE",
        slot_id="SLOT-SERENITY-ENTITY-1",
        trade_date=date(2026, 8, 26),
        prompt="frozen prompt",
        schema_path=schema_path,
    )
    raw = json.dumps(_score_output(), ensure_ascii=False)

    def runner(_spec: ModelCallSpec) -> CachedModelResult:
        return CachedModelResult(
            raw_output=raw,
            response_sha256=hashlib.sha256(b"response").hexdigest(),
            output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            context_id="DEEPSEEK-test-context",
            api_request_id="req-test",
            system_fingerprint=None,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            cost_micros_cny=1,
        )

    execute_cached_call(store.connection, spec, runner)
    store.connection.execute("UPDATE semantic_model_calls SET raw_output_json='{}'")

    with pytest.raises(RuntimeError, match="persisted model output hash mismatch"):
        execute_cached_call(store.connection, spec, runner)
    store.close()


def test_score_validation_rejects_unbound_quote() -> None:
    output = _score_output()
    output["dimensions"][0]["evidence"][0]["quote"] = "模型虚构的句子"

    errors = validate_score_output(
        output,
        entity_id="ENTITY-1",
        documents={"doc-1": {1: "唯一供应商，扩产建设周期为24个月。"}},
    )

    assert any("原文" in error for error in errors)


def test_score_validation_requires_counter_evidence_for_zero_rating() -> None:
    output = _score_output()
    output["dimensions"][0]["rating"] = 0

    errors = validate_score_output(
        output,
        entity_id="ENTITY-1",
        documents={"doc-1": {1: "唯一供应商，扩产建设周期为24个月。"}},
    )

    assert any("0分" in error for error in errors)


def test_full_score_keeps_unknown_dimensions_unscored() -> None:
    output = _score_output()
    output["dimensions"][0].update(
        {"status": "UNKNOWN", "rating": None, "evidence": [], "reason": "未知"}
    )

    result = compute_full_score(30.0, output)

    assert result["pdf_points"] is None
    assert result["raw_factor_score"] is None
    assert result["status"] == "DATA_INSUFFICIENT"


def test_full_score_uses_original_weights_and_known_penalties() -> None:
    output = _score_output()
    output["penalties"][0].update({"status": "EVIDENCED", "rating": 2, "reason": "存在融资压力"})

    result = compute_full_score(30.0, output)

    expected_pdf = sum(3 / 5 * weight for weight in DIMENSION_WEIGHTS.values())
    assert result["pdf_points"] == round(expected_pdf, 2)
    assert result["raw_factor_score"] == round(30.0 + expected_pdf, 2)
    assert result["known_penalty_points"] == 2 * PENALTY_MULTIPLIER
    assert result["risk_adjusted_score_upper_bound"] == round(30.0 + expected_pdf - 4, 2)
    assert result["risk_adjusted_score_exact"] is None
    assert result["status"] == "FACTOR_SCORE_COMPLETE_RISK_INCOMPLETE"


def test_materialization_recomputes_current_day_base_score_and_requires_exact_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PilotStore(tmp_path / "pilot")
    initialize_semantic_tables(store.connection)
    decision_dates = [date(2026, 8, day) for day in range(20, 27)]
    store.set_meta("manifest", {"decision_dates": [value.isoformat() for value in decision_dates]})
    store.connection.execute(
        """
        INSERT INTO universe VALUES
        ('000001.SZ', '000001', '测试公司', 'chain-a', '产业链A', '环节A',
         1, 'mid', 1, '[]', 100.0, 10.0, DATE '2026-08-26')
        """
    )
    for decision_date, base_score in (
        (date(2026, 8, 25), 10.0),
        (date(2026, 8, 26), 30.0),
    ):
        store.connection.execute(
            "INSERT INTO decisions VALUES (?, '000001.SZ', 'chain-a', 'serenity', ?, 1, false, 'input', current_timestamp)",
            [decision_date, base_score],
        )
    output = _score_output()
    for dimension in output["dimensions"]:
        dimension["rating"] = 4
    for penalty in output["penalties"]:
        penalty.update(
            {
                "status": "CONTRADICTED",
                "rating": 0,
                "reason": "有明确反证",
                "evidence": [
                    {
                        "document_id": "doc-1",
                        "page_number": 1,
                        "quote": "唯一供应商，扩产建设周期为24个月。",
                        "direction": "COUNTER",
                    }
                ],
            }
        )
    initial = compute_full_score(10.0, output)
    store.connection.execute(
        """
        INSERT INTO semantic_score_results VALUES
        (?, '000001.SZ', DATE '2026-08-25', 'ENTITY-1', ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, 'input', current_timestamp)
        """,
        [
            scoring.REPLAY_ID,
            initial["status"],
            initial["base_36_points"],
            initial["pdf_points"],
            initial["raw_factor_score"],
            initial["known_penalty_points"],
            initial["risk_adjusted_score_upper_bound"],
            initial["risk_adjusted_score_exact"],
            json.dumps(output["dimensions"]),
            json.dumps(output["penalties"]),
            json.dumps(output),
        ],
    )
    monkeypatch.setattr(scoring, "_load_market_rows", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(scoring, "_load_index_rows", lambda *_args, **_kwargs: [])

    scoring.materialize_decisions_and_outcomes(store, {"run_root": tmp_path / "run"})
    rows = store.connection.execute(
        """
        SELECT decision_date, base_36_points, research_score, research_selected
        FROM serenity_full_decisions
        WHERE replay_id=? AND decision_date IN (DATE '2026-08-25', DATE '2026-08-26')
        ORDER BY decision_date
        """,
        [scoring.REPLAY_ID],
    ).fetchall()
    store.close()

    assert rows[0] == (date(2026, 8, 25), 10.0, 61.2, False)
    assert rows[1] == (date(2026, 8, 26), 30.0, 81.2, True)
