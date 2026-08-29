from __future__ import annotations

# Requirements: AM-S1-002, AM-S2-009, AM-S7-001 through AM-S7-010.
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.alpha_mining.hypotheses import AlphaHypothesisStore
from app.api.alpha_mining import (
    AlphaMiningStartRequest,
    _attach_failure_lineage,
    _attach_hypothesis_contract,
    _validate_factor_datasets,
    router,
)


def test_alpha_catalog_exposes_frozen_registry_and_open_coverage_slot() -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    engines = client.get("/api/alpha-mining/v1/engines")
    assert engines.status_code == 200
    body = engines.json()
    assert body["registry_frozen"] is True
    assert len(body["items"]) == 8
    assert {item["engine_id"] for item in body["items"]} >= {
        "cross_sectional_rank",
        "event_sequence",
        "financial_revision",
        "market_sector_timing",
        "matched_outcomes",
        "network_diffusion",
        "nonlinear_interaction",
        "portfolio_residual",
    }
    assert set(body["taxonomy"]) == {
        "information_domain",
        "mechanism",
        "discovery",
        "prediction_object",
    }

    charter = client.get("/api/alpha-mining/v1/charter")
    assert charter.status_code == 200
    roadmap = charter.json()["coverage_roadmap"]
    assert any(item["status"] == "extension_slot" for item in roadmap)
    assert charter.json()["extension_contract"]["orchestrator_edit_required_for_new_engine"] is False


def test_financial_factor_is_rejected_when_announcement_time_dataset_is_unavailable() -> None:
    catalog = SimpleNamespace(datasets={
        "financial_pit": SimpleNamespace(ready=False, reasons=("缺少公告时间",)),
        "share_history_pit": SimpleNamespace(ready=False, reasons=("缺少点时股本",)),
    })

    with pytest.raises(ValueError, match="所选财务因子不能进入正式历史研究"):
        _validate_factor_datasets(["momentum_5d", "roe_latest"], catalog)

    _validate_factor_datasets(["momentum_5d"], catalog)

    with pytest.raises(ValueError, match="所选换手率因子不能进入正式历史研究"):
        _validate_factor_datasets(["turnover_z_60d"], catalog)


def test_hypothesis_run_request_accepts_browser_iso_dates() -> None:
    from app.api.alpha_mining import AlphaHypothesisRunRequest

    payload = AlphaHypothesisRunRequest.model_validate({
        "start": "2025-08-26",
        "end": "2026-08-28",
    })
    assert payload.start == date(2025, 8, 26)
    assert payload.end == date(2026, 8, 28)


def test_failure_lineage_requires_real_suggestion_and_freezes_explicit_diff() -> None:
    source_request = {
        "engine_ids": ["cross_sectional_rank", "matched_outcomes"],
        "factor_names": ["momentum_5d"],
        "asset_type": "stock",
        "start": "2025-01-01",
        "end": "2026-01-01",
        "budget_profile": "exploratory",
        "forward_horizon": 5,
        "commission_pct": 0.0002,
        "stamp_tax_pct": 0.0005,
        "slippage_bps": 5.0,
        "max_positions": 10,
        "max_candidates_per_engine": 2,
        "max_trials_per_engine": 24,
    }
    source_result = {
        "next_research_suggestions": [{
            "suggestion_id": "next-capacity-1",
            "request_patch": {"engine_ids": ["cross_sectional_rank"]},
        }],
    }

    class Store:
        def get(self, run_id):
            assert run_id == "alpha-source"
            return {"run_id": run_id, "status": "succeeded", "request": source_request}

        def read_summary(self, run_id):
            assert run_id == "alpha-source"
            return source_result

    manager = SimpleNamespace(store=Store())
    before = deepcopy(source_result)
    next_request = {
        **source_request,
        "engine_ids": ["cross_sectional_rank"],
        "source_run_id": "alpha-source",
        "source_suggestion_id": "next-capacity-1",
    }

    _attach_failure_lineage(manager, next_request)

    assert source_result == before
    assert next_request["source_diff"] == {
        "engine_ids": {
            "before": ["cross_sectional_rank", "matched_outcomes"],
            "after": ["cross_sectional_rank"],
        },
    }


def test_failure_lineage_fields_must_be_paired() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        AlphaMiningStartRequest(
            engine_ids=["cross_sectional_rank"],
            factor_names=["momentum_5d"],
            source_run_id="alpha-source",
        )

    with pytest.raises(ValueError, match="exactly one lineage source"):
        AlphaMiningStartRequest(
            engine_ids=["cross_sectional_rank"],
            factor_names=["momentum_5d"],
            source_run_id="alpha-source",
            source_suggestion_id="next-1",
            source_candidate_id="ac-source",
        )


def test_hypothesis_contract_prevents_request_drift(tmp_path) -> None:
    store = AlphaHypothesisStore(tmp_path)
    hypothesis = store.get("ah-system-selling-exhaustion-v1")
    manager = SimpleNamespace(_data_dir=tmp_path)
    request = {
        "hypothesis_id": hypothesis["hypothesis_id"],
        "engine_ids": hypothesis["test_spec"]["engine_ids"],
        "factor_names": hypothesis["test_spec"]["factor_names"],
        "asset_type": hypothesis["asset_type"],
        "forward_horizon": hypothesis["forward_horizon"],
    }

    _attach_hypothesis_contract(manager, request)

    assert request["hypothesis_contract"]["thesis"] == hypothesis["thesis"]
    assert request["hypothesis_contract"]["test_spec"]["expected_directions"] == {
        "momentum_5d": -1,
        "vol_ratio_5d": -1,
        "close_position": 1,
    }

    drifted = {**request, "factor_names": ["momentum_5d"]}
    with pytest.raises(ValueError, match="偏离冻结Alpha假设"):
        _attach_hypothesis_contract(manager, drifted)


def test_hypothesis_api_supplies_system_ideas_and_accepts_manual_idea(tmp_path) -> None:
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    client = TestClient(app)

    catalog = client.get("/api/alpha-mining/v1/hypotheses")
    assert catalog.status_code == 200
    assert len(catalog.json()["items"]) >= 3
    assert all("readiness" in item for item in catalog.json()["items"])

    created = client.post("/api/alpha-mining/v1/hypotheses", json={
        "title": "低流动性折价修复",
        "thesis": "流动性改善且估值较低的股票未来10日净收益更高。",
        "mechanism": "流动性冲击消退后, 被迫卖出造成的折价可能均值回归。",
        "asset_type": "stock",
        "forward_horizon": 10,
        "information_domains": ["liquidity"],
        "test_spec": {
            "engine_ids": ["cross_sectional_rank"],
            "factor_names": ["amihud_20d", "turnover_z_60d"],
            "expected_directions": {"amihud_20d": -1, "turnover_z_60d": 1},
            "weights": {"amihud_20d": 0.5, "turnover_z_60d": 0.5},
        },
        "falsification": ["独立样本外硬门槛失败"],
        "data_requirements": ["daily_enriched", "historical_universe"],
    })
    assert created.status_code == 200
    hypothesis_id = created.json()["hypothesis_id"]
    assert client.get(f"/api/alpha-mining/v1/hypotheses/{hypothesis_id}").json()["title"] == "低流动性折价修复"

def test_strict_validation_lineage_is_full_history_and_candidate_scoped(monkeypatch) -> None:
    source_request = {
        "engine_ids": ["cross_sectional_rank", "matched_outcomes"],
        "factor_names": ["momentum_5d"],
        "asset_type": "stock",
        "start": "2025-01-01",
        "end": "2026-01-01",
        "budget_profile": "exploratory",
        "forward_horizon": 5,
        "commission_pct": 0.0002,
        "stamp_tax_pct": 0.0005,
        "slippage_bps": 5.0,
        "max_positions": 10,
        "max_candidates_per_engine": 2,
        "max_trials_per_engine": 24,
    }

    class Store:
        def get(self, _run_id):
            return {"status": "succeeded", "request": source_request}

        def read_summary(self, _run_id):
            return {"status": "succeeded"}

    class Evidence:
        def get_candidate(self, _candidate_id):
            return {
                "candidate_id": "ac-source",
                "run_id": "alpha-source",
                "engine_id": "cross_sectional_rank",
                "state": {"state": "validation_candidate"},
            }

    manager = SimpleNamespace(store=Store(), evidence=Evidence(), _data_dir="/tmp/alpha-test")
    monkeypatch.setattr(
        "app.api.alpha_mining.enriched_partition_dates",
        lambda *_args: [date(2013, 1, 4), date(2026, 1, 1)],
    )
    request = {
        **source_request,
        "engine_ids": ["cross_sectional_rank"],
        "start": "2013-01-04",
        "end": "2026-01-01",
        "budget_profile": "strict",
        "max_candidates_per_engine": 8,
        "max_trials_per_engine": 128,
        "source_run_id": "alpha-source",
        "source_candidate_id": "ac-source",
        "source_suggestion_id": None,
    }

    _attach_failure_lineage(manager, request)

    assert request["source_diff"]["engine_ids"]["after"] == ["cross_sectional_rank"]
    assert request["source_diff"]["budget_profile"] == {
        "before": "exploratory",
        "after": "strict",
    }
    assert request["source_diff"]["start"]["after"] == "2013-01-04"
