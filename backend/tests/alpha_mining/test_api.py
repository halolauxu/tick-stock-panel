from __future__ import annotations

# Requirements: AM-S1-002, AM-S2-009, AM-S7-001 through AM-S7-010.
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.alpha_mining import router


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
