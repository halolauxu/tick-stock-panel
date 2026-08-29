from __future__ import annotations

# Requirements: AM-S1-008 through AM-S1-010 and AM-S2-016.
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import mining
from app.optional_alpha import load_alpha_api


def test_missing_alpha_module_does_not_disable_legacy_mining(monkeypatch) -> None:
    def fail_alpha(name: str):
        if name == "app.api.alpha_mining":
            raise ImportError("simulated deleted Alpha subsystem")
        raise AssertionError(name)

    monkeypatch.setattr("app.optional_alpha.importlib.import_module", fail_alpha)
    assert load_alpha_api() is None

    app = FastAPI()
    app.include_router(mining.router)
    response = TestClient(app).get("/api/backtest/mining/config")
    assert response.status_code == 200
    assert "mining_schedule_enabled" in response.json()
