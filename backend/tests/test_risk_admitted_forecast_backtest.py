from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.strategy import get_strategy, list_strategies
from app.backtest.strategy import StrategyBacktestConfig, StrategyBacktestService
from app.services.risk_admitted_forecast_backtest import (
    EXECUTION_BACKEND,
    run_artifact_backtest,
    strategy_detail,
)
from app.services.risk_admitted_forecast_paper import RESULT_FILE, STRATEGY_ID


def _artifact(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "research" / RESULT_FILE
    path.parent.mkdir(parents=True)
    payload = {
        "contract_frozen": "2026-09-03",
        "results": {
            "known_stress": {
                "period": {"start": "2024-01-02", "end": "2026-08-28"},
                "daily_equity": [
                    {
                        "date": "2024-01-02",
                        "equity": 199_994,
                        "cash": 198_994,
                        "position_count": 1,
                        "cash_ratio": 0.99497,
                    },
                    {
                        "date": "2024-01-03",
                        "equity": 200_050,
                        "cash": 198_994,
                        "position_count": 1,
                        "cash_ratio": 0.99472,
                    },
                    {
                        "date": "2024-01-04",
                        "equity": 200_087,
                        "cash": 200_087,
                        "position_count": 0,
                        "cash_ratio": 1.0,
                    },
                ],
                "orders": [
                    {
                        "date": "2024-01-02",
                        "signal_date": "2023-12-29",
                        "symbol": "600001.SH",
                        "side": "BUY",
                        "status": "FILLED",
                        "family": "main_board_microcap",
                        "target_weight": 0.05,
                        "raw_shares": 100,
                        "gross": 1_000,
                        "commission": 5,
                        "stamp_tax": 0,
                        "slippage": 1,
                    },
                    {
                        "date": "2024-01-04",
                        "symbol": "600001.SH",
                        "side": "SELL",
                        "status": "FILLED",
                        "family": "main_board_microcap",
                        "gross": 1_100,
                        "commission": 5,
                        "stamp_tax": 1,
                        "slippage": 1,
                    },
                ],
                "settlements": [],
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_adapter_restores_native_backtest_contract(tmp_path: Path):
    _, digest = _artifact(tmp_path)

    result = run_artifact_backtest(
        tmp_path,
        start=date(2024, 1, 2),
        end=date(2024, 1, 4),
        expected_sha256=digest,
    )

    assert result["stats"]["execution_backend"] == EXECUTION_BACKEND
    assert result["stats"]["n_trades"] == 1
    assert result["trades"][0]["symbol"] == "600001.SH"
    assert result["trades"][0]["pnl_amount"] == pytest.approx(87)
    assert result["config"]["max_positions"] == 20
    assert result["config"]["position_sizing"] == "frozen_target_weight"
    assert len(result["equity_curve"]) == 3


def test_artifact_adapter_fails_closed_on_tamper(tmp_path: Path):
    path, digest = _artifact(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="哈希不一致"):
        run_artifact_backtest(
            tmp_path,
            start=date(2024, 1, 2),
            end=date(2024, 1, 4),
            expected_sha256=digest,
        )


def test_managed_portfolio_is_only_added_to_backtest_catalog(tmp_path: Path):
    class EmptyEngine:
        def list_strategies(self):
            return []

        def load_errors(self):
            return []

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                strategy_engine=EmptyEngine(),
                repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
            )
        )
    )

    standard = list_strategies(request, asset_type="stock", timeframe="1d")
    backtest = list_strategies(
        request,
        asset_type="stock",
        timeframe="1d",
        context="backtest",
    )

    assert standard["strategies"] == []
    assert [row["id"] for row in backtest["strategies"]] == [STRATEGY_ID]
    assert backtest["strategies"][0]["immutable_contract"] is True
    assert backtest["strategies"][0]["artifact_verified"] is False


def test_managed_detail_does_not_require_generic_strategy_registration(tmp_path: Path):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)))
        )
    )

    result = get_strategy(STRATEGY_ID, request)

    assert result["id"] == STRATEGY_ID
    assert result["execution_backend"] == EXECUTION_BACKEND
    assert strategy_detail(tmp_path)["backtest_defaults"]["end"] == "2026-08-28"


def test_strategy_backtest_service_dispatches_before_generic_registry(monkeypatch, tmp_path: Path):
    class GenericRegistryMustNotRun:
        def get(self, _strategy_id):
            raise AssertionError("generic registry must not execute managed portfolio")

    engine = SimpleNamespace(repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)))
    service = StrategyBacktestService(engine, GenericRegistryMustNotRun())

    def fake_adapter(*_args, **_kwargs):
        return {
            "config": {"start": "2024-01-02", "end": "2024-01-03"},
            "stats": {"mode": "position", "total_return": 0.01},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "trades": [],
            "open_positions": [],
            "pending_orders": [],
            "per_symbol_stats": [],
            "strategy_info": {"id": STRATEGY_ID},
        }

    monkeypatch.setattr(
        "app.services.risk_admitted_forecast_backtest.run_artifact_backtest",
        fake_adapter,
    )
    result = service.run(
        StrategyBacktestConfig(
            strategy_id=STRATEGY_ID,
            symbols=None,
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
        )
    )

    assert result.error is None
    assert result.stats["total_return"] == 0.01
