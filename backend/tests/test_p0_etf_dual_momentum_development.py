from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_etf_dual_momentum_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_etf_momentum", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_prepare_panel_uses_adjusted_price_and_causal_momentum() -> None:
    days = [date(2013, 1, 1) + timedelta(days=index) for index in range(121)]
    daily = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "date": days,
            "open": [1.0] * len(days),
            "high": [1.0] * len(days),
            "low": [1.0] * len(days),
            "close": [1.0] * len(days),
            "volume": [1_000.0] * len(days),
            "amount": [100_000_000.0] * len(days),
            "source": ["test"] * len(days),
        }
    )
    adjustments = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "trade_date": days,
            "adj_factor": [1.0 + index / 120 for index in range(len(days))],
        }
    )
    master = pl.DataFrame(
        {"symbol": ["510001.SH"], "list_date": [date(2010, 1, 1)]}
    )

    result = study.prepare_panel(daily, adjustments, master)

    assert result["raw_close"][-1] == 1.0
    assert result["close"][-1] == 2.0
    assert result["momentum_120d"][-1] == 1.0


def test_gate_requires_base_and_capacity_accounts() -> None:
    def account(annualized: float) -> dict:
        return {
            "metrics": {
                "annualized": annualized,
                "max_drawdown": -0.20,
                "positive_years": 6,
            },
            "execution": {
                "buy": {"execution_rate": 0.95},
                "sell": {"execution_rate": 0.95},
            },
            "integrity": {"max_cash_reconciliation_error": 0.0},
        }

    accounts = {"cny_200k": account(0.60), "cny_1m": account(0.55)}
    benchmark = {"annualized": 0.20}

    assert study.evaluate_gate(accounts, benchmark)["passed"] is True
    accounts["cny_1m"] = account(0.49)
    assert study.evaluate_gate(accounts, benchmark)["passed"] is False
