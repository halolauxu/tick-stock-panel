from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_convertible_bond_double_low_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_cb_double_low", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_double_low_score_and_amount_are_causal() -> None:
    days = [date(2017, 1, 1) + timedelta(days=index) for index in range(20)]
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH"] * 20,
            "date": days,
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [100.0] * 20,
            "amount": [20_000_000.0] * 20,
            "bond_value": [90.0] * 20,
            "bond_over_rate": [10.0] * 20,
            "cb_value": [95.0] * 20,
            "cb_over_rate": [5.0] * 20,
        }
    )
    master = pl.DataFrame(
        {
            "symbol": ["110001.SH"],
            "list_date": [date(2016, 1, 1)],
            "delist_date": [None],
            "maturity_date": [date(2022, 1, 1)],
        }
    )

    result = study.prepare_panel(daily, master)

    assert result["double_low_score"][-1] == 105.0
    assert result["mean_amount_20d"][-1] == 20_000_000.0


def test_gate_requires_base_and_capacity_accounts() -> None:
    def account(annualized: float) -> dict:
        return {
            "metrics": {
                "annualized": annualized,
                "max_drawdown": -0.20,
                "positive_years": 3,
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
