from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_cn_commodity_futures_trend_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_trend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_roll_signal_uses_new_contract_intraday_return_not_contract_gap() -> None:
    days = [date(2020, 1, 2), date(2020, 1, 3)]
    continuous = pl.DataFrame(
        {"series": ["M.DCE", "M.DCE"], "date": days}
    )
    mapping = pl.DataFrame(
        {
            "series": ["M.DCE", "M.DCE"],
            "date": days,
            "contract": ["M2001.DCE", "M2005.DCE"],
        }
    )
    contract_daily = pl.DataFrame(
        {
            "contract": ["M2001.DCE", "M2005.DCE"],
            "date": days,
            "open": [100.0, 200.0],
            "settle": [100.0, 202.0],
        }
    )

    result = study.prepare_signal_panel(continuous, mapping, contract_daily)

    assert result["roll_changed"][1] is True
    assert result["corrected_return"][1] == pytest.approx(0.01)


def test_target_quantity_rounds_toward_zero_and_keeps_direction() -> None:
    assert study.target_quantity(200_000, 0.50, 3_000, 10) == 3
    assert study.target_quantity(200_000, -0.50, 3_000, 10) == -3
    assert study.target_quantity(200_000, 0.01, 40_000, 10) == 0


def test_failed_roll_close_does_not_open_or_overwrite_new_contract() -> None:
    first = date(2020, 1, 2)
    roll = date(2020, 1, 3)
    signals = pl.DataFrame(
        {
            "entry_date": [first],
            "series": ["M.DCE"],
            "target_weight": [1.0],
        }
    )
    mapping = pl.DataFrame(
        {
            "date": [first, roll],
            "series": ["M.DCE", "M.DCE"],
            "contract": ["M2001.DCE", "M2005.DCE"],
        }
    )
    contract_daily = pl.DataFrame(
        {
            "date": [first, roll, roll],
            "contract": ["M2001.DCE", "M2001.DCE", "M2005.DCE"],
            "open": [100.0, 100.0, 101.0],
            "settle": [100.0, 100.0, 101.0],
            "volume": [1_000.0, 0.0, 1_000.0],
        }
    )
    contracts = pl.DataFrame(
        {"contract": ["M2001.DCE", "M2005.DCE"], "per_unit": [1.0, 1.0]}
    )

    result = study.simulate_account(
        signals,
        mapping,
        contract_daily,
        contracts,
        [first, roll],
        1_000.0,
    )

    assert all(order["contract"] != "M2005.DCE" for order in result["orders"])
    assert result["integrity"]["ending_positions"] == 1
    assert result["execution"]["rejections"] == {"REJECTED_CAPACITY": 2}


def test_one_price_limit_up_session_rejects_buy() -> None:
    previous = date(2020, 1, 2)
    locked = date(2020, 1, 3)
    signals = pl.DataFrame(
        {
            "entry_date": [locked],
            "series": ["M.DCE"],
            "target_weight": [1.0],
        }
    )
    mapping = pl.DataFrame(
        {
            "date": [previous, locked],
            "series": ["M.DCE", "M.DCE"],
            "contract": ["M2005.DCE", "M2005.DCE"],
        }
    )
    contract_daily = pl.DataFrame(
        {
            "date": [previous, locked],
            "contract": ["M2005.DCE", "M2005.DCE"],
            "open": [100.0, 110.0],
            "high": [101.0, 110.0],
            "low": [99.0, 110.0],
            "close": [100.0, 110.0],
            "settle": [100.0, 110.0],
            "volume": [1_000.0, 1_000.0],
        }
    )
    contracts = pl.DataFrame({"contract": ["M2005.DCE"], "per_unit": [1.0]})

    result = study.simulate_account(
        signals,
        mapping,
        contract_daily,
        contracts,
        [previous, locked],
        1_000.0,
    )

    assert result["execution"]["rejections"] == {
        "REJECTED_LIMIT_UP_LOCK": 1
    }
    assert result["integrity"]["ending_positions"] == 0


def test_gate_requires_both_capital_accounts() -> None:
    def account(annualized: float) -> dict:
        return {
            "metrics": {
                "annualized": annualized,
                "max_drawdown": -0.25,
                "positive_years": 5,
            },
            "execution": {"execution_rate": 0.98},
            "integrity": {
                "missing_settlement_quotes": 0,
                "margin_breaches": 0,
                "ending_positions": 0,
            },
        }

    accounts = {
        "cny_200k": account(0.60),
        "cny_300k": account(0.58),
        "cny_500k": account(0.56),
        "cny_1m": account(0.55),
    }
    benchmark = {"annualized": 0.20}

    assert study.evaluate_gate(accounts, benchmark)["passed"] is True
    accounts["cny_1m"] = account(0.49)
    assert study.evaluate_gate(accounts, benchmark)["passed"] is False
    del accounts["cny_300k"]
    assert study.evaluate_gate(accounts, benchmark)["passed"] is False
