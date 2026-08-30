from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_academic_factor_development_screen.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_academic_factor_screen", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_price_features_are_backward_looking() -> None:
    rows = 260
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ"] * rows,
            "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(rows)],
            "_global_index": list(range(rows)),
            "close": [float(i + 1) for i in range(rows)],
            "daily_return": [0.01] * rows,
            "amount": [1e8] * rows,
        }
    )

    result = study.attach_price_features(panel).tail(1)

    assert result["high_52week_proximity"][0] == pytest.approx(1.0)
    assert result["momentum_120d"][0] == pytest.approx(260 / 140 - 1)


def test_annual_factors_use_all_three_statement_availability_dates() -> None:
    income = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "report_period_end": [date(2018, 12, 31), date(2019, 12, 31)],
            "income_announce_date": [date(2019, 3, 1), date(2020, 3, 1)],
            "revenue": [100.0, 120.0],
            "operating_cost": [60.0, 70.0],
            "net_income_attributable": [10.0, 15.0],
        }
    )
    cashflow = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "report_period_end": [date(2018, 12, 31), date(2019, 12, 31)],
            "cashflow_announce_date": [date(2019, 3, 2), date(2020, 3, 3)],
            "net_operating_cash_flow": [12.0, 18.0],
        }
    )
    balance = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "report_period_end": [date(2018, 12, 31), date(2019, 12, 31)],
            "balance_announce_date": [date(2019, 3, 4), date(2020, 3, 5)],
            "total_assets": [100.0, 108.0],
            "total_liabilities": [40.0, 42.0],
            "total_equity": [60.0, 66.0],
        }
    )

    result = study.compute_annual_factors(income, cashflow, balance).tail(1)

    assert result["financial_available_date"][0] == date(2020, 3, 5)
    assert result["asset_growth"][0] == pytest.approx(0.08)
    assert result["gross_profitability"][0] == pytest.approx(50 / 108)
