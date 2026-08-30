from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_earnings_surprise_drift_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_earnings_surprise_drift", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_sue_uses_single_quarter_eps_and_only_prior_innovation_scale() -> None:
    rows = []
    increments = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.12,
        0.24,
        0.33,
        0.46,
        0.15,
        0.29,
        0.40,
        0.60,
        0.30,
        0.50,
        0.70,
        1.00,
    ]
    for offset, year in enumerate(range(2010, 2014)):
        cumulative = 0.0
        for quarter, month in enumerate((3, 6, 9, 12), start=1):
            cumulative += increments[offset * 4 + quarter - 1]
            rows.append(
                {
                    "symbol": "A.SZ",
                    "report_period_end": date(year, month, 30),
                    "report_announce_date": date(year, month, 30)
                    + timedelta(days=30),
                    "eps_basic": cumulative,
                }
            )

    result = study.compute_sue(pl.DataFrame(rows))

    assert result.filter(
        pl.col("report_period_end") == date(2010, 6, 30)
    )["quarter_eps"][0] == pytest.approx(0.20)
    assert result.sort("report_period_end")["sue"][-1] is not None


def test_positive_sue_event_remains_active_for_sixty_trading_days() -> None:
    all_dates = [date(2020, 1, 2) + timedelta(days=i) for i in range(70)]
    event = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "report_period_end": [date(2019, 12, 31)],
            "report_announce_date": [date(2020, 1, 2)],
            "quarter_eps": [0.5],
            "eps_innovation": [0.2],
            "historical_innovation_std": [0.05],
            "sue": [4.0],
            "revenue_yoy": [20.0],
            "roe": [10.0],
            "operating_cash_to_revenue": [10.0],
            "debt_to_asset_ratio": [40.0],
        }
    )
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "date": [date(2020, 1, 2)],
            "raw_close": [10.0],
            "amount": [1e8],
            "market_cap": [1e10],
        }
    )

    result = study.build_candidates(event, panel, all_dates)

    assert result.height == study.HOLDING_TRADING_DAYS
    assert result["entry_date"][0] == date(2020, 1, 3)
