from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_etf_premium_reversal_development.py"
    )
    spec = importlib.util.spec_from_file_location(
        "p0_etf_premium_reversal", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_prepare_panel_keeps_raw_signal_price_and_adjusted_mark() -> None:
    days = [date(2019, 1, 1) + timedelta(days=index) for index in range(25)]
    daily = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "date": days,
            "open": [1.0] * len(days),
            "high": [1.0] * len(days),
            "low": [1.0] * len(days),
            "close": [1.0] * len(days),
            "volume": [10_000.0] * len(days),
            "amount": [30_000_000.0] * len(days),
            "source": ["test"] * len(days),
        }
    )
    adjustments = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "trade_date": days,
            "adj_factor": [2.0] * len(days),
        }
    )
    master = pl.DataFrame(
        {
            "symbol": ["510001.SH"],
            "list_date": [date(2010, 1, 1)],
            "delist_date": [None],
        },
        schema_overrides={"delist_date": pl.Date},
    )

    result = study.prepare_panel(daily, adjustments, master)

    assert result["raw_close"][-1] == 1.0
    assert result["close"][-1] == 2.0
    assert result["mean_amount_20d"][-1] == 30_000_000.0


def test_candidates_use_exact_nav_and_strict_pre_entry_announcement() -> None:
    symbols = [f"{510000 + index:06d}.SH" for index in range(20)]
    signal_date = date(2020, 1, 3)
    entry_date = date(2020, 1, 6)
    panel = pl.DataFrame(
        {
            "symbol": symbols,
            "date": [signal_date] * len(symbols),
            "raw_close": [0.80, 0.85] + [1.0 + index / 100 for index in range(18)],
            "listing_days": [500] * len(symbols),
            "mean_amount_20d": [30_000_000.0] * len(symbols),
            "amount": [30_000_000.0] * len(symbols),
        }
    )
    nav = pl.DataFrame(
        {
            "symbol": symbols,
            "nav_date": [signal_date] * len(symbols),
            "ann_date": [date(2020, 1, 4)] * 19 + [entry_date],
            "unit_nav": [1.0] * len(symbols),
        }
    )
    schedule = pl.DataFrame(
        {"signal_date": [signal_date], "entry_date": [entry_date]}
    )

    low, high, audit = study.build_candidate_legs(panel, nav, schedule)

    assert low["symbol"].to_list() == symbols[:2]
    assert low["premium"].to_list() == pytest.approx([-0.2, -0.15])
    assert symbols[-1] not in high["symbol"].to_list()
    assert audit["same_day_or_late_announcement_rows"] == 1


def test_weekly_schedule_does_not_open_beyond_development_period() -> None:
    panel = pl.DataFrame(
        {
            "date": [
                date(2020, 12, 24),
                date(2020, 12, 25),
                date(2020, 12, 28),
                date(2020, 12, 31),
                date(2021, 1, 4),
            ]
        }
    )

    result = study.weekly_schedule(panel)

    assert result.filter(pl.col("entry_date") > date(2020, 12, 31)).is_empty()


def test_gate_requires_direction_and_market_excess_for_all_capital_levels() -> None:
    def result(annualized: float) -> dict:
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
            "integrity": {
                "max_cash_reconciliation_error": 0.0,
                "ending_unresolved_positions": 0,
            },
            "account": {"ending_positions": 0},
        }

    low = {f"cny_{cash}k": result(0.60) for cash in (200, 300, 500, 1000)}
    high = {f"cny_{cash}k": result(0.30) for cash in (200, 300, 500, 1000)}
    decision = study.evaluate_gate(low, high, {"annualized": 0.30}, 100)

    assert decision["passed"] is True
    high["cny_1000k"] = result(0.41)
    assert study.evaluate_gate(low, high, {"annualized": 0.30}, 100)[
        "passed"
    ] is False
