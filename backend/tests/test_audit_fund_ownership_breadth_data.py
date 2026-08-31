from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_fund_ownership_breadth_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_fund_ownership_breadth_data", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()


def _row(period: date, symbol: str, coverage: int, market_count: int) -> dict:
    return {
        "period_end": period,
        "available_after": period,
        "symbol": symbol,
        "name_at_source": symbol,
        "fund_coverage_count": coverage,
        "market_fund_count": market_count,
        "coverage_share": coverage / market_count,
        "total_shares": 1_000_000.0,
        "total_market_value_cny": coverage * 2_000_000.0,
        "average_market_value_per_fund_cny": 2_000_000.0,
        "partition_year": period.year,
        "partition_quarter": (period.month - 1) // 3 + 1,
    }


def test_required_periods_cover_38_quarters() -> None:
    periods = audit_module.required_periods()

    assert len(periods) == 38
    assert periods[0] == (2017, 1)
    assert periods[-1] == (2026, 2)


def test_same_depth_change_uses_two_quarter_lag() -> None:
    events = pl.DataFrame(
        [
            _row(date(2020, 3, 31), "000001.SZ", 10, 1_000),
            _row(date(2020, 6, 30), "000001.SZ", 100, 1_000),
            _row(date(2020, 9, 30), "000001.SZ", 30, 1_000),
        ]
    )
    quality = audit_module.quarter_quality(events)
    changes = audit_module.same_depth_changes(events, quality)
    september = changes.filter(pl.col("period_end") == date(2020, 9, 30))

    assert september["previous_fund_coverage_count"][0] == 10
    assert september["fund_count_increase"][0] == 20
    assert september["coverage_share_growth"][0] == 2.0


def test_quarter_quality_rejects_collapsed_same_depth_coverage() -> None:
    events = pl.DataFrame(
        [
            _row(date(2025, 12, 31), "000001.SZ", 1_000, 10_000),
            _row(date(2026, 6, 30), "000001.SZ", 62, 14_000),
        ]
    )
    quality = audit_module.quarter_quality(events)

    assert quality.filter(pl.col("period_end") == date(2026, 6, 30))[
        "complete"
    ][0] is False
