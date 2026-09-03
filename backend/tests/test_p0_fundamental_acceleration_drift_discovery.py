from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_fundamental_acceleration_drift_discovery as study  # noqa: E402


def _metric(
    period_end: date,
    announce_date: date,
    revenue_yoy: float,
    net_income_yoy: float,
    cash: float = 10.0,
) -> dict[str, object]:
    return {
        "symbol": "600000.SH",
        "period_end": period_end,
        "announce_date": announce_date,
        "roe": 8.0,
        "debt_to_asset_ratio": 50.0,
        "revenue_yoy": revenue_yoy,
        "net_income_yoy": net_income_yoy,
        "operating_cash_to_revenue": cash,
    }


def test_comparison_requires_consecutive_period_and_monotonic_announcement() -> None:
    rows = [
        _metric(date(2019, 12, 31), date(2020, 3, 1), 5.0, 10.0),
        _metric(date(2020, 3, 31), date(2020, 4, 20), 12.0, 25.0),
        _metric(date(2020, 9, 30), date(2020, 10, 20), 20.0, 40.0),
    ]

    result = study.build_report_comparisons(pl.DataFrame(rows))

    assert result.height == 1
    assert result["period_end"][0] == date(2020, 3, 31)
    assert result["revenue_acceleration"][0] == 7.0
    assert result["profit_acceleration"][0] == 15.0


def test_classification_separates_acceleration_and_cash_quality() -> None:
    base = {
        "symbol": ["600000.SH", "600001.SH", "600002.SH"],
        "period_end": [date(2020, 3, 31)] * 3,
        "announce_date": [date(2020, 4, 20)] * 3,
        "roe": [8.0] * 3,
        "debt_to_asset_ratio": [50.0] * 3,
        "revenue_yoy": [20.0] * 3,
        "net_income_yoy": [30.0] * 3,
        "operating_cash_to_revenue": [10.0, 10.0, -1.0],
        "revenue_acceleration": [6.0, 4.0, 6.0],
        "profit_acceleration": [11.0, 11.0, 11.0],
    }

    result = study.classify_events(pl.DataFrame(base))
    categories = dict(result.select("symbol", "category").iter_rows())

    assert categories == {
        "600000.SH": study.CANDIDATE,
        "600001.SH": study.LEVEL_CONTROL,
        "600002.SH": study.CASH_POOR_CONTROL,
    }


def test_evaluate_requires_candidate_to_beat_both_controls() -> None:
    candidate = {
        "universe_eligible_events": 1_100,
        "tradable_events": 1_050,
        "announcement_days": 350,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 5,
        "mean_net_return": 0.02,
        "mean_excess_return": 0.015,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.30,
    }
    primary = {
        study.CANDIDATE: candidate,
        study.LEVEL_CONTROL: {"mean_excess_return": 0.009},
        study.CASH_POOR_CONTROL: {"mean_excess_return": 0.005},
    }

    decision = study.evaluate(primary)

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_ACCOUNT_CONTRACT"
