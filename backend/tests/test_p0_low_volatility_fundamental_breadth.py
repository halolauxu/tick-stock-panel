from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_low_volatility_fundamental_breadth as study  # noqa: E402


def test_breadth_state_compares_same_season_prior_year() -> None:
    rows = []
    for year, flags in ((2019, [True] * 4 + [False] * 6), (2020, [True] * 6 + [False] * 4)):
        for index, flag in enumerate(flags):
            rows.append(
                {
                    "symbol": f"600{year % 100:02d}{index:01d}.SH",
                    "announce_date": date(year, 3, 15),
                    "is_flag": flag,
                }
            )
    frame = pl.DataFrame(rows)
    original_minimum = study.MIN_REPORTS_PER_WINDOW
    study.MIN_REPORTS_PER_WINDOW = 1
    try:
        result = study.build_fundamental_breadth(
            frame.with_columns(pl.col("is_flag").alias("is_acceleration")),
            [date(2020, 4, 30)],
        )
    finally:
        study.MIN_REPORTS_PER_WINDOW = original_minimum

    assert result["fundamental_acceleration_breadth"][0] == 0.6
    assert result["prior_year_acceleration_breadth"][0] == 0.4
    assert result["fundamental_breadth_active"][0] is True


def _account(annualized: float, drawdown: float) -> dict:
    return {
        "metrics": {
            "annualized": annualized,
            "max_drawdown": drawdown,
            "positive_years": 5,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }


def test_evaluate_requires_risk_improvement_without_return_collapse() -> None:
    decision = study.evaluate(
        _account(0.16, -0.23),
        _account(0.17, -0.32),
        {"annualized": 0.10},
        0.50,
    )

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_VALIDATION_CONTRACT"
