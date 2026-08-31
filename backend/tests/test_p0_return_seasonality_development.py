from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "run_p0_return_seasonality_development.py"
)
SPEC = importlib.util.spec_from_file_location("return_seasonality", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_monthly_returns_require_consecutive_calendar_months() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * 3,
            "month_end": [
                date(2010, 1, 29),
                date(2010, 2, 26),
                date(2010, 4, 30),
            ],
            "adjusted_close": [10.0, 11.0, 22.0],
        }
    )
    result = study._monthly_returns(frame, date_column="month_end", close_column="adjusted_close")
    returns = result.get_column("monthly_return").to_list()
    assert returns[0] is None
    assert returns[1] == pytest.approx(0.1)
    assert returns[2] is None


def test_scores_use_five_prior_same_months_and_exclude_target_year() -> None:
    rows = []
    for year in range(2008, 2013):
        for month in range(1, 13):
            rows.append(
                {
                    "symbol": "600000.SH",
                    "year": year,
                    "month": month,
                    "monthly_return": month / 100.0,
                }
            )
    scores = study.build_seasonality_scores(pl.DataFrame(rows))
    january_2013 = scores.filter(
        (pl.col("target_year") == 2013) & (pl.col("target_month") == 1)
    ).to_dicts()[0]
    assert january_2013["same_month_count"] == 5
    assert january_2013["other_month_count"] == 55
    assert january_2013["same_month_score"] == 0.01
    assert january_2013["other_month_score"] > january_2013["same_month_score"]


def test_candidates_use_only_top_decile_for_requested_score() -> None:
    count = 100
    ranked = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * count,
            "entry_date": [date(2020, 2, 3)] * count,
            "symbol": [f"{index:06d}.SZ" for index in range(count)],
            "target_year": [2020] * count,
            "target_month": [2] * count,
            "same_month_score": [float(index) for index in range(count)],
            "other_month_score": [float(99 - index) for index in range(count)],
            "same_month_decile": [min(index // 10, 9) for index in range(count)],
            "other_month_decile": [min((99 - index) // 10, 9) for index in range(count)],
            "market_cap_decile": [5] * count,
            "amount": [100_000_000.0] * count,
        }
    )
    same = study.build_candidates(ranked, use_same_month=True)
    other = study.build_candidates(ranked, use_same_month=False)
    assert same.height == 10
    assert other.height == 10
    assert same.get_column("symbol")[0] == "000099.SZ"
    assert other.get_column("symbol")[0] == "000000.SZ"


def test_gate_requires_increment_and_closed_account() -> None:
    candidate = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_full_years": 6,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_open_positions": 1,
            "max_cash_reconciliation_error": 0.0,
        },
        "completed_trades": 400,
        "profit_concentration": {"largest_positive_symbol_share": 0.10},
    }
    control = {"metrics": {"annualized": 0.55}}
    benchmark = {"annualized": 0.10}
    decision = study.evaluate_gate(candidate, control, benchmark)
    assert decision["passed"] is False
    assert decision["checks"]["other_month_increment_at_least_10pp"] is False
    assert decision["checks"]["no_ending_open_positions"] is False
