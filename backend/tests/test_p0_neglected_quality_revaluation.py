from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_neglected_quality_revaluation as study  # noqa: E402


def _ranked() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * 4,
            "entry_date": [date(2020, 1, 6)] * 4,
            "symbol": [f"60000{i}.SH" for i in range(4)],
            "market_cap": [2_000_000_000.0] * 4,
            "market_cap_percentile": [0.5] * 4,
            "size_bin": [0] * 4,
            "mean_turnover_20d": [0.01] * 4,
            "turnover_percentile": [0.05] * 4,
            "financial_available_date": [date(2019, 4, 1)] * 4,
            "financial_age_days": [277] * 4,
            "earnings_yield": [0.02, -0.01, 0.02, 0.02],
            "cashflow_yield": [0.03, 0.03, -0.01, 0.03],
            "roe_proxy": [0.12, -0.01, 0.12, 0.12],
            "debt_ratio": [0.60, 0.60, 0.60, 0.85],
            "amount": [100_000_000.0] * 4,
        }
    )


def test_quality_and_value_trap_are_directionally_separate() -> None:
    quality = study.build_quality_candidates(_ranked(), study.QUALITY)
    trap = study.build_quality_candidates(_ranked(), study.VALUE_TRAP)

    assert quality["symbol"].to_list() == ["600000.SH"]
    assert set(trap["symbol"].to_list()) == {
        "600001.SH",
        "600002.SH",
        "600003.SH",
    }


def test_gate_requires_improvement_over_low_turnover_mother() -> None:
    candidate = {
        "metrics": {
            "annualized": 0.25,
            "max_drawdown": -0.25,
            "positive_years": 5,
            "mean_cash_ratio": 0.40,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
        "account": {"trade_count": 400},
    }
    mother = {"metrics": {"annualized": 0.19, "max_drawdown": -0.31}}
    benchmark = {"annualized": 0.14}

    assert study.evaluate(candidate, mother, benchmark)["passed"] is True
    mother["metrics"]["annualized"] = 0.21
    assert study.evaluate(candidate, mother, benchmark)["passed"] is False
