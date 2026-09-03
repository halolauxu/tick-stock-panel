from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_neglected_to_recognition_transition as study  # noqa: E402


def _ranked() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [study.base.DEVELOPMENT_START] * 5,
            "entry_date": [study.base.DEVELOPMENT_START] * 5,
            "symbol": [f"60000{i}.SH" for i in range(5)],
            "market_cap": [2_000_000_000.0] * 5,
            "market_cap_percentile": [0.5] * 5,
            "size_bin": [0] * 5,
            "mean_turnover_20d": [0.01] * 5,
            "turnover_percentile": [0.05] * 5,
            "turnover_transition_ratio": [1.2, 2.0, 2.01, 1.0, 0.9],
            "return_5d": [0.0, 0.05, 0.03, 0.0, -0.01],
            "amount": [100_000_000.0] * 5,
        }
    )


def test_transition_and_control_use_frozen_closed_bands() -> None:
    candidate = study.build_transition_candidates(_ranked(), study.TRANSITION)
    control = study.build_transition_candidates(_ranked(), study.STILL_NEGLECTED)

    assert candidate["symbol"].to_list() == ["600001.SH", "600000.SH"]
    assert control["symbol"].to_list() == ["600004.SH", "600003.SH"]


def test_gate_requires_return_and_drawdown_improvement() -> None:
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
    mother["metrics"]["max_drawdown"] = -0.29
    assert study.evaluate(candidate, mother, benchmark)["passed"] is False
