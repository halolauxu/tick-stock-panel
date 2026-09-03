from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_microcap_idiosyncratic_forecast_priority_screen as study  # noqa: E402


def test_combine_curves_sweeps_only_event_cash_to_core() -> None:
    event = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "equity": [200_000.0, 202_000.0, 204_020.0],
            "cash": [200_000.0, 100_000.0, 0.0],
            "position_value": [0.0, 102_000.0, 204_020.0],
            "position_count": [0, 1, 2],
        }
    )
    core = [
        {"date": "2024-01-02", "equity": 200_000.0},
        {"date": "2024-01-03", "equity": 204_000.0},
        {"date": "2024-01-04", "equity": 208_080.0},
    ]

    curve, combined, core_summary = study.combine_curves(event, core)

    # Day one: event +1%, half of the account cash earns the core's +2%.
    assert curve[1]["daily_return"] == pytest.approx(0.02)
    # Day two: event is fully invested, so no core return is double counted.
    assert curve[2]["daily_return"] == pytest.approx(0.01)
    assert combined["active_event_days"] == 2
    assert combined["mean_event_invested_weight"] == pytest.approx(0.75)
    assert core_summary["total_return"] == pytest.approx(0.0404)


def test_evaluate_requires_stress_improvement_and_event_integrity() -> None:
    event = {
        "intent_execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.96},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
        "account": {"trade_count": 120},
    }
    results = {
        "development": {"combined": {"annualized": 0.2}},
        "validation": {
            "combined": {
                "annualized": 0.25,
                "yearly": [
                    {"year": 2021, "return": 0.1},
                    {"year": 2022, "return": 0.1},
                    {"year": 2023, "return": 0.1},
                ],
            }
        },
        "known_stress": {
            "combined": {"annualized": 0.25, "max_drawdown": -0.35},
            "core": {"annualized": 0.20, "max_drawdown": -0.45},
            "event": event,
        },
    }

    decision = study.evaluate(results)

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_UNIFIED_ACCOUNT"
