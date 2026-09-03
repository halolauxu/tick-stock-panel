from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_microcap_idiosyncratic_forecast_priority_screen as study  # noqa: E402
import run_p0_microcap_idiosyncratic_forecast_unified_account as unified  # noqa: E402


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


def test_daily_targets_replace_four_microcap_slots_per_event() -> None:
    start = date(2024, 1, 2)
    next_day = date(2024, 1, 3)
    micro = pl.DataFrame(
        {
            "date": [start] * 20,
            "entry_date": [start] * 20,
            "symbol": [f"600{index:03d}.SH" for index in range(20)],
            "signal_amount": [100_000_000.0] * 20,
            "cap_rank": list(range(1, 21)),
        }
    )
    events = pl.DataFrame(
        {
            "date": [start],
            "entry_date": [next_day],
            "symbol": ["000001.SZ"],
            "signal_amount": [100_000_000.0],
            "cap_rank": [1],
        }
    )

    targets = unified.build_daily_targets(micro, events, [start, next_day])
    day_one = targets.filter(pl.col("entry_date") == start)
    day_two = targets.filter(pl.col("entry_date") == next_day)

    assert day_one.height == 20
    assert day_one.get_column("target_weight").sum() == pytest.approx(1.0)
    assert day_two.height == 17
    assert day_two.filter(pl.col("family") == unified.EVENT_FAMILY).height == 1
    assert day_two.filter(pl.col("family") == unified.MICROCAP_FAMILY).height == 16
    assert day_two.get_column("target_weight").sum() == pytest.approx(1.0)


def test_weighted_account_uses_candidate_target_weight() -> None:
    day = date(2024, 1, 2)
    candidates = pl.DataFrame(
        {
            "date": [day, day],
            "entry_date": [day, day],
            "symbol": ["000001.SZ", "600000.SH"],
            "signal_amount": [100_000_000.0, 100_000_000.0],
            "cap_rank": [1, 2],
            "target_weight": [0.2, 0.8],
            "family": [unified.EVENT_FAMILY, unified.MICROCAP_FAMILY],
        }
    )
    grid = pl.DataFrame(
        {
            "entry_date": [day, day],
            "symbol": ["000001.SZ", "600000.SH"],
            "exact_quote": [True, True],
            "is_excluded_name": [False, False],
            "entry_volume": [1_000_000.0, 1_000_000.0],
            "entry_amount": [100_000_000.0, 100_000_000.0],
            "raw_open": [10.0, 10.0],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "limit_up_price": [11.0, 11.0],
            "limit_down_price": [9.0, 9.0],
        }
    )

    result = unified.account.simulate_account(
        candidates,
        grid,
        initial_cash=200_000.0,
        target_positions=2,
        action_dates=[day],
        candidate_weight_column="target_weight",
    )
    buys = [row for row in result["orders"] if row["side"] == "BUY"]

    assert buys[0]["target_notional"] == pytest.approx(40_000.0)
    assert buys[0]["family"] == unified.EVENT_FAMILY
    assert buys[1]["target_notional"] == pytest.approx(160_000.0)
