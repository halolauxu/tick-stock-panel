from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_microcap_idiosyncratic_forecast_unified_account as unified  # noqa: E402
import run_p1_microcap_participation_allocator as allocator  # noqa: E402


def _feature(day: date, score: int, *, ready: bool = True) -> dict:
    value = 1.0 if ready else None
    return {
        "date": day,
        "participation_score": score,
        "microcap_absolute_20d": value,
        "microcap_relative_20d": value,
        "microcap_breadth_20d": value,
        "microcap_liquidity_20d_60d": value,
        "severe_limit_down": False,
    }


def test_allocator_is_fail_closed_and_delays_only_exposure_upgrades() -> None:
    days = [date(2026, 9, day) for day in range(1, 8)]
    features = pl.DataFrame(
        [
            _feature(days[0], 4, ready=False),
            _feature(days[1], 4),
            _feature(days[2], 4),
            _feature(days[3], 4),
            _feature(days[4], 2),
            _feature(days[5], 1),
            _feature(days[6], 4),
        ]
    )

    slots, decisions = allocator.build_allocation_clock(features)

    assert slots[days[1]] == 0
    assert slots[days[3]] == 0
    assert slots[days[4]] == 20
    assert slots[days[5]] == 10
    assert slots[days[6]] == 0
    assert decisions[0]["raw_microcap_slots"] == 0


def test_target_builder_respects_microcap_budget_without_diluting_events() -> None:
    day = date(2026, 9, 8)
    micro = pl.DataFrame(
        {
            "date": [day] * 20,
            "entry_date": [day] * 20,
            "symbol": [f"600{index:03d}.SH" for index in range(20)],
            "signal_amount": [100_000_000.0] * 20,
            "cap_rank": list(range(1, 21)),
        }
    )
    events = pl.DataFrame(
        {
            "date": [day],
            "entry_date": [day],
            "symbol": ["000001.SZ"],
            "signal_amount": [100_000_000.0],
            "cap_rank": [1],
        }
    )

    targets = unified.build_daily_targets(
        micro,
        events,
        [day],
        microcap_slots_by_date={day: 10},
    )

    assert targets.filter(pl.col("family") == unified.EVENT_FAMILY).height == 1
    assert targets.filter(pl.col("family") == unified.MICROCAP_FAMILY).height == 10
    assert targets.get_column("target_weight").sum() == pytest.approx(0.70)


def test_target_builder_rejects_invalid_slot_budget() -> None:
    day = date(2026, 9, 8)
    micro = pl.DataFrame(
        {
            "date": [day],
            "entry_date": [day],
            "symbol": ["600000.SH"],
            "signal_amount": [100_000_000.0],
            "cap_rank": [1],
        }
    )
    events = micro.head(0)

    try:
        unified.build_daily_targets(
            micro,
            events,
            [day],
            microcap_slots_by_date={day: 21},
        )
    except ValueError as exc:
        assert "invalid micro-cap slot budget" in str(exc)
    else:
        raise AssertionError("invalid slot budget should fail closed")


def test_zero_slot_action_date_can_still_sell_at_an_exact_quote() -> None:
    buy_day = date(2026, 9, 7)
    cash_day = date(2026, 9, 8)
    candidates = pl.DataFrame(
        {
            "date": [buy_day],
            "entry_date": [buy_day],
            "symbol": ["600000.SH"],
            "signal_amount": [100_000_000.0],
            "cap_rank": [1],
            "target_weight": [0.05],
            "family": [unified.MICROCAP_FAMILY],
        }
    )
    quotes = pl.DataFrame(
        {
            "date": [buy_day, cash_day],
            "symbol": ["600000.SH", "600000.SH"],
            "amount": [100_000_000.0, 100_000_000.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "raw_open": [10.0, 10.2],
            "open": [10.0, 10.2],
            "close": [10.1, 10.2],
            "is_excluded_name": [False, False],
            "limit_up_price": [11.0, 11.11],
            "limit_down_price": [9.0, 9.09],
        }
    )
    execution_seed = candidates.select("symbol").unique().join(
        pl.DataFrame({"entry_date": [buy_day, cash_day]}),
        how="cross",
    )
    grid = unified.account.build_execution_grid(execution_seed, quotes)

    result = unified.account.simulate_account(
        candidates,
        grid,
        initial_cash=200_000.0,
        target_positions=20,
        action_dates=[buy_day, cash_day],
        candidate_weight_column="target_weight",
    )

    sells = [row for row in result["orders"] if row["side"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["status"] == "FILLED"
