from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import fixed_horizon_account as fixed  # noqa: E402


def test_fixed_horizon_exits_after_exact_trading_days() -> None:
    dates = [date(2020, 1, 2) + timedelta(days=index) for index in range(30)]
    candidates = pl.DataFrame(
        {
            "date": [dates[0]],
            "entry_date": [dates[1]],
            "symbol": ["600000.SH"],
            "signal_amount": [100_000_000.0],
            "cap_rank": [1],
        }
    )
    quotes = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * len(dates),
            "date": dates,
            "open": [10.0] * len(dates),
            "raw_open": [10.0] * len(dates),
            "close": [10.0] * len(dates),
            "volume": [1_000_000.0] * len(dates),
            "amount": [100_000_000.0] * len(dates),
            "limit_up_price": [11.0] * len(dates),
            "limit_down_price": [9.0] * len(dates),
            "is_excluded_name": [False] * len(dates),
        }
    )

    result = fixed.simulate(
        candidates,
        quotes,
        dates,
        initial_cash=200_000.0,
        target_positions=10,
        holding_trading_days=5,
        maximum_exit_delay=20,
        period_start=dates[0],
        period_end=dates[-1],
    )

    assert result["integrity"]["holding_days_distribution"] == {5: 1}
    assert result["integrity"]["ending_unresolved_positions"] == 0
    assert result["account"]["trade_count"] == 2


def test_entry_cutoff_leaves_full_exit_delay_window() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(30)]
    candidates = pl.DataFrame(
        {
            "date": [dates[2], dates[5]],
            "entry_date": [dates[3], dates[6]],
            "symbol": ["600000.SH", "600001.SH"],
            "signal_amount": [1.0, 1.0],
            "cap_rank": [1, 1],
        }
    )

    result = fixed.prepare_candidates(
        candidates,
        dates,
        holding_trading_days=5,
        maximum_exit_delay=20,
    )

    assert result["symbol"].to_list() == ["600000.SH"]
