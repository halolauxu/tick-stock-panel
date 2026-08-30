from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_opening_auction_imbalance_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_auction", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _context(symbol: str, event_date: date) -> dict:
    return {
        "symbol": symbol,
        "date": event_date,
        "trade_index": 10,
        "raw_open": 10.2,
        "raw_high": 10.3,
        "raw_low": 10.1,
        "raw_close": 10.2,
        "open": 10.2,
        "amount": 100_000_000.0,
        "volume": 10_000_000.0,
        "excluded_name": False,
        "limit_up_price": 11.0,
        "limit_down_price": 9.0,
        "reference_close": 10.0,
        "previous_amount": 100_000_000.0,
    }


def test_categorize_events_uses_active_auction_and_global_cooldown():
    first = date(2025, 9, 1)
    second = first + timedelta(days=10)
    auction = pl.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "date": first,
                "session": "open",
                "open": 10.1,
                "high": 10.2,
                "low": 10.1,
                "close": 10.2,
                "volume_shares": 500_000.0,
                "amount": 4_000_000.0,
                "vwap": 10.15,
            },
            {
                "symbol": "000001.SZ",
                "date": second,
                "session": "open",
                "open": 9.7,
                "high": 9.8,
                "low": 9.7,
                "close": 9.8,
                "volume_shares": 500_000.0,
                "amount": 4_000_000.0,
                "vwap": 9.75,
            },
        ]
    )
    context = pl.DataFrame([_context("000001.SZ", first), _context("000001.SZ", second)])

    events = study.categorize_events(auction, context)

    assert events.height == 1
    assert events.row(0, named=True)["category"] == "demand_continuation"


def test_categorize_events_rejects_middle_activity_not_in_control():
    event_date = date(2025, 9, 1)
    auction = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [event_date],
            "session": ["open"],
            "open": [10.1],
            "high": [10.2],
            "low": [10.1],
            "close": [10.2],
            "volume_shares": [100_000.0],
            "amount": [700_000.0],
            "vwap": [10.15],
        }
    )
    context = pl.DataFrame([_context("000001.SZ", event_date)])

    events = study.categorize_events(auction, context)

    assert events.is_empty()


def test_cluster_t_aggregates_same_day_events():
    frame = pl.DataFrame(
        {
            "ann_date": [date(2025, 9, 1), date(2025, 9, 1), date(2025, 9, 2)],
            "excess_return": [0.01, 0.03, 0.02],
        }
    )

    value = study._cluster_t(frame, "excess_return")

    assert value is None


def test_build_trades_enforces_next_trading_day_exit():
    entry_date = date(2025, 9, 1)
    exit_date = date(2025, 9, 2)
    events = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "ann_date": [entry_date],
            "trade_index": [10],
            "category": ["demand_continuation"],
            "universe_eligible": [True],
        }
    )
    minute = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [entry_date, exit_date],
            "open": [10.2, 10.4],
            "high": [10.3, 10.5],
            "low": [10.1, 10.3],
            "close": [10.25, 10.45],
            "volume": [1_000_000.0, 1_000_000.0],
            "amount": [10_000_000.0, 10_000_000.0],
        }
    )
    context = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [entry_date, exit_date],
            "trade_index": [10, 11],
            "limit_up_price": [11.0, 11.2],
            "limit_down_price": [9.0, 9.2],
            "excluded_name": [False, False],
        }
    )

    trades = study.build_trades(events, minute, context)

    row = trades.row(0, named=True)
    assert row["tradable"] is True
    assert row["exit_delay"] == 0
    assert row["actual_exit_date"] == exit_date
    assert row["net_return"] > 0
