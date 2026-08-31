from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "research" / "run_p0_early_first_board_upper_bound.py"
    )
    spec = importlib.util.spec_from_file_location("p0_early_first_board", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _row(hour: int, minute: int, *, high: float, close: float) -> dict:
    return {
        "datetime": datetime(2026, 1, 5, hour, minute),
        "open": close,
        "high": high,
        "low": close,
        "close": close,
        "volume": 10_000.0,
        "amount": 10_000_000.0,
    }


def test_detects_first_post_open_seal() -> None:
    rows = [
        _row(9, 30, high=10.80, close=10.75),
        _row(9, 31, high=10.95, close=10.90),
        _row(9, 32, high=11.00, close=11.00),
    ]

    result = study.detect_early_first_board(rows, 11.0)

    assert result is not None
    assert result["signal_datetime"] == datetime(2026, 1, 5, 9, 32)
    assert result["entry_price"] == 11.0


def test_rejects_open_locked_board() -> None:
    rows = [
        _row(9, 30, high=11.00, close=11.00),
        _row(9, 31, high=11.00, close=11.00),
    ]

    assert study.detect_early_first_board(rows, 11.0) is None


def test_rejects_prior_touch_before_later_seal() -> None:
    rows = [
        _row(9, 30, high=10.80, close=10.75),
        _row(9, 31, high=11.00, close=10.90),
        _row(9, 32, high=11.00, close=11.00),
    ]

    assert study.detect_early_first_board(rows, 11.0) is None


def test_rejects_signal_after_cutoff() -> None:
    rows = [
        _row(9, 30, high=10.80, close=10.75),
        _row(10, 31, high=11.00, close=11.00),
    ]

    assert study.detect_early_first_board(rows, 11.0) is None


def test_commission_respects_five_yuan_minimum() -> None:
    assert study.commission(10_000.0) == 5.0
    assert study.commission(100_000.0) == 20.0


def test_universe_eligibility_uses_retained_market_panel_columns() -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "amount": [60_000_000.0, 70_000_000.0, 60_000_000.0, 70_000_000.0],
            "close": [10.0, 10.5, 10.0, 10.5],
            "raw_close": [10.0, 10.5, 10.0, 10.5],
            "daily_return": [None, 0.05, None, 0.05],
            "_is_excluded": [False, False, False, False],
            "_promotion_pool": [False, False, False, True],
            "limit_up_price": [11.0, 11.55, 11.0, 11.55],
        }
    )

    result = study.add_universe_eligibility(panel)

    assert result.get_column("universe_eligible").to_list() == [
        False,
        True,
        False,
        False,
    ]


def test_account_uses_next_day_window_and_finishes_flat() -> None:
    first = date(2025, 8, 27)
    second = date(2025, 8, 28)
    context = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [first, second],
            "_global_index": [1, 2],
            "close": [11.0, 11.5],
            "adj_factor": [1.0, 1.0],
            "limit_down_price": [9.0, 9.9],
        }
    )
    windows = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [second],
            "window_amount": [20_000_000.0],
            "window_volume": [17_391.304347826088],
            "window_high": [11.6],
            "window_minutes": [5],
            "window_vwap": [11.5],
        }
    )
    events = [
        {
            "symbol": "000001.SZ",
            "date": first,
            "entry_price": 11.0,
            "signal_amount": 20_000_000.0,
            "adj_factor": 1.0,
            "adjusted_close": 11.0,
        }
    ]

    result = study.simulate_account(200_000.0, events, context, windows)

    assert result["entry_fills"] == 1
    assert result["completed_trades"] == 1
    assert result["open_positions"] == 0
    assert result["final_equity"] > 200_000.0
    assert result["gross_realized_pnl_before_costs"] == 2_250.0
    assert result["total_slippage"] > 0
    assert result["max_exit_delay"] == 0


def test_account_does_not_fill_after_twenty_day_exit_limit() -> None:
    first = date(2025, 8, 27)
    too_late = date(2025, 9, 29)
    context = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [first, too_late],
            "_global_index": [1, 23],
            "close": [11.0, 11.5],
            "adj_factor": [1.0, 1.0],
            "limit_down_price": [9.0, 9.9],
        }
    )
    windows = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [too_late],
            "window_amount": [20_000_000.0],
            "window_volume": [17_391.304347826088],
            "window_high": [11.6],
            "window_minutes": [5],
            "window_vwap": [11.5],
        }
    )
    events = [
        {
            "symbol": "000001.SZ",
            "date": first,
            "entry_price": 11.0,
            "signal_amount": 20_000_000.0,
            "adj_factor": 1.0,
            "adjusted_close": 11.0,
        }
    ]

    result = study.simulate_account(200_000.0, events, context, windows)

    assert result["completed_trades"] == 0
    assert result["open_positions"] == 1
