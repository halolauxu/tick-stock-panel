from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_preclose_near_limit_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_preclose_limit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _frames(pre_signal_high: float = 10.90):
    signal_day = date(2026, 1, 5)
    exit_day = date(2026, 1, 6)
    late = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [signal_day],
            "late_open": [10.60],
            "signal_close": [10.85],
            "late_amount": [25_000_000.0],
            "late_bars": [15],
            "late_return": [10.85 / 10.60 - 1.0],
            "pre_signal_high": [pre_signal_high],
        }
    )
    entry = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [signal_day],
            "entry_open": [10.80],
            "entry_amount": [3_000_000.0],
        }
    )
    exit_frame = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [exit_day],
            "exit_open": [11.00],
            "exit_amount": [3_000_000.0],
        }
    )
    context = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [signal_day, exit_day],
            "trade_index": [0, 1],
            "excluded_name": [False, False],
            "limit_up_price": [11.0, 11.9],
            "limit_down_price": [9.0, 9.7],
            "adjustment_factor": [1.0, 1.0],
            "previous_amount": [150_000_000.0, 150_000_000.0],
            "previous_raw_close": [10.0, 10.8],
        }
    )
    return late, entry, exit_frame, context


def test_near_limit_signal_is_bought_after_signal_and_sold_next_day() -> None:
    observations, _ = study.build_observations(*_frames())

    assert observations.height == 1
    assert observations["tradable"][0]
    assert observations["capacity_cny"][0] == 30_000.0
    assert observations["net_return"][0] > 0


def test_near_limit_signal_rejects_stock_that_already_touched_limit() -> None:
    observations, _ = study.build_observations(*_frames(pre_signal_high=11.0))

    assert observations.is_empty()


def test_evaluation_requires_both_halves_and_confirmation_depth() -> None:
    market_dates = [date(2025, 1, 2) + timedelta(days=2 * i) for i in range(160)]
    rows = []
    for day in market_dates:
        for event in range(7):
            rows.append(
                {
                    "date": day,
                    "tradable": True,
                    "net_return": 0.01,
                    "excess_return": 0.008,
                    "capacity_cny": 200_000.0,
                    "symbol": f"000{event:03d}.SZ",
                }
            )

    result = study.evaluate(pl.DataFrame(rows), market_dates)

    assert result["promotion_passed"] is True
