from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_short_horizon_execution_calibration.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_short_horizon_execution_calibration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_uses_auction_open_and_side_aware_adverse_bps() -> None:
    study = _load_module()
    orders = pl.DataFrame(
        {
            "account": ["a", "a"],
            "order_index": [0, 1],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "symbol": ["000001.SZ", "000002.SZ"],
            "side": ["BUY", "SELL"],
            "raw_shares": [1000, None],
            "gross": [10_000.0, 20_000.0],
        }
    )
    auction = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "auction_open": [10.0, 20.0],
            "auction_amount": [2_000_000.0, 2_000_000.0],
        }
    )
    daily = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "daily_raw_open": [10.0, 20.0],
        }
    )
    minute = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "minute_0931_open": [10.1, 19.8],
            "minute_0931_amount": [1_000_000.0, 1_000_000.0],
        }
    )

    matched = study.match_orders(orders, daily, auction, minute)

    assert matched.get_column("auction_abs_bps").to_list() == [0.0, 0.0]
    assert matched.get_column("minute_0931_adverse_bps").to_list() == pytest.approx([100.0, 100.0])
    assert matched.get_column("auction_capacity_ok").to_list() == [True, True]


def test_summary_requires_both_price_sources() -> None:
    study = _load_module()
    frame = pl.DataFrame(
        {
            "account": ["a"],
            "side": ["BUY"],
            "auction_open": [10.0],
            "minute_0931_open": [None],
            "auction_abs_bps": [0.0],
            "auction_adverse_bps": [0.0],
            "minute_0931_abs_bps": [None],
            "minute_0931_adverse_bps": [None],
            "auction_capacity_ok": [True],
            "minute_0931_capacity_ok": [None],
            "date": [date(2026, 1, 5)],
            "symbol": ["000001.SZ"],
            "order_index": [0],
            "raw_shares": [1000],
            "gross": [10_000.0],
            "daily_fill_price": [10.0],
            "auction_amount": [2_000_000.0],
            "minute_0931_amount": [None],
        }
    )

    summary = study.summarize_group(frame)

    assert summary["auction_coverage"] == 1.0
    assert summary["minute_0931_coverage"] == 0.0
    assert summary["minute_0931_abs_bps_median"] is None
