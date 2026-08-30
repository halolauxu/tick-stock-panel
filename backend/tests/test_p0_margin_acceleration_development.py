from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_margin_acceleration_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_margin", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _margin(symbol: str, trade_date: date, balance_change: float) -> dict:
    previous = 100_000_000.0
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "rzye": previous * (1.0 + balance_change),
        "rqye": 0.0,
        "rzmre": 20_000_000.0,
        "rqyl": 0.0,
        "rzche": 0.0,
        "rqchl": 0.0,
        "rqmcl": 0.0,
        "rzrqye": previous * (1.0 + balance_change),
        "previous_rzye": previous,
        "margin_balance_change": balance_change,
        "margin_dates_adjacent": True,
    }


def test_categorize_margin_divergence_continuation_and_control():
    event_date = date(2020, 1, 2)
    margin = pl.DataFrame(
        [
            _margin("000001.SZ", event_date, 0.10),
            _margin("000002.SZ", event_date, 0.10),
            _margin("000003.SZ", event_date, -0.10),
        ]
    )
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": [event_date, event_date, event_date],
            "event_daily_amount": [100_000_000.0] * 3,
            "event_return": [-0.02, 0.02, -0.02],
        }
    )

    events = study.categorize_events(margin, panel)

    assert events.select("symbol", "category").to_dicts() == [
        {"symbol": "000001.SZ", "category": "margin_price_divergence"},
        {"symbol": "000002.SZ", "category": "margin_price_continuation"},
        {"symbol": "000003.SZ", "category": "deleverage_control"},
    ]


def test_categorize_uses_global_symbol_cooldown():
    first = date(2020, 1, 2)
    second = first + timedelta(days=10)
    margin = pl.DataFrame(
        [
            _margin("000001.SZ", first, 0.10),
            _margin("000001.SZ", second, -0.10),
        ]
    )
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": [first, second],
            "event_daily_amount": [100_000_000.0, 100_000_000.0],
            "event_return": [-0.02, -0.02],
        }
    )

    events = study.categorize_events(margin, panel)

    assert events.height == 1
    assert events.row(0, named=True)["ann_date"] == first
