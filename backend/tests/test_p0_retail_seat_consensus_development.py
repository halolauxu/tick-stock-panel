from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_retail_seat_consensus_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_retail_seat", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _seat_rows(event_date: date, symbol: str) -> list[dict]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "trade_date": event_date,
                "symbol": symbol,
                "seat_name": f"营业部{index}",
                "buy": 2_000_000.0,
                "buy_rate": 2.0,
                "sell": 0.0,
                "sell_rate": 0.0,
                "net_buy": 2_000_000.0,
                "reason": "reason-a",
                "side": "buy",
            }
        )
    rows.append({**rows[0], "reason": "duplicate-reason"})
    rows.append(
        {
            **rows[0],
            "seat_name": "机构专用",
            "buy": 50_000_000.0,
            "net_buy": 50_000_000.0,
        }
    )
    return rows


def test_consensus_counts_distinct_retail_seats_and_excludes_institutions() -> None:
    event_date = date(2020, 1, 2)
    details = pl.DataFrame(_seat_rows(event_date, "000001.SZ"))
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [event_date],
            "amount": [200_000_000.0],
            "raw_close": [10.0],
            "limit_up_price": [11.0],
            "excluded_name": [False],
        }
    )

    result = study.aggregate_retail_consensus(details, panel)

    assert result.height == 1
    assert result["positive_seats"][0] == 8
    assert result["retail_net_buy"][0] == 16_000_000.0


def test_consensus_rejects_signal_day_limit_up() -> None:
    event_date = date(2020, 1, 2)
    details = pl.DataFrame(_seat_rows(event_date, "000001.SZ"))
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [event_date],
            "amount": [200_000_000.0],
            "raw_close": [11.0],
            "limit_up_price": [11.0],
            "excluded_name": [False],
        }
    )

    assert study.aggregate_retail_consensus(details, panel).is_empty()


def test_consensus_applies_twenty_calendar_day_cooldown() -> None:
    first = date(2020, 1, 2)
    second = date(2020, 1, 10)
    details = pl.DataFrame(
        _seat_rows(first, "000001.SZ") + _seat_rows(second, "000001.SZ")
    )
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [first, second],
            "amount": [200_000_000.0, 200_000_000.0],
            "raw_close": [10.0, 10.0],
            "limit_up_price": [11.0, 11.0],
            "excluded_name": [False, False],
        }
    )

    result = study.aggregate_retail_consensus(details, panel)

    assert result.height == 1
    assert result["ann_date"][0] == first
