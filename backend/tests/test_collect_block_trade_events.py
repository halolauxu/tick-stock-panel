from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_block_trade_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_block_trade", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(trade_date: str = "20260130") -> dict:
    return {
        "ts_code": "000001.SZ",
        "trade_date": trade_date,
        "price": "10.5",
        "vol": "12.5",
        "amount": "131.25",
        "buyer": "买方席位",
        "seller": "卖方席位",
    }


class _Client:
    def __init__(self, monthly_rows):
        self.monthly_rows = monthly_rows
        self.calls = []

    def query(self, api_name, params, fields):
        self.calls.append(params)
        if "start_date" in params:
            return list(self.monthly_rows)
        return [_row(params["trade_date"])] if params["trade_date"] == "20260130" else []


def test_month_query_is_used_below_limit() -> None:
    client = _Client([_row()])

    rows, source = collector.fetch_month(client, 2026, 1)

    assert source == "monthly_range"
    assert len(rows) == 1
    assert len(client.calls) == 1


def test_limit_hit_falls_back_to_each_calendar_day() -> None:
    client = _Client([_row()] * collector.ROW_LIMIT)

    rows, source = collector.fetch_month(client, 2026, 1)

    assert source == "daily_fallback"
    assert len(rows) == 1
    assert len(client.calls) == 32


def test_normalize_converts_documented_ten_thousand_units() -> None:
    frame = collector.normalize([_row(), _row()], 2026)

    assert frame.height == 1
    assert frame["trade_date"][0] == date(2026, 1, 30)
    assert frame["volume_shares"][0] == 125_000.0
    assert frame["notional_cny"][0] == 1_312_500.0
