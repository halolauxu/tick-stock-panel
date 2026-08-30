from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_holder_trade_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_holder_trade", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(ann_date: str = "20260102") -> dict:
    return {
        "ts_code": "000001.SZ",
        "ann_date": ann_date,
        "holder_name": "测试股东",
        "holder_type": "G",
        "in_de": "IN",
        "change_vol": "10000",
        "change_ratio": "0.12",
        "after_share": "20000",
        "after_ratio": "0.24",
        "avg_price": "10.5",
        "total_share": "20000",
        "begin_date": "20260101",
        "close_date": "20260102",
    }


class _Client:
    def __init__(self, monthly_rows):
        self.monthly_rows = monthly_rows
        self.calls = []

    def query(self, api_name, params, fields):
        self.calls.append(params)
        if "start_date" in params:
            return list(self.monthly_rows)
        return [_row(params["ann_date"])] if params["ann_date"] == "20260102" else []


def test_month_query_is_used_when_below_limit() -> None:
    client = _Client([_row()])

    rows, source = collector.fetch_month(client, 2026, 1)

    assert source == "monthly_range"
    assert len(rows) == 1
    assert len(client.calls) == 1


def test_limit_hit_falls_back_to_every_announcement_day() -> None:
    client = _Client([_row()] * collector.ROW_LIMIT)

    rows, source = collector.fetch_month(client, 2026, 1)

    assert source == "daily_fallback"
    assert len(rows) == 1
    assert len(client.calls) == 32


def test_normalize_converts_schema_and_deduplicates() -> None:
    frame = collector.normalize([_row(), _row()], 2026)

    assert frame.height == 1
    assert frame["ann_date"][0] == date(2026, 1, 2)
    assert frame["direction"][0] == "IN"
    assert frame["change_ratio"][0] == 0.12
