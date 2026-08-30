from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_share_float_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_share_float", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


class FakeClient:
    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key
        self.calls = []

    def query(self, api_name, params, fields):
        self.calls.append((api_name, params, fields))
        key = params.get("float_date") or f"{params['start_date']}:{params['end_date']}"
        return list(self.rows_by_key.get(key, []))


def test_fetch_month_uses_monthly_range_below_limit():
    client = FakeClient(
        {
            "20200101:20200131": [
                {"ts_code": "000001.SZ", "float_date": "20200115"}
            ]
        }
    )

    rows, source = collector.fetch_month(client, 2020, 1)

    assert source == "monthly_range"
    assert len(rows) == 1
    assert len(client.calls) == 1


def test_normalize_converts_units_and_rejects_late_announcements():
    rows = [
        {
            "ts_code": "000001.SZ",
            "ann_date": "20200101",
            "float_date": "20200115",
            "float_share": "12.5",
            "float_ratio": "5.2",
            "holder_name": "甲",
            "share_type": "首发原股东限售股份",
        },
        {
            "ts_code": "000002.SZ",
            "ann_date": "20200201",
            "float_date": "20200115",
            "float_share": "20",
            "float_ratio": "6",
            "holder_name": "乙",
            "share_type": "定向增发机构配售股份",
        },
    ]

    frame = collector.normalize(rows, 2020)

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["symbol"] == "000001.SZ"
    assert row["ann_date"] == date(2020, 1, 1)
    assert row["float_date"] == date(2020, 1, 15)
    assert row["float_shares"] == 125_000.0
    assert row["float_ratio"] == 5.2
