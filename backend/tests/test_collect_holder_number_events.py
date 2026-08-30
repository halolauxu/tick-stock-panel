from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "collect_holder_number_events.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_holder_number", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def test_normalize_preserves_point_in_time_keys_and_rejects_invalid_counts():
    rows = [
        {
            "ts_code": "000001.SZ",
            "ann_date": "20200131",
            "end_date": "20191231",
            "holder_num": 10000,
        },
        {
            "ts_code": "000001.SZ",
            "ann_date": "20200131",
            "end_date": "20191231",
            "holder_num": 10000,
        },
        {
            "ts_code": "000002.SZ",
            "ann_date": "20200131",
            "end_date": "20191231",
            "holder_num": None,
        },
    ]

    frame, invalid = collector.normalize(rows, 2020)

    assert frame.height == 1
    assert invalid == 1
    assert frame.row(0, named=True)["symbol"] == "000001.SZ"


class _Client:
    def __init__(self):
        self.calls = 0

    def query(self, api_name, params, fields):
        self.calls += 1
        if "start_date" in params:
            return [{}] * collector.DOCUMENTED_ROW_LIMIT
        return []


def test_month_query_falls_back_when_documented_limit_is_reached():
    client = _Client()

    rows, source = collector.fetch_month(client, 2020, 2)

    assert rows == []
    assert source == "daily_fallback"
    assert client.calls == 30
