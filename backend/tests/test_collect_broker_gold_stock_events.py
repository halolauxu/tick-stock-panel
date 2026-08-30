from __future__ import annotations

import importlib.util
import stat
from datetime import date
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_broker_gold_stock_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_broker_gold_stock_events", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(broker: str = "甲证券", symbol: str = "000001.SZ") -> dict:
    return {
        "month": "202107",
        "broker": broker,
        "ts_code": symbol,
        "name": "平安银行",
    }


def test_period_is_bounded_to_observed_history() -> None:
    collector.validate_period(2020, 7)
    collector.validate_period(2026, 8)
    with pytest.raises(ValueError, match=r"2020-07\.\.2026-08"):
        collector.validate_period(2020, 6)


def test_fetch_month_uses_metadata_only() -> None:
    calls = []

    def fetch(api_name, params, fields):
        calls.append((api_name, params, fields))
        return [_row()]

    rows = collector.fetch_month(fetch, 2021, 7)

    assert len(rows) == 1
    assert calls == [
        (
            "broker_recommend",
            {"month": "202107"},
            ("month", "broker", "ts_code", "name"),
        )
    ]


def test_normalize_deduplicates_and_defers_availability_until_day_three() -> None:
    frame = collector.normalize([_row(), _row()], 2021, 7)

    assert frame.height == 1
    assert frame["recommendation_month"][0] == date(2021, 7, 1)
    assert frame["available_after"][0] == date(2021, 7, 3)
    assert frame["broker"][0] == "甲证券"
    assert frame["symbol"][0] == "000001.SZ"


def test_collect_month_persists_empty_partition_atomically(tmp_path) -> None:
    def fetch(_api_name, _params, _fields):
        return []

    result = collector.collect_month(fetch, tmp_path, 2020, 7)
    path = Path(result["path"])
    frame = collector.pl.read_parquet(path)

    assert result["events"] == result["brokers"] == result["symbols"] == 0
    assert frame.schema == collector.EVENT_SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
