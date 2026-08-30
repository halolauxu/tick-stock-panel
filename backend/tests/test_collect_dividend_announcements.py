from __future__ import annotations

import importlib.util
import stat
from datetime import date
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_dividend_announcements.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_dividend_announcements", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(symbol: str = "000001.SZ") -> dict:
    return {
        "ts_code": symbol,
        "end_date": "20191231",
        "ann_date": "20200115",
        "div_proc": "预案",
        "stk_div": 0,
        "stk_bo_rate": 0,
        "stk_co_rate": 0,
        "cash_div": 0.2,
        "cash_div_tax": 0.2,
    }


def test_period_is_bounded_to_frozen_metadata_years() -> None:
    collector.validate_period(2012, 1)
    collector.validate_period(2020, 12)
    with pytest.raises(ValueError, match="2012-2020"):
        collector.validate_period(2021, 1)


def test_fetch_month_queries_each_calendar_day_without_price_fields() -> None:
    calls = []

    def fetch(api_name, params, fields):
        calls.append((api_name, params, fields))
        return [_row()] if params["ann_date"] == "20200115" else []

    rows, maximum_daily_rows = collector.fetch_month(fetch, 2020, 1)

    assert len(calls) == 31
    assert len(rows) == maximum_daily_rows == 1
    assert all(call[0] == "dividend" for call in calls)
    assert all(set(call[1]) == {"ann_date"} for call in calls)
    assert "ex_date" not in calls[0][2]
    assert "pay_date" not in calls[0][2]


def test_normalize_is_unique_and_uses_announcement_date() -> None:
    frame = collector.normalize([_row(), _row()], 2020, 1)

    assert frame.height == 1
    assert frame["symbol"][0] == "000001.SZ"
    assert frame["period_end"][0] == date(2019, 12, 31)
    assert frame["ann_date"][0] == date(2020, 1, 15)
    assert frame["cash_dividend_pre_tax_per_share"][0] == pytest.approx(0.2)


def test_collect_month_persists_empty_partition_atomically(tmp_path) -> None:
    def fetch(_api_name, _params, _fields):
        return []

    result = collector.collect_month(fetch, tmp_path, 2012, 1)
    path = Path(result["path"])
    frame = collector.pl.read_parquet(path)

    assert result["events"] == result["symbols"] == 0
    assert frame.schema == collector.EVENT_SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
