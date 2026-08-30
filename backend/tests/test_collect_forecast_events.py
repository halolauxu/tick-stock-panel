from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_forecast_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_forecast", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(ann_date: str = "20260130"):
    return {
        "ts_code": "000001.SZ",
        "ann_date": ann_date,
        "end_date": "20251231",
        "type": "预增",
        "p_change_min": "50",
        "p_change_max": "80",
        "net_profit_min": "1000",
        "net_profit_max": "1200",
        "last_parent_net": "700",
        "first_ann_date": ann_date,
        "summary": "x",
        "change_reason": "y",
    }


class _VipClient:
    def query(self, api_name, params, fields):
        assert api_name == "forecast_vip"
        return [_row()]


class _FallbackClient:
    def query(self, api_name, params, fields):
        if api_name == "forecast_vip":
            raise RuntimeError("no permission")
        assert api_name == "forecast"
        return [_row()] if params["ann_date"] == "20260130" else []


def test_vip_collection_filters_and_deduplicates_by_announcement_year() -> None:
    rows, source = collector.fetch_year(_VipClient(), 2026)
    frame = collector.normalize(rows, 2026, source)

    assert source == "forecast_vip_by_period"
    assert frame.height == 1
    assert frame["ann_date"][0] == date(2026, 1, 30)
    assert frame["p_change_min"][0] == 50.0


def test_daily_fallback_collects_all_calendar_dates() -> None:
    rows, source = collector.fetch_year(_FallbackClient(), 2026)
    frame = collector.normalize(rows, 2026, source)

    assert source == "forecast_by_announcement_day"
    assert frame.height == 1
    assert frame["symbol"][0] == "000001.SZ"


def test_collection_run_is_bounded_to_two_years(tmp_path: Path) -> None:
    try:
        collector.run(tmp_path, 2020, 2022)
    except ValueError as error:
        assert "at most two years" in str(error)
    else:
        raise AssertionError("expected bounded collection error")
