from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "research" / "collect_p0_etf_premium_reversal_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_etf_premium_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_normalize_nav_keeps_earliest_point_in_time_version() -> None:
    rows = [
        {
            "ts_code": "510050.SH",
            "ann_date": "20200104",
            "nav_date": "20200103",
            "unit_nav": "3.1",
            "accum_nav": "4.0",
            "adj_nav": "4.2",
        },
        {
            "ts_code": "510050.SH",
            "ann_date": "20200105",
            "nav_date": "20200103",
            "unit_nav": "3.2",
            "accum_nav": "4.1",
            "adj_nav": "4.3",
        },
        {
            "ts_code": "OTHER.SH",
            "ann_date": "20200104",
            "nav_date": "20200103",
            "unit_nav": "9.9",
            "accum_nav": "9.9",
            "adj_nav": "9.9",
        },
    ]

    result = study.normalize_nav(rows, "510050.SH")

    assert result.height == 1
    assert result["ann_date"][0] == date(2020, 1, 4)
    assert result["unit_nav"][0] == 3.1


def test_audit_rejects_future_announcements_and_bad_units() -> None:
    master = pl.DataFrame({"symbol": ["A.SH"], "fund_type": ["股票型"]})
    daily = pl.DataFrame({"symbol": ["A.SH"], "date": [date(2020, 1, 2)]})
    nav = pl.DataFrame(
        {
            "symbol": ["A.SH"],
            "ann_date": [date(2020, 1, 1)],
            "nav_date": [date(2020, 1, 2)],
            "unit_nav": [0.0],
            "accum_nav": [1.0],
            "adj_nav": [1.0],
        }
    )

    result = study.audit(master, daily, nav)

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["announcement_not_before_nav"] is False
    assert result["checks"]["unit_nav_positive"] is False


def test_audit_checks_exact_nav_market_dates_not_nearest_dates() -> None:
    master = pl.DataFrame({"symbol": ["A.SH"], "fund_type": ["股票型"]})
    daily = pl.DataFrame({"symbol": ["A.SH"], "date": [date(2020, 1, 3)]})
    nav = pl.DataFrame(
        {
            "symbol": ["A.SH"],
            "ann_date": [date(2020, 1, 3)],
            "nav_date": [date(2020, 1, 2)],
            "unit_nav": [1.0],
            "accum_nav": [1.0],
            "adj_nav": [1.0],
        }
    )

    result = study.audit(master, daily, nav)

    assert result["counts"]["exact_market_date_matches"] == 0
    assert result["checks"]["exact_market_date_match_at_least_90pct"] is False
