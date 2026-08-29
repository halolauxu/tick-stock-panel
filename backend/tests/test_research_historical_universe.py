from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "research_backfill_tushare_historical_universe.py"


def _module():
    spec = importlib.util.spec_from_file_location("historical_universe_backfill", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daily_normalization_matches_panel_units():
    module = _module()
    result = module._normalize_daily(
        [
            {
                "ts_code": "600001.SH",
                "trade_date": "20200102",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "vol": 123.4,
                "amount": 456.7,
            }
        ],
        date(2020, 1, 1),
        date(2020, 1, 31),
    )

    assert result["volume"][0] == 123.4
    assert result["amount"][0] == 456_700.0
    assert result["date"][0] == date(2020, 1, 2)


def test_share_normalization_keeps_only_change_points():
    module = _module()
    rows = [
        {
            "ts_code": "600001.SH",
            "trade_date": day,
            "total_share": total,
            "float_share": floating,
        }
        for day, total, floating in (
            ("20200102", 100, 80),
            ("20200103", 100, 80),
            ("20200106", 110, 85),
        )
    ]

    result = module._normalize_shares(rows, date(2020, 1, 1), date(2020, 1, 31))

    assert result["period_end"].to_list() == ["2020-01-02", "2020-01-06"]
    assert result["total_shares"].to_list() == [1_000_000.0, 1_100_000.0]
    assert result["float_shares"].to_list() == [800_000.0, 850_000.0]


def test_instrument_frame_uses_last_point_in_time_share_record():
    module = _module()
    universe = pl.DataFrame(
        {
            "symbol": ["600001.SH"],
            "name": ["测试退"],
            "market": ["主板"],
            "exchange": ["SSE"],
            "list_status": ["D"],
            "list_date": [date(2000, 1, 1)],
            "delist_date": [date(2020, 1, 31)],
        }
    )
    shares = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH"],
            "period_end": ["2020-01-02", "2020-01-06"],
            "announce_date": ["2020-01-02", "2020-01-06"],
            "total_shares": [1_000_000.0, 1_100_000.0],
            "float_shares": [800_000.0, 850_000.0],
        }
    )

    result = module._instrument_frame(universe, shares, date(2026, 8, 27))

    assert result["code"][0] == "600001"
    assert result["exchange"][0] == "SH"
    assert result["total_shares"][0] == 1_100_000.0
    assert result["float_shares"][0] == 850_000.0


def test_a_share_boards_and_listing_overlap_are_explicit():
    module = _module()

    assert module._is_a_share_equity("600001.SH")
    assert module._is_a_share_equity("002001.SZ")
    assert module._is_a_share_equity("300001.SZ")
    assert module._is_a_share_equity("688001.SH")
    assert module._is_a_share_equity("920001.BJ")
    assert not module._is_a_share_equity("000001.HK")
    assert module._overlaps(
        {"list_date": "20100101", "delist_date": "20150101"},
        date(2013, 1, 1),
        date(2020, 1, 1),
    )
    assert not module._overlaps(
        {"list_date": "20100101", "delist_date": "20120101"},
        date(2013, 1, 1),
        date(2020, 1, 1),
    )


def test_name_history_adds_safe_interval_for_stock_without_renaming():
    module = _module()
    universe = pl.DataFrame({
        "symbol": ["600001.SH", "300001.SZ"],
        "name": ["历史更名", "从未更名"],
        "list_date": [date(2000, 1, 1), date(2010, 1, 1)],
        "delist_date": [None, None],
    })
    names = pl.DataFrame({
        "symbol": ["600001.SH"],
        "name": ["历史更名"],
        "start_date": [date(2000, 1, 1)],
        "end_date": [None],
        "announce_date": [None],
        "change_reason": ["更名"],
    })

    result = module._complete_name_history(universe, names)

    assert set(result["symbol"].to_list()) == {"600001.SH", "300001.SZ"}
    baseline = result.filter(pl.col("symbol") == "300001.SZ").row(0, named=True)
    assert baseline["start_date"] == date(2010, 1, 1)
    assert baseline["change_reason"] == "stock_basic未发生更名"
