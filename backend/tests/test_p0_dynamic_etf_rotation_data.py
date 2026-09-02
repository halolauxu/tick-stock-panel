from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "research" / "collect_p0_dynamic_etf_rotation_data.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_etf_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def test_master_keeps_delisted_etfs_and_rejects_future_listings() -> None:
    rows = [
        {
            "ts_code": "510001.SH",
            "name": "存续ETF",
            "fund_type": "股票型",
            "found_date": "20120101",
            "list_date": "20130101",
            "delist_date": None,
            "market": "E",
        },
        {
            "ts_code": "510002.SH",
            "name": "退市ETF",
            "fund_type": "股票型",
            "found_date": "20120101",
            "list_date": "20130101",
            "delist_date": "20180101",
            "market": "E",
        },
        {
            "ts_code": "510003.SH",
            "name": "未来ETF",
            "fund_type": "股票型",
            "found_date": "20270101",
            "list_date": "20270102",
            "delist_date": None,
            "market": "E",
        },
    ]

    result = collector.normalize_master(rows)

    assert result.get_column("symbol").to_list() == [
        "510001.SH",
        "510002.SH",
    ]


def test_request_range_extends_legacy_and_starts_new_fund_at_listing() -> None:
    legacy = {
        "symbol": "510001.SH",
        "list_date": date(2010, 1, 1),
        "delist_date": None,
    }
    new = {
        "symbol": "510002.SH",
        "list_date": date(2022, 6, 1),
        "delist_date": date(2024, 3, 1),
    }

    assert collector.request_range(legacy, {"510001.SH"}) == (
        date(2021, 1, 1),
        collector.END,
    )
    assert collector.request_range(new, {"510001.SH"}) == (
        date(2022, 6, 1),
        date(2024, 3, 1),
    )


def test_adjustments_are_chunked_below_the_provider_row_cap() -> None:
    assert collector.adjustment_ranges(date(2013, 7, 1), date(2026, 2, 1)) == [
        (date(2013, 7, 1), date(2020, 12, 31)),
        (date(2021, 1, 1), date(2026, 2, 1)),
    ]


def test_daily_amount_is_converted_from_thousand_yuan() -> None:
    result = collector.normalize_daily(
        [
            {
                "ts_code": "510001.SH",
                "trade_date": "20210104",
                "open": "1",
                "high": "1.1",
                "low": "0.9",
                "close": "1.05",
                "vol": "1000",
                "amount": "123.5",
            }
        ],
        start=date(2021, 1, 1),
        end=date(2021, 12, 31),
    )

    assert result.row(0, named=True)["amount"] == 123_500.0
