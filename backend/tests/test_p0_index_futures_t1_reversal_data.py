from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_index_futures_t1_reversal_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_index_futures_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_year_ranges_are_bounded_and_nonoverlapping() -> None:
    result = study.year_ranges(date(2020, 3, 2), date(2022, 5, 4))

    assert result == [
        (date(2020, 3, 2), date(2020, 12, 31)),
        (date(2021, 1, 1), date(2021, 12, 31)),
        (date(2022, 1, 1), date(2022, 5, 4)),
    ]


def test_normalize_market_deduplicates_and_casts_future_rows() -> None:
    rows = [
        {
            "ts_code": "IF.CFX",
            "trade_date": "20200102",
            "open": "4000",
            "high": "4050",
            "low": "3980",
            "close": "4040",
            "settle": "4035",
            "vol": "100",
            "amount": "10",
            "oi": "20",
        },
        {
            "ts_code": "IF.CFX",
            "trade_date": "20200102",
            "open": "4001",
            "high": "4050",
            "low": "3980",
            "close": "4040",
            "settle": "4035",
            "vol": "100",
            "amount": "10",
            "oi": "20",
        },
    ]

    result = study.normalize_market(rows, future=True)

    assert result.height == 1
    assert result["open"][0] == 4001.0


def test_normalize_mapping_keeps_one_contract_per_series_day() -> None:
    result = study.normalize_mapping(
        [
            {"ts_code": "IF.CFX", "trade_date": "20200102", "mapping_ts_code": "IF2001.CFX"},
            {"ts_code": "IF.CFX", "trade_date": "20200102", "mapping_ts_code": "IF2002.CFX"},
        ]
    )

    assert result.height == 1
    assert result["contract"][0] == "IF2002.CFX"


def test_normalize_master_uses_exchange_contract_multiplier_not_per_unit() -> None:
    result = study.normalize_master(
        [
            {
                "ts_code": "IF2001.CFX",
                "symbol": "IF2001",
                "exchange": "CFFEX",
                "name": "IF2001",
                "fut_code": "IF",
                "per_unit": 1,
                "list_date": "20190101",
                "delist_date": "20200117",
            },
            {
                "ts_code": "IC2001.CFX",
                "symbol": "IC2001",
                "exchange": "CFFEX",
                "name": "IC2001",
                "fut_code": "IC",
                "per_unit": 1,
                "list_date": "20190101",
                "delist_date": "20200117",
            },
        ]
    )

    multipliers = dict(zip(result["fut_code"], result["contract_multiplier"], strict=True))
    assert multipliers == {"IC": 200.0, "IF": 300.0}
