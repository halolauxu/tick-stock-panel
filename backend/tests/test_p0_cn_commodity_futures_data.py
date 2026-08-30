from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_cn_commodity_futures_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_master_uses_real_contract_and_positive_unit() -> None:
    rows = [
        {
            "ts_code": "M2101.DCE",
            "symbol": "M2101",
            "exchange": "DCE",
            "name": "豆粕2101",
            "fut_code": "M",
            "per_unit": 10.0,
            "trade_unit": "吨",
            "quote_unit": "元/吨",
            "list_date": "20200102",
            "delist_date": "20210115",
            "d_month": "202101",
            "last_ddate": "20210120",
        }
    ]

    result = study.normalize_master(rows)

    assert result["contract"][0] == "M2101.DCE"
    assert result["list_date"][0] == date(2020, 1, 2)
    assert result["per_unit"][0] == 10.0


def test_master_keeps_pre_rename_methanol_contract() -> None:
    rows = [
        {
            "ts_code": "ME1501.ZCE",
            "symbol": "ME1501",
            "exchange": "CZCE",
            "name": "甲醇1501",
            "fut_code": "ME",
            "per_unit": 10.0,
            "trade_unit": "吨",
            "quote_unit": "元/吨",
            "list_date": "20140102",
            "delist_date": "20150115",
            "d_month": "201501",
            "last_ddate": "20150120",
        }
    ]

    result = study.normalize_master(rows)

    assert result["contract"].to_list() == ["ME1501.ZCE"]


def test_daily_and_mapping_normalize_same_key() -> None:
    daily = study.normalize_daily(
        [
            {
                "ts_code": "M.DCE",
                "trade_date": "20201231",
                "open": 3100,
                "high": 3150,
                "low": 3080,
                "close": 3120,
                "settle": 3110,
                "vol": 1000,
                "amount": 100,
                "oi": 500,
            }
        ]
    )
    mapping = study.normalize_mapping(
        [
            {
                "ts_code": "M.DCE",
                "trade_date": "20201231",
                "mapping_ts_code": "M2105.DCE",
            }
        ]
    )

    joined = daily.join(mapping, on=["series", "date"])

    assert joined.height == 1
    assert joined["contract"][0] == "M2105.DCE"
