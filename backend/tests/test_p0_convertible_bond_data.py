from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_convertible_bond_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_cb_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_master_keeps_development_delist_and_rejects_old_bond() -> None:
    rows = [
        {
            "ts_code": "110001.SH",
            "bond_short_name": "开发退市债",
            "stk_code": "600001.SH",
            "list_date": "20170103",
            "delist_date": "20191231",
            "conv_start_date": "20170703",
            "conv_end_date": "20191230",
            "maturity_date": "20221231",
            "conv_price": 10.0,
            "issue_size": 1.0,
            "remain_size": 0.0,
        },
        {
            "ts_code": "110002.SH",
            "bond_short_name": "过早退市债",
            "stk_code": "600002.SH",
            "list_date": "20140103",
            "delist_date": "20161231",
            "conv_start_date": "20140703",
            "conv_end_date": "20161230",
            "maturity_date": "20191231",
            "conv_price": 10.0,
            "issue_size": 1.0,
            "remain_size": 0.0,
        },
    ]

    result = study.normalize_master(rows)

    assert result["symbol"].to_list() == ["110001.SH"]
    assert result["delist_date"][0] == date(2019, 12, 31)


def test_daily_converts_amount_from_ten_thousand_yuan() -> None:
    rows = [
        {
            "ts_code": "110001.SH",
            "trade_date": "20180102",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vol": 123.0,
            "amount": 45.6,
            "bond_value": 95.0,
            "bond_over_rate": 5.8,
            "cb_value": 90.0,
            "cb_over_rate": 11.7,
        }
    ]

    result = study.normalize_daily(rows)

    assert result.height == 1
    assert result["date"][0] == date(2018, 1, 2)
    assert result["amount"][0] == 456_000.0
    assert result["volume"][0] == 123.0
