from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_etf_cross_asset_data_v2.py"
    )
    spec = importlib.util.spec_from_file_location("p0_etf_data_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_tushare_daily_normalization_converts_amount_to_yuan() -> None:
    rows = [
        {
            "ts_code": "159802.SZ",
            "trade_date": "20200515",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "vol": 123.45,
            "amount": 678.9,
        }
    ]

    result = study.normalize_tushare_daily(rows)

    assert result.height == 1
    assert result["symbol"][0] == "159802.SZ"
    assert result["date"][0] == date(2020, 5, 15)
    assert result["volume"][0] == 123.45
    assert result["amount"][0] == 678_900.0
    assert result["source"][0] == "tushare_gap_fill"


def test_tushare_daily_normalization_deduplicates_and_bounds_dates() -> None:
    rows = [
        {
            "ts_code": "159802.SZ",
            "trade_date": "20121231",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "vol": 1.0,
            "amount": 1.0,
        },
        {
            "ts_code": "159802.SZ",
            "trade_date": "20200515",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.01,
            "vol": 1.0,
            "amount": 1.0,
        },
        {
            "ts_code": "159802.SZ",
            "trade_date": "20200515",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.02,
            "vol": 1.0,
            "amount": 1.0,
        },
    ]

    result = study.normalize_tushare_daily(rows)

    assert result.height == 1
    assert result["close"][0] == 1.02
