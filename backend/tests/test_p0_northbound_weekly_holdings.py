from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_northbound_weekly_holdings.py"
    )
    spec = importlib.util.spec_from_file_location("p0_northbound_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_normalize_holdings_keeps_only_northbound_a_shares() -> None:
    rows = [
        {
            "code": "90000",
            "trade_date": "20201231",
            "ts_code": "600000.SH",
            "name": "浦发银行",
            "vol": 1000,
            "ratio": 1.2,
            "exchange": "SH",
        },
        {
            "code": "2",
            "trade_date": "20201231",
            "ts_code": "00002.HK",
            "name": "中电控股",
            "vol": 1000,
            "ratio": 0.1,
            "exchange": "HK",
        },
    ]

    result = study.normalize_holdings(rows)

    assert result.height == 1
    assert result["symbol"][0] == "600000.SH"
    assert result["date"][0] == date(2020, 12, 31)
    assert result["holding_shares"][0] == 1000.0
