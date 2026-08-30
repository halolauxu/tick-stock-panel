from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_cn_commodity_futures_contract_data_v3.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_contract_daily_keeps_real_contract_key() -> None:
    rows = [
        {
            "ts_code": "M2105.DCE",
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

    result = study.normalize_contract_daily(rows)

    assert result["contract"][0] == "M2105.DCE"
    assert result["date"][0] == date(2020, 12, 31)
    assert result["settle"][0] == 3110.0
