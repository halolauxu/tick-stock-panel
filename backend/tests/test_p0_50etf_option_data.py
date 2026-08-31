from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "research" / "collect_p0_50etf_option_data.py"
    spec = importlib.util.spec_from_file_location("p0_50etf_option_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_normalize_master_filters_target_and_preserves_contract_terms() -> None:
    rows = [
        {
            "ts_code": "10000001.SH",
            "symbol": "x",
            "exchange": "SSE",
            "name": "x",
            "per_unit": 10000,
            "opt_code": study.OPTION_CODE,
            "opt_type": "ETF期权",
            "call_put": "C",
            "exercise_price": 3.0,
            "opt_multiplier": 10000,
            "maturity_date": "20200325",
            "list_date": "20200101",
            "delist_date": "20200325",
            "min_price_chg": "0.0001",
        },
        {
            "ts_code": "10000002.SH",
            "symbol": "y",
            "exchange": "SSE",
            "name": "y",
            "per_unit": 10000,
            "opt_code": "OP510300.SH",
            "opt_type": "ETF期权",
            "call_put": "P",
            "exercise_price": 4.0,
            "opt_multiplier": 10000,
            "maturity_date": "20200325",
            "list_date": "20200101",
            "delist_date": "20200325",
            "min_price_chg": "0.0001",
        },
    ]

    result = study.normalize_master(rows)

    assert result.height == 1
    assert result["contract"][0] == "10000001.SH"
    assert result["opt_multiplier"][0] == 10000.0
    assert result["min_price_chg"][0] == 0.0001


def test_normalize_options_filters_other_option_families_and_deduplicates() -> None:
    base = {
        "trade_date": "20200102",
        "exchange": "SSE",
        "pre_settle": 0.1,
        "pre_close": 0.1,
        "open": 0.11,
        "high": 0.12,
        "low": 0.09,
        "close": 0.105,
        "settle": 0.106,
        "vol": 100,
        "amount": 10,
        "oi": 500,
    }
    rows = [
        {**base, "ts_code": "10000001.SH"},
        {**base, "ts_code": "10000001.SH", "open": 0.12},
        {**base, "ts_code": "10000002.SH"},
    ]

    result = study.normalize_options(rows, {"10000001.SH"})

    assert result.height == 1
    assert result["open"][0] == 0.12
