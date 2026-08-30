from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_etf_cross_asset_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_etf_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_master_keeps_delisted_etf_and_rejects_future_or_non_etf() -> None:
    rows = [
        {
            "ts_code": "510001.SH",
            "name": "历史ETF",
            "fund_type": "股票型",
            "found_date": "20100101",
            "list_date": "20100201",
            "delist_date": "20191231",
            "market": "E",
        },
        {
            "ts_code": "510002.SH",
            "name": "未来ETF",
            "fund_type": "股票型",
            "found_date": "20210101",
            "list_date": "20210201",
            "delist_date": None,
            "market": "E",
        },
        {
            "ts_code": "160001.SZ",
            "name": "普通LOF",
            "fund_type": "股票型",
            "found_date": "20100101",
            "list_date": "20100201",
            "delist_date": None,
            "market": "E",
        },
    ]

    result = study.normalize_master(rows)

    assert result.get_column("symbol").to_list() == ["510001.SH"]
    assert result["list_date"][0] == date(2010, 2, 1)
    assert result["delist_date"][0] == date(2019, 12, 31)


def test_adjustment_normalization_deduplicates_and_rejects_invalid() -> None:
    rows = [
        {"ts_code": "510001.SH", "trade_date": "20200102", "adj_factor": 1.0},
        {"ts_code": "510001.SH", "trade_date": "20200102", "adj_factor": 1.1},
        {"ts_code": "510001.SH", "trade_date": "20200103", "adj_factor": 0.0},
    ]

    result = study.normalize_adjustments(rows)

    assert result.height == 1
    assert result["adj_factor"][0] == 1.1
