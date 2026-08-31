from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_etf_share_flow_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_etf_share_flow_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_eligible_master_keeps_delisted_stock_etf_only() -> None:
    master = pl.DataFrame(
        {
            "symbol": ["510001.SH", "510002.SH", "511001.SH", "510003.SH"],
            "fund_type": ["股票型", "股票型", "货币型", "股票型"],
            "list_date": [
                date(2010, 1, 1),
                date(2021, 1, 1),
                date(2010, 1, 1),
                date(2010, 1, 1),
            ],
            "delist_date": [date(2018, 1, 1), None, None, date(2012, 1, 1)],
        }
    )

    result = study.eligible_master(master)

    assert result["symbol"].to_list() == ["510001.SH"]


def test_normalize_shares_deduplicates_and_rejects_bad_rows() -> None:
    rows = [
        {"ts_code": "510001.SH", "trade_date": "20180102", "fd_share": 10.0},
        {"ts_code": "510001.SH", "trade_date": "20180102", "fd_share": 11.0},
        {"ts_code": "510001.SH", "trade_date": "20180103", "fd_share": 0.0},
        {"ts_code": "510001.SH", "trade_date": "20210104", "fd_share": 12.0},
    ]

    result = study.normalize_shares(rows)

    assert result.height == 1
    assert result["date"][0] == date(2018, 1, 2)
    assert result["shares_10k"][0] == 11.0


def test_audit_never_reads_price_outcomes() -> None:
    symbols = [f"{510000 + index:06d}.SH" for index in range(100)]
    master = pl.DataFrame(
        {
            "symbol": symbols,
            "fund_type": ["股票型"] * len(symbols),
            "list_date": [date(2010, 1, 1)] * len(symbols),
            "delist_date": [None] * len(symbols),
        }
    )
    dates = [date(year, month, 1) for year in range(2013, 2021) for month in range(1, 13)]
    shares = pl.DataFrame(
        {
            "symbol": [symbol for symbol in symbols for _ in dates],
            "date": dates * len(symbols),
            "shares_10k": [100.0] * (len(symbols) * len(dates)),
        }
    )

    result = study.audit(master, shares)

    assert result["price_data_read"] is False
    assert result["future_returns_read"] is False
    assert result["checks"]["outcome_fields_absent"] is True
    assert result["status"] == "SAMPLE_SPARSE"
