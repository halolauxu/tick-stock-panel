from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "collect_p0_return_seasonality_warmup.py"
)
SPEC = importlib.util.spec_from_file_location("return_seasonality_warmup", MODULE_PATH)
assert SPEC and SPEC.loader
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def test_expected_months_are_contiguous() -> None:
    months = collector.expected_months()
    assert months[0] == "2007-12"
    assert months[-1] == "2012-12"
    assert len(months) == 61


def test_month_end_trading_dates_uses_last_open_day() -> None:
    rows = [
        {"cal_date": "20071228", "is_open": 1},
        {"cal_date": "20071229", "is_open": 0},
    ]
    for month in collector.expected_months()[1:]:
        rows.append({"cal_date": month.replace("-", "") + "01", "is_open": 1})
    values = collector.month_end_trading_dates(rows)
    assert values[0] == date(2007, 12, 28)
    assert len(values) == 61


def test_normalize_month_joins_exact_adjustment_factor() -> None:
    frame, audit = collector.normalize_month(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "20071228",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "vol": 100,
                "amount": 1_000,
                "pct_chg": 1.0,
            }
        ],
        [{"ts_code": "600000.SH", "trade_date": "20071228", "adj_factor": 2}],
        date(2007, 12, 28),
    )
    assert audit["missing_factors"] == 0
    assert frame.get_column("adjusted_close").item() == 22.0
    assert frame.get_column("source").item() == "tushare_monthly_adj_factor"


def test_normalize_month_excludes_b_shares() -> None:
    rows = []
    factors = []
    for symbol in ("600000.SH", "900901.SH"):
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": "20071228",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "vol": 100,
                "amount": 1_000,
                "pct_chg": 1.0,
            }
        )
        factors.append({"ts_code": symbol, "trade_date": "20071228", "adj_factor": 1})
    frame, audit = collector.normalize_month(rows, factors, date(2007, 12, 28))
    assert frame.get_column("symbol").to_list() == ["600000.SH"]
    assert audit["excluded_non_a_rows"] == 1


def test_normalize_month_rejects_missing_factor() -> None:
    monthly = [
        {
            "ts_code": "600000.SH",
            "trade_date": "20071228",
            "open": 10,
            "high": 12,
            "low": 9,
            "close": 11,
            "vol": 100,
            "amount": 1_000,
            "pct_chg": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="failed audit"):
        collector.normalize_month(
            monthly,
            [{"ts_code": "000001.SZ", "trade_date": "20071228", "adj_factor": 2}],
            date(2007, 12, 28),
        )


def test_final_unique_key_shape() -> None:
    frame, _ = collector.normalize_month(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20071228",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "vol": 100,
                "amount": 1_000,
                "pct_chg": 1.0,
            }
        ],
        [{"ts_code": "000001.SZ", "trade_date": "20071228", "adj_factor": 1}],
        date(2007, 12, 28),
    )
    assert frame.select(pl.struct("symbol", "month_end").n_unique()).item() == 1
