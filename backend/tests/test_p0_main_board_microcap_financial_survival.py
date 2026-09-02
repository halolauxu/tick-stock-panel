from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_main_board_microcap_financial_survival.py"
    )
    spec = importlib.util.spec_from_file_location("p0_microcap_financial_survival", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _candidates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 5, 8)] * 2,
            "entry_date": [date(2020, 5, 11)] * 2,
            "symbol": ["600001.SH", "600002.SH"],
            "market_cap": [1.0, 2.0],
            "signal_amount": [1_000_000.0, 1_000_000.0],
            "cap_rank": [1, 2],
        }
    )


def _snapshots() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "period_end": [date(2019, 12, 31)] * 2,
            "financial_available_date": [date(2020, 4, 30)] * 2,
            "net_income_attributable": [1.0, -1.0],
            "net_operating_cash_flow": [1.0, 1.0],
            "total_assets": [10.0, 10.0],
            "total_liabilities": [4.0, 4.0],
            "total_equity": [6.0, 6.0],
            "goodwill": [0.0, 0.0],
            "debt_ratio": [0.4, 0.4],
            "goodwill_ratio": [0.0, 0.0],
        }
    )


def test_survival_filter_uses_only_available_positive_financials() -> None:
    result = study.attach_survival_filter(_candidates(), _snapshots())

    assert result["symbol"].to_list() == ["600001.SH"]
    assert result.columns == _candidates().columns


def test_survival_filter_rejects_stale_snapshot() -> None:
    candidates = _candidates().with_columns(pl.lit(date(2022, 1, 1)).alias("date"))

    result = study.attach_survival_filter(candidates, _snapshots())

    assert result.is_empty()
