from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_dividend_announcement_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_dividend_announcement_data", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _frame(year: int, month: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [f"{year % 1_000_000:06d}.SZ"],
            "period_end": [date(year - 1, 12, 31)],
            "ann_date": [date(year, month, 15)],
            "dividend_stage": ["预案"],
            "stock_dividend_per_share": [0.0],
            "bonus_share_per_share": [0.0],
            "capitalization_share_per_share": [0.0],
            "cash_dividend_pre_tax_per_share": [0.2],
            "cash_dividend_after_tax_per_share": [0.2],
        }
    )


def test_incomplete_months_never_qualify(tmp_path) -> None:
    target = (
        tmp_path
        / "event_data"
        / "dividend_announcements"
        / "year=2012"
        / "month=01"
        / "part.parquet"
    )
    target.parent.mkdir(parents=True)
    _frame(2012, 1).write_parquet(target)

    result = audit.audit(tmp_path)

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["all_months_present"] is False
    assert result["period"]["future_returns_read"] is False


def test_complete_unique_metadata_qualifies_without_outcomes(tmp_path) -> None:
    root = tmp_path / "event_data" / "dividend_announcements"
    for year in range(2012, 2021):
        for month in range(1, 13):
            target = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
            target.parent.mkdir(parents=True)
            rows = []
            if year >= 2014:
                rows = [
                    {
                        **row,
                        "symbol": f"{month * 100 + index:06d}.SZ",
                    }
                    for index, row in enumerate(
                        _frame(year, month).to_dicts() * 8,
                        start=1,
                    )
                ]
            frame = pl.DataFrame(rows, schema=_frame(2012, 1).schema)
            frame.write_parquet(target)

    result = audit.audit(tmp_path)

    assert result["status"] == "DATA_QUALIFIED"
    assert result["development_annual_cash_plans"] >= 500
    assert result["duplicate_rows"] == 0
