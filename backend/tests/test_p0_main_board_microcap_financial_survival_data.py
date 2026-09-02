from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "audit_p0_main_board_microcap_financial_survival_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_microcap_financial_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _snapshots() -> pl.DataFrame:
    rows = []
    for year in range(study.START_REPORT_YEAR, study.END_REPORT_YEAR + 1):
        for number in range(study.MIN_SYMBOLS_PER_YEAR):
            rows.append(
                {
                    "symbol": f"{600000 + number:06d}.SH",
                    "period_end": date(year, 12, 31),
                    "financial_available_date": date(year + 1, 4, 30),
                    "net_income_attributable": 1.0,
                    "net_operating_cash_flow": 1.0,
                    "total_assets": 10.0,
                    "total_liabilities": 4.0,
                    "total_equity": 6.0,
                    "debt_ratio": 0.4,
                    "goodwill_ratio": 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_audit_accepts_complete_price_free_annual_snapshots() -> None:
    result = study.audit(_snapshots())

    assert result["status"] == "DATA_QUALIFIED"
    assert result["price_data_read"] is False
    assert result["future_returns_read"] is False


def test_audit_rejects_missing_year_coverage() -> None:
    snapshots = _snapshots().filter(pl.col("period_end").dt.year() != 2017)

    result = study.audit(snapshots)

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["every_year_has_at_least_1000_symbols"] is False


def test_audit_rejects_price_fields() -> None:
    result = study.audit(_snapshots().with_columns(pl.lit(1.0).alias("close")))

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["price_and_return_fields_absent"] is False
