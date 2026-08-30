from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_high_stock_split_unlock_metadata.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_high_stock_split_unlock_metadata", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()

DIVIDEND_SCHEMA = {
    "symbol": pl.String,
    "period_end": pl.Date,
    "ann_date": pl.Date,
    "dividend_stage": pl.String,
    "stock_dividend_per_share": pl.Float64,
    "bonus_share_per_share": pl.Float64,
    "capitalization_share_per_share": pl.Float64,
    "cash_dividend_pre_tax_per_share": pl.Float64,
    "cash_dividend_after_tax_per_share": pl.Float64,
}
UNLOCK_SCHEMA = {
    "symbol": pl.String,
    "ann_date": pl.Date,
    "float_date": pl.Date,
    "float_shares": pl.Float64,
    "float_ratio": pl.Float64,
    "holder_name": pl.String,
    "share_type": pl.String,
}


def _write_complete_dataset(tmp_path: Path, event_count: int) -> None:
    dividend_root = tmp_path / "event_data" / "dividend_announcements"
    dividend_rows: dict[tuple[int, int], list[dict]] = {}
    unlock_rows: dict[int, list[dict]] = {}
    for index in range(event_count):
        year = 2014 + index // 15
        day = index % 15 + 1
        announcement = date(year, 3, day)
        symbol = f"{index + 1:06d}.SZ"
        dividend_rows.setdefault((year, 3), []).append(
            {
                "symbol": symbol,
                "period_end": date(year - 1, 12, 31),
                "ann_date": announcement,
                "dividend_stage": "预案",
                "stock_dividend_per_share": 1.0,
                "bonus_share_per_share": 0.0,
                "capitalization_share_per_share": 1.0,
                "cash_dividend_pre_tax_per_share": 0.0,
                "cash_dividend_after_tax_per_share": 0.0,
            }
        )
        unlock_day = announcement + timedelta(days=60)
        unlock_rows.setdefault(unlock_day.year, []).append(
            {
                "symbol": symbol,
                "ann_date": announcement - timedelta(days=10),
                "float_date": unlock_day,
                "float_shares": 1_000_000.0,
                "float_ratio": 6.0,
                "holder_name": "测试股东",
                "share_type": "首发原股东限售股份",
            }
        )
    for year in range(2012, 2021):
        for month in range(1, 13):
            target = (
                dividend_root
                / f"year={year}"
                / f"month={month:02d}"
                / "part.parquet"
            )
            target.parent.mkdir(parents=True)
            pl.DataFrame(
                dividend_rows.get((year, month), []), schema=DIVIDEND_SCHEMA
            ).write_parquet(target)
    unlock_root = tmp_path / "event_data" / "share_float"
    for year in range(2014, 2022):
        target = unlock_root / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True)
        pl.DataFrame(unlock_rows.get(year, []), schema=UNLOCK_SCHEMA).write_parquet(
            target
        )


def test_incomplete_metadata_never_opens_outcomes(tmp_path) -> None:
    result = audit_module.audit(tmp_path)

    assert result["status"] == "DATA_INCOMPLETE"
    assert result["future_returns_read"] is False
    assert result["price_data_read"] is False


def test_complete_sparse_proposal_ledger_stops_before_unlock_collection(
    tmp_path,
) -> None:
    _write_complete_dataset(tmp_path, 10)
    unlock_root = tmp_path / "event_data" / "share_float"
    for path in unlock_root.glob("year=*/part.parquet"):
        path.unlink()

    result = audit_module.audit(tmp_path)

    assert result["status"] == "SAMPLE_SPARSE"
    assert result["rows"]["high_split_proposals"] == 10
    assert result["rows"]["unlock_details"] is None
    assert result["checks"]["proposal_upper_bound_at_least_40"] is False
    assert result["future_returns_read"] is False


def test_sufficient_point_in_time_intersection_qualifies_sample(tmp_path) -> None:
    _write_complete_dataset(tmp_path, 45)

    result = audit_module.audit(tmp_path)

    assert result["status"] == "SAMPLE_SUFFICIENT"
    assert result["rows"]["high_split_proposals"] == 45
    assert result["rows"]["high_split_with_upcoming_unlock"] == 45
    assert result["coverage"]["matched_signal_days"] == 45
    assert result["coverage"]["matched_years"] == 3
    assert result["future_returns_read"] is False


def test_unlock_announced_after_split_is_not_point_in_time_evidence(tmp_path) -> None:
    _write_complete_dataset(tmp_path, 45)
    path = (
        tmp_path / "event_data" / "share_float" / "year=2014" / "part.parquet"
    )
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("symbol") == "000001.SZ")
        .then(pl.col("ann_date") + pl.duration(days=30))
        .otherwise(pl.col("ann_date"))
        .alias("ann_date")
    )
    frame.write_parquet(path)

    result = audit_module.audit(tmp_path)

    assert result["status"] == "SAMPLE_SUFFICIENT"
    assert result["rows"]["high_split_with_upcoming_unlock"] == 44
    assert result["checks"]["unlock_announced_no_later_than_split"] is True
