"""Audit dividend announcement metadata without reading market outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START_YEAR = 2012
END_YEAR = 2020
DEVELOPMENT_START_YEAR = 2014
SOURCE_ROW_LIMIT = 3_000
KEY = ["symbol", "period_end", "ann_date", "dividend_stage"]
FUTURE_FIELDS = {
    "record_date",
    "ex_date",
    "pay_date",
    "dividend_list_date",
    "implementation_announcement_date",
}


def expected_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "dividend_announcements"
    return [
        root / f"year={year}" / f"month={month:02d}" / "part.parquet"
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]


def audit(data_dir: Path) -> dict[str, Any]:
    planned = expected_paths(data_dir)
    present = [path for path in planned if path.is_file()]
    missing = [str(path) for path in planned if not path.is_file()]
    if not present:
        return {
            "status": "DATA_INCOMPLETE",
            "planned_partitions": len(planned),
            "present_partitions": 0,
            "missing_partitions": missing,
            "future_returns_read": False,
        }
    frame = pl.read_parquet(present, hive_partitioning=False)
    duplicate_rows = frame.height - frame.unique(KEY).height
    invalid_rows = frame.filter(
        ~pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
        | pl.col("period_end").is_null()
        | pl.col("ann_date").is_null()
    ).height
    maximum_daily_rows = (
        frame.group_by("ann_date").len()["len"].max() if frame.height else 0
    )
    annual_cash_plans = (
        frame.filter(
            (pl.col("dividend_stage") == "预案")
            & (pl.col("period_end").dt.month() == 12)
            & (pl.col("cash_dividend_pre_tax_per_share").fill_null(0) > 0)
        )
        .sort(["symbol", "period_end", "ann_date"])
        .unique(["symbol", "period_end"], keep="first")
    )
    yearly = (
        annual_cash_plans.filter(
            pl.col("ann_date").dt.year().is_between(
                DEVELOPMENT_START_YEAR, END_YEAR, closed="both"
            )
        )
        .with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("annual_cash_plans"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("year")
    )
    checks = {
        "all_months_present": not missing,
        "event_key_unique": duplicate_rows == 0,
        "symbols_and_dates_valid": invalid_rows == 0,
        "future_corporate_action_fields_absent": not (
            FUTURE_FIELDS & set(frame.columns)
        ),
        "normalized_daily_rows_below_source_limit": maximum_daily_rows
        < SOURCE_ROW_LIMIT,
        "development_has_at_least_500_annual_cash_plans": annual_cash_plans.filter(
            pl.col("ann_date").dt.year().is_between(
                DEVELOPMENT_START_YEAR, END_YEAR, closed="both"
            )
        ).height
        >= 500,
        "every_development_year_has_annual_cash_plans": yearly.height
        == END_YEAR - DEVELOPMENT_START_YEAR + 1,
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "period": {
            "start": date(START_YEAR, 1, 1),
            "end": date(END_YEAR, 12, 31),
            "future_returns_read": False,
        },
        "planned_partitions": len(planned),
        "present_partitions": len(present),
        "missing_partitions": missing,
        "rows": frame.height,
        "symbols": frame["symbol"].n_unique(),
        "duplicate_rows": duplicate_rows,
        "invalid_rows": invalid_rows,
        "maximum_normalized_daily_rows": maximum_daily_rows,
        "annual_cash_plans": annual_cash_plans.height,
        "development_annual_cash_plans": annual_cash_plans.filter(
            pl.col("ann_date").dt.year().is_between(
                DEVELOPMENT_START_YEAR, END_YEAR, closed="both"
            )
        ).height,
        "yearly": yearly.to_dicts(),
        "checks": checks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "p0-dividend-announcement-data-audit-v1",
        "contract_frozen": "2026-08-31",
        **audit(data_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": digest},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_dividend_announcement_data_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
