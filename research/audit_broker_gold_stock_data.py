"""Audit broker gold-stock metadata and consensus sample sizes without prices."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START_YEAR = 2020
START_MONTH = 7
END_YEAR = 2026
END_MONTH = 8
SOURCE_ROW_LIMIT = 1_000
CONSENSUS_THRESHOLDS = (2, 3, 4, 5)


def required_periods() -> list[tuple[int, int]]:
    periods = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            current = year * 100 + month
            if START_YEAR * 100 + START_MONTH <= current <= END_YEAR * 100 + END_MONTH:
                periods.append((year, month))
    return periods


def load_events(data_dir: Path) -> tuple[pl.DataFrame, int]:
    root = data_dir / "event_data" / "broker_gold_stocks"
    frames = []
    missing = []
    for year, month in required_periods():
        path = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue
        frames.append(
            pl.read_parquet(path).with_columns(
                pl.lit(year).alias("partition_year"),
                pl.lit(month).alias("partition_month"),
            )
        )
    if missing:
        raise ValueError(f"all 74 broker gold-stock partitions are required; missing={len(missing)}")
    return pl.concat(frames, how="vertical_relaxed"), len(frames)


def build_consensus(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(
            schema={
                "recommendation_month": pl.Date,
                "available_after": pl.Date,
                "symbol": pl.String,
                "broker_count": pl.UInt32,
            }
        )
    return (
        events.unique(
            subset=["recommendation_month", "broker", "symbol"], keep="last"
        )
        .group_by(["recommendation_month", "available_after", "symbol"])
        .agg(pl.col("broker").n_unique().cast(pl.UInt32).alias("broker_count"))
        .sort(["recommendation_month", "symbol"])
    )


def _candidate_summary(consensus: pl.DataFrame, threshold: int) -> dict[str, Any]:
    signals = consensus.filter(pl.col("broker_count") >= threshold)
    if signals.is_empty():
        return {
            "signals": 0,
            "signal_months": 0,
            "symbols": 0,
            "median_signals_per_month": 0.0,
            "maximum_signals_per_month": 0,
            "yearly": [],
        }
    monthly_counts = signals.group_by("recommendation_month").len().sort(
        "recommendation_month"
    )
    yearly = (
        signals.with_columns(pl.col("recommendation_month").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("signals"),
            pl.col("recommendation_month").n_unique().alias("months"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("year")
        .to_dicts()
    )
    counts = monthly_counts.get_column("len").to_list()
    return {
        "signals": signals.height,
        "signal_months": signals.get_column("recommendation_month").n_unique(),
        "symbols": signals.get_column("symbol").n_unique(),
        "median_signals_per_month": statistics.median(counts),
        "maximum_signals_per_month": max(counts),
        "yearly": yearly,
    }


def audit(events: pl.DataFrame, partition_count: int) -> dict[str, Any]:
    duplicate_rows = events.height - events.unique(
        subset=["recommendation_month", "broker", "symbol"]
    ).height
    invalid_symbols = events.filter(
        ~pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
    ).height
    partition_mismatch = events.filter(
        (pl.col("recommendation_month").dt.year() != pl.col("partition_year"))
        | (pl.col("recommendation_month").dt.month() != pl.col("partition_month"))
    ).height
    availability_mismatch = events.filter(
        pl.col("available_after")
        != pl.col("recommendation_month").dt.offset_by("2d")
    ).height
    consensus = build_consensus(events)
    nonempty_months = events.get_column("recommendation_month").n_unique()
    monthly = (
        events.group_by("recommendation_month")
        .agg(
            pl.len().alias("rows"),
            pl.col("broker").n_unique().alias("brokers"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("recommendation_month")
    )
    maximum_month_rows = monthly.get_column("rows").max() or 0
    passed = bool(
        partition_count == len(required_periods())
        and nonempty_months >= 60
        and duplicate_rows == 0
        and invalid_symbols == 0
        and partition_mismatch == 0
        and availability_mismatch == 0
        and maximum_month_rows < SOURCE_ROW_LIMIT
    )
    return {
        "schema_version": "p0-broker-gold-stock-metadata-audit-v1",
        "status": "PASS_METADATA" if passed else "DATA_GAP",
        "outcome_fields_read": False,
        "period": {"start": "2020-07", "end": "2026-08"},
        "data": {
            "partitions": partition_count,
            "required_partitions": len(required_periods()),
            "rows": events.height,
            "nonempty_months": nonempty_months,
            "brokers": events.get_column("broker").n_unique(),
            "symbols": events.get_column("symbol").n_unique(),
            "duplicate_rows": duplicate_rows,
            "invalid_symbols": invalid_symbols,
            "partition_mismatch": partition_mismatch,
            "availability_mismatch": availability_mismatch,
            "maximum_month_rows": maximum_month_rows,
            "source_row_limit": SOURCE_ROW_LIMIT,
            "monthly": monthly.to_dicts(),
        },
        "candidate_sample_sizes": {
            f"brokers_gte_{threshold}": _candidate_summary(consensus, threshold)
            for threshold in CONSENSUS_THRESHOLDS
        },
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    events, partition_count = load_events(data_dir)
    payload = audit(events, partition_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_broker_gold_stock_metadata_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
