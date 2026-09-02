"""Audit market-wide margin-balance data without reading prices or returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START = date(2014, 1, 1)
END = date(2020, 12, 31)
YEARS = tuple(range(2014, 2021))


def expected_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "margin_detail"
    return [root / f"year={year}" / "part.parquet" for year in YEARS]


def load_margin(data_dir: Path) -> pl.DataFrame:
    paths = expected_paths(data_dir)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"margin_detail yearly partitions are missing: {missing}")
    return (
        pl.read_parquet(paths, hive_partitioning=False)
        .filter(pl.col("trade_date").is_between(START, END, closed="both"))
        .select("symbol", "trade_date", "rzye")
        .sort(["trade_date", "symbol"])
    )


def comparable_margin(frame: pl.DataFrame) -> pl.DataFrame:
    calendar = frame.select("trade_date").unique().sort("trade_date").with_row_index("trade_index")
    return (
        frame.join(calendar, on="trade_date", how="left")
        .sort(["symbol", "trade_date"])
        .with_columns(
            pl.col("rzye").shift(1).over("symbol").alias("previous_rzye"),
            pl.col("trade_index").shift(1).over("symbol").alias("previous_trade_index"),
        )
        .filter(
            (pl.col("previous_rzye") > 0)
            & (pl.col("trade_index") == pl.col("previous_trade_index") + 1)
        )
        .with_columns((pl.col("rzye") / pl.col("previous_rzye") - 1.0).alias("balance_change"))
    )


def audit(data_dir: Path, output: Path) -> dict[str, Any]:
    paths = expected_paths(data_dir)
    margin = load_margin(data_dir)
    comparable = comparable_margin(margin)
    duplicate_keys = margin.height - margin.select("symbol", "trade_date").unique().height
    null_core = margin.filter(
        pl.col("symbol").is_null() | pl.col("trade_date").is_null() | pl.col("rzye").is_null()
    ).height
    negative_balance = margin.filter(pl.col("rzye") < 0).height
    year_coverage = {
        row["year"]: {"trading_days": row["trading_days"], "rows": row["rows"]}
        for row in (
            margin.with_columns(pl.col("trade_date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("trade_date").n_unique().alias("trading_days"),
                pl.len().alias("rows"),
            )
            .sort("year")
            .iter_rows(named=True)
        )
    }
    daily_comparable = comparable.group_by("trade_date").agg(pl.len().alias("comparable_symbols"))
    checks = {
        "all_2014_2020_partitions_present": len(paths) == len(YEARS)
        and all(path.is_file() for path in paths),
        "each_year_at_least_240_trading_days": all(
            year_coverage.get(year, {}).get("trading_days", 0) >= 240 for year in YEARS
        ),
        "core_fields_complete": null_core == 0,
        "unique_symbol_dates": duplicate_keys == 0,
        "nonnegative_margin_balances": negative_balance == 0,
        "each_comparable_day_at_least_100_symbols": daily_comparable.height > 0
        and daily_comparable["comparable_symbols"].min() >= 100,
    }
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    payload: dict[str, Any] = {
        "schema_version": "p0-main-board-microcap-margin-deleveraging-data-v1",
        "contract_frozen": "2026-09-03",
        "period": {"start": START, "end": END, "price_or_returns_read": False},
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "counts": {
            "rows": margin.height,
            "symbols": margin["symbol"].n_unique(),
            "trading_days": margin["trade_date"].n_unique(),
            "comparable_rows": comparable.height,
            "minimum_daily_comparable_symbols": daily_comparable["comparable_symbols"].min(),
            "null_core_rows": null_core,
            "duplicate_symbol_dates": duplicate_keys,
            "negative_balance_rows": negative_balance,
            "year_coverage": year_coverage,
        },
        "checks": checks,
        "source_sha256": digest.hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    payload["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_main_board_microcap_margin_deleveraging_data.json"),
    )
    args = parser.parse_args()
    audit(args.data_dir, args.output)


if __name__ == "__main__":
    main()
