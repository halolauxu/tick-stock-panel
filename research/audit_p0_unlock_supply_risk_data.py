"""Audit scheduled share-unlock data without reading prices or returns."""

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
MIN_FLOAT_RATIO_PCT = 5.0


def expected_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "share_float"
    return [root / f"year={year}" / "part.parquet" for year in YEARS]


def load_details(data_dir: Path) -> pl.DataFrame:
    paths = expected_paths(data_dir)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"share_float yearly partitions are missing: {missing}")
    return pl.read_parquet(paths).filter(pl.col("float_date").is_between(START, END, closed="both"))


def aggregate_material_events(details: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    active_main_board = universe.filter(pl.col("market") == "主板").select(
        "symbol", "list_date", "delist_date"
    )
    return (
        details.group_by("symbol", "float_date")
        .agg(
            pl.col("ann_date").min().alias("first_ann_date"),
            pl.col("ann_date").max().alias("last_ann_date"),
            pl.col("float_shares").sum().alias("float_shares"),
            pl.col("float_ratio").sum().alias("float_ratio_pct"),
            pl.len().alias("detail_rows"),
        )
        .join(active_main_board, on="symbol", how="inner")
        .filter(
            (pl.col("last_ann_date") <= pl.col("float_date"))
            & (pl.col("float_date") >= pl.col("list_date"))
            & (pl.col("delist_date").is_null() | (pl.col("float_date") <= pl.col("delist_date")))
            & (pl.col("float_shares") > 0)
            & (pl.col("float_ratio_pct") >= MIN_FLOAT_RATIO_PCT)
            & (pl.col("float_ratio_pct") <= 100)
        )
        .sort(["float_date", "symbol"])
    )


def audit(data_dir: Path, output: Path) -> dict[str, Any]:
    paths = expected_paths(data_dir)
    details = load_details(data_dir)
    universe = pl.read_parquet(data_dir / "research" / "historical_stock_universe.parquet")
    material = aggregate_material_events(details, universe)
    rows_by_year = {
        row["year"]: row["rows"]
        for row in (
            details.with_columns(pl.col("float_date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.len().alias("rows"))
            .sort("year")
            .iter_rows(named=True)
        )
    }
    material_by_year = {
        row["year"]: row["events"]
        for row in (
            material.with_columns(pl.col("float_date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.len().alias("events"))
            .sort("year")
            .iter_rows(named=True)
        )
    }
    null_core = details.filter(
        pl.any_horizontal(
            pl.col(column).is_null()
            for column in (
                "symbol",
                "ann_date",
                "float_date",
                "float_shares",
                "float_ratio",
            )
        )
    ).height
    invalid_ratio = details.filter(
        (pl.col("float_ratio") <= 0) | (pl.col("float_ratio") > 100)
    ).height
    announced_late = details.filter(pl.col("ann_date") > pl.col("float_date")).height
    checks = {
        "all_2014_2020_partitions_present": len(paths) == len(YEARS)
        and all(path.is_file() for path in paths),
        "every_year_has_rows": all(rows_by_year.get(year, 0) > 0 for year in YEARS),
        "core_fields_complete": null_core == 0,
        "detail_ratios_within_0_100": invalid_ratio == 0,
        "all_details_announced_no_later_than_unlock": announced_late == 0,
        "each_year_at_least_100_material_main_board_events": all(
            material_by_year.get(year, 0) >= 100 for year in YEARS
        ),
    }
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    payload: dict[str, Any] = {
        "schema_version": "p0-main-board-microcap-unlock-supply-data-v1",
        "contract_frozen": "2026-09-03",
        "period": {"start": START, "end": END, "price_or_returns_read": False},
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "counts": {
            "detail_rows": details.height,
            "symbols": details["symbol"].n_unique(),
            "null_core_rows": null_core,
            "invalid_ratio_rows": invalid_ratio,
            "announced_after_unlock_rows": announced_late,
            "material_main_board_events": material.height,
            "rows_by_year": rows_by_year,
            "material_main_board_events_by_year": material_by_year,
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
        default=Path("/app/data/research/p0_main_board_microcap_unlock_supply_data.json"),
    )
    args = parser.parse_args()
    audit(args.data_dir, args.output)


if __name__ == "__main__":
    main()
