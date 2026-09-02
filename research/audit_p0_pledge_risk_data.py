"""Audit point-in-time pledge disclosures without reading prices or returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START = date(2014, 1, 1)
END = date(2026, 8, 31)
DEVELOPMENT_YEARS = tuple(range(2014, 2021))


def load_events(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "event_data" / "pledge_detail").glob("year=*/part.parquet"))
    if not paths:
        raise ValueError("pledge_detail yearly partitions are missing")
    return pl.read_parquet(paths).filter(
        pl.col("ann_date").is_between(START, END, closed="both")
    )


def audit(data_dir: Path, output: Path) -> dict[str, Any]:
    events = load_events(data_dir)
    universe = pl.read_parquet(data_dir / "research" / "historical_stock_universe.parquet")
    main_board = universe.filter(pl.col("market") == "主板").select(
        "symbol", "list_date", "delist_date"
    )
    eligible = events.join(main_board, on="symbol", how="inner").filter(
        (pl.col("ann_date") >= pl.col("list_date"))
        & (pl.col("delist_date").is_null() | (pl.col("ann_date") <= pl.col("delist_date")))
    )
    material = eligible.filter(
        ~pl.col("is_release").is_in(["1", "Y", "是"])
        & (pl.col("pledge_ratio") >= 5.0)
    )
    rows_by_year = {
        row["year"]: row["rows"]
        for row in (
            events.with_columns(pl.col("ann_date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.len().alias("rows"))
            .sort("year")
            .iter_rows(named=True)
        )
    }
    material_by_year = {
        row["year"]: row["events"]
        for row in (
            material.with_columns(pl.col("ann_date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.len().alias("events"))
            .sort("year")
            .iter_rows(named=True)
        )
    }
    usable = events.filter(pl.col("pledge_ratio").is_not_null()).height
    invalid_ratios = events.filter(
        pl.col("pledge_ratio").is_not_null()
        & ((pl.col("pledge_ratio") < 0) | (pl.col("pledge_ratio") > 100))
    ).height
    usable_rate = usable / events.height
    checks = {
        "every_year_has_rows": all(rows_by_year.get(year, 0) > 0 for year in range(2014, 2027)),
        "pledge_ratio_usable_rate_at_least_95pct": usable_rate >= 0.95,
        "pledge_ratios_within_0_100": invalid_ratios == 0,
        "development_years_each_at_least_50_material_main_board_events": all(
            material_by_year.get(year, 0) >= 50 for year in DEVELOPMENT_YEARS
        ),
        "symbol_announcement_dates_present": events.filter(
            pl.col("symbol").is_null() | pl.col("ann_date").is_null()
        ).height
        == 0,
    }
    paths = sorted((data_dir / "event_data" / "pledge_detail").glob("year=*/part.parquet"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    payload: dict[str, Any] = {
        "schema_version": "p0-main-board-microcap-pledge-risk-data-v1",
        "contract_frozen": "2026-09-03",
        "period": {"start": START, "end": END, "future_returns_read": False},
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "counts": {
            "rows": events.height,
            "symbols": events["symbol"].n_unique(),
            "usable_ratio_rows": usable,
            "usable_ratio_rate": usable_rate,
            "invalid_ratio_rows": invalid_ratios,
            "main_board_rows": eligible.height,
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
        default=Path("/app/data/research/p0_main_board_microcap_pledge_risk_data.json"),
    )
    args = parser.parse_args()
    audit(args.data_dir, args.output)


if __name__ == "__main__":
    main()
