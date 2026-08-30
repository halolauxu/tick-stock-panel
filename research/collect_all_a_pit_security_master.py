"""Collect a point-in-time security master for all Shanghai/Shenzhen A shares."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

START_YEAR = 1990
SYMBOL_PATTERN = r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"
STOCK_BASIC_FIELDS = (
    "ts_code",
    "name",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
)
NAMECHANGE_FIELDS = (
    "ts_code",
    "name",
    "start_date",
    "end_date",
    "ann_date",
    "change_reason",
)


def normalize_universe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.col("name").cast(pl.Utf8),
            pl.col("market").cast(pl.Utf8),
            pl.col("exchange").cast(pl.Utf8),
            pl.col("list_status").cast(pl.Utf8),
            pl.col("list_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("delist_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
        )
        .filter(
            pl.col("symbol").str.contains(SYMBOL_PATTERN)
            & pl.col("list_date").is_not_null()
        )
        .sort(["symbol", "list_status"])
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
    )


def normalize_names(
    rows: list[dict[str, Any]],
    universe: pl.DataFrame,
) -> pl.DataFrame:
    symbols = universe.select("symbol")
    history = (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.col("name").cast(pl.Utf8),
            pl.col("start_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("end_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("ann_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False)
            .alias("announce_date"),
            pl.col("change_reason").cast(pl.Utf8),
        )
        .join(symbols, on="symbol", how="inner")
        .drop_nulls(["symbol", "name", "start_date"])
        .sort(["symbol", "start_date", "announce_date"])
        .unique(subset=["symbol", "start_date"], keep="last")
    )
    missing = universe.join(
        history.select("symbol").unique(),
        on="symbol",
        how="anti",
    ).select(
        "symbol",
        "name",
        pl.col("list_date").alias("start_date"),
        pl.lit(None).cast(pl.Date).alias("end_date"),
        pl.col("list_date").alias("announce_date"),
        pl.lit("stock_basic_fallback").alias("change_reason"),
    )
    return pl.concat([history, missing], how="vertical_relaxed").sort(
        ["symbol", "start_date"]
    )


def validate_master(
    universe: pl.DataFrame,
    names: pl.DataFrame,
) -> dict[str, Any]:
    duplicate_universe = universe.select(pl.col("symbol").is_duplicated().sum()).item()
    duplicate_names = names.select(
        pl.struct("symbol", "start_date").is_duplicated().sum()
    ).item()
    missing_names = universe.join(
        names.select("symbol").unique(),
        on="symbol",
        how="anti",
    ).height
    invalid_intervals = names.filter(
        pl.col("end_date").is_not_null()
        & (pl.col("end_date") < pl.col("start_date"))
    ).height
    if duplicate_universe or duplicate_names or missing_names or invalid_intervals:
        raise ValueError(
            "invalid all-A PIT master: "
            f"duplicate_universe={duplicate_universe}, "
            f"duplicate_names={duplicate_names}, missing_names={missing_names}, "
            f"invalid_intervals={invalid_intervals}"
        )
    return {
        "universe_rows": universe.height,
        "universe_symbols": universe.get_column("symbol").n_unique(),
        "name_rows": names.height,
        "name_symbols": names.get_column("symbol").n_unique(),
        "fallback_name_symbols": names.filter(
            pl.col("change_reason") == "stock_basic_fallback"
        ).height,
        "prefix_counts": (
            universe.with_columns(pl.col("symbol").str.slice(0, 2).alias("prefix"))
            .group_by("prefix")
            .len()
            .sort("prefix")
            .to_dicts()
        ),
        "board_counts": universe.group_by("market").len().sort("market").to_dicts(),
        "list_status_counts": (
            universe.group_by("list_status").len().sort("list_status").to_dicts()
        ),
    }


def collect(output_dir: Path, *, end_year: int) -> dict[str, Any]:
    client = TushareClient(get_api_key())
    try:
        universe_rows: list[dict[str, Any]] = []
        for status in ("L", "D", "P"):
            rows = client.query(
                "stock_basic",
                {"list_status": status},
                STOCK_BASIC_FIELDS,
            )
            universe_rows.extend(rows)
            print(f"stock_basic status={status} rows={len(rows)}", flush=True)
        universe = normalize_universe(universe_rows)
        name_rows: list[dict[str, Any]] = []
        for year in range(START_YEAR, end_year + 1):
            rows = client.query(
                "namechange",
                {
                    "start_date": f"{year}0101",
                    "end_date": f"{year}1231",
                },
                NAMECHANGE_FIELDS,
            )
            name_rows.extend(rows)
            print(
                f"namechange year={year} rows={len(rows)} total={len(name_rows)}",
                flush=True,
            )
    finally:
        client.close()
    names = normalize_names(name_rows, universe)
    audit = validate_master(universe, names)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe.write_parquet(output_dir / "historical_stock_universe_all_a.parquet")
    names.write_parquet(output_dir / "historical_stock_names_all_a.parquet")
    payload = {
        "schema_version": "all-a-pit-security-master-v1",
        "collected_through_year": end_year,
        **audit,
    }
    (output_dir / "historical_stock_master_all_a_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/data/research"),
    )
    parser.add_argument("--end-year", type=int, default=date.today().year)
    args = parser.parse_args()
    collect(args.output_dir, end_year=args.end_year)


if __name__ == "__main__":
    main()
