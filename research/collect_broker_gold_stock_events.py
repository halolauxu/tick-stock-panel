"""Collect one month of point-in-time broker gold-stock recommendations."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START_YEAR = 2020
START_MONTH = 7
END_YEAR = 2026
END_MONTH = 8
SOURCE_ROW_LIMIT = 1_000
FIELDS = ("month", "broker", "ts_code", "name")
EVENT_SCHEMA = {
    "recommendation_month": pl.Date,
    "available_after": pl.Date,
    "broker": pl.String,
    "symbol": pl.String,
    "name_at_source": pl.String,
}


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def validate_period(year: int, month: int) -> None:
    current = year * 100 + month
    lower = START_YEAR * 100 + START_MONTH
    upper = END_YEAR * 100 + END_MONTH
    if month not in range(1, 13) or not lower <= current <= upper:
        raise ValueError("broker gold-stock collection must be within 2020-07..2026-08")


def normalize(rows: list[dict[str, Any]], year: int, month: int) -> pl.DataFrame:
    if not rows:
        return _empty_events()
    expected_month = f"{year:04d}{month:02d}"
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "month": "source_month",
                "ts_code": "symbol",
                "name": "name_at_source",
            }
        )
        .with_columns(
            pl.col("source_month").cast(pl.Utf8).str.strip_chars(),
            pl.col("broker").cast(pl.Utf8).str.strip_chars(),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("name_at_source").cast(pl.Utf8).str.strip_chars(),
        )
        .filter(
            (pl.col("source_month") == expected_month)
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
            & (pl.col("broker").str.len_chars() > 0)
        )
        .with_columns(
            pl.lit(date(year, month, 1)).alias("recommendation_month"),
            # The provider only promises that the monthly list is updated on
            # calendar days 1-3.  Treat it as unavailable until day 3 ends.
            pl.lit(date(year, month, 3)).alias("available_after"),
        )
        .unique(
            subset=["recommendation_month", "broker", "symbol"], keep="last"
        )
        .sort(["broker", "symbol"])
    )
    return frame.select(*EVENT_SCHEMA) if not frame.is_empty() else _empty_events()


def fetch_month(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
    year: int,
    month: int,
) -> list[dict[str, Any]]:
    rows = fetch(
        "broker_recommend",
        {"month": f"{year:04d}{month:02d}"},
        FIELDS,
    )
    if len(rows) >= SOURCE_ROW_LIMIT:
        raise ValueError(
            f"Tushare broker_recommend response may be truncated: {len(rows)}"
        )
    return rows


def _atomic_write(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def collect_month(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
    root: Path,
    year: int,
    month: int,
) -> dict[str, Any]:
    validate_period(year, month)
    rows = fetch_month(fetch, year, month)
    frame = normalize(rows, year, month)
    target = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
    _atomic_write(frame, target)
    return {
        "year": year,
        "month": month,
        "path": str(target),
        "raw_rows": len(rows),
        "events": frame.height,
        "brokers": frame.get_column("broker").n_unique(),
        "symbols": frame.get_column("symbol").n_unique(),
        "source_row_limit": SOURCE_ROW_LIMIT,
    }


def run(data_dir: Path, year: int, month: int) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        result = collect_month(
            client.query,
            data_dir / "event_data" / "broker_gold_stocks",
            year,
            month,
        )
    finally:
        client.close()
    payload = {
        "dataset": "tushare_broker_monthly_gold_stock_metadata",
        "outcome_fields_persisted": False,
        "exact_publication_timestamp_available": False,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year, args.month)


if __name__ == "__main__":
    main()
