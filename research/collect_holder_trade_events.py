"""Collect shareholder increase/decrease events into complete yearly partitions."""
from __future__ import annotations

import argparse
import calendar
import contextlib
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

FIELDS = (
    "ts_code",
    "ann_date",
    "holder_name",
    "holder_type",
    "in_de",
    "change_vol",
    "change_ratio",
    "after_share",
    "after_ratio",
    "avg_price",
    "total_share",
    "begin_date",
    "close_date",
)
NUMERIC_FIELDS = (
    "change_vol",
    "change_ratio",
    "after_share",
    "after_ratio",
    "avg_price",
    "total_share",
)
ROW_LIMIT = 3000


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def fetch_month(
    client: Any, year: int, month: int
) -> tuple[list[dict[str, Any]], str]:
    start, end = _month_bounds(year, month)
    rows = client.query(
        "stk_holdertrade",
        {
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        },
        FIELDS,
    )
    if len(rows) < ROW_LIMIT:
        return rows, "monthly_range"
    daily_rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        daily_rows.extend(
            client.query(
                "stk_holdertrade",
                {"ann_date": current.strftime("%Y%m%d")},
                FIELDS,
            )
        )
        current += timedelta(days=1)
    return daily_rows, "daily_fallback"


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for field in FIELDS:
        if field not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(field))
    return (
        frame.select(FIELDS)
        .rename({"ts_code": "symbol", "in_de": "direction"})
        .with_columns(
            pl.col("ann_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("begin_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("close_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(field).cast(pl.Float64, strict=False)
                for field in NUMERIC_FIELDS
            ],
        )
        .filter(pl.col("ann_date").dt.year() == year)
        .drop_nulls(["symbol", "ann_date", "holder_name", "holder_type", "direction"])
        .unique(
            subset=[
                "symbol",
                "ann_date",
                "holder_name",
                "holder_type",
                "direction",
                "change_vol",
                "change_ratio",
                "avg_price",
                "begin_date",
                "close_date",
            ],
            keep="last",
        )
        .sort(["ann_date", "symbol", "direction", "holder_type", "holder_name"])
    )


def collect_year(client: Any, root: Path, year: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fallback_months = []
    for month in range(1, 13):
        month_rows, source = fetch_month(client, year, month)
        rows.extend(month_rows)
        if source == "daily_fallback":
            fallback_months.append(month)
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(f"stk_holdertrade returned no rows for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_ann_date": frame.get_column("ann_date").min(),
        "last_ann_date": frame.get_column("ann_date").max(),
        "fallback_months": fallback_months,
        "direction_counts": frame.group_by("direction")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
        "holder_type_counts": frame.group_by("holder_type")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    root = data_dir / "event_data" / "holder_trade"
    try:
        results = [
            collect_year(client, root, year)
            for year in range(start_year, end_year + 1)
        ]
    finally:
        client.close()
    payload = {
        "dataset": "stk_holdertrade",
        "start_year": start_year,
        "end_year": end_year,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
