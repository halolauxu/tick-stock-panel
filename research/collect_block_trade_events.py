"""Collect block-trade details into complete yearly partitions."""
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

FIELDS = ("ts_code", "trade_date", "price", "vol", "amount", "buyer", "seller")
ROW_LIMIT = 1000


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
        "block_trade",
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
        daily = client.query(
            "block_trade", {"trade_date": current.strftime("%Y%m%d")}, FIELDS
        )
        if len(daily) >= ROW_LIMIT:
            raise ValueError(f"block_trade hit row limit on {current}")
        daily_rows.extend(daily)
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
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("price").cast(pl.Float64, strict=False),
            (pl.col("vol").cast(pl.Float64, strict=False) * 10_000.0).alias(
                "volume_shares"
            ),
            (pl.col("amount").cast(pl.Float64, strict=False) * 10_000.0).alias(
                "notional_cny"
            ),
        )
        .drop("vol", "amount")
        .filter(pl.col("trade_date").dt.year() == year)
        .drop_nulls(["symbol", "trade_date", "price", "volume_shares", "notional_cny"])
        .unique(
            subset=[
                "symbol",
                "trade_date",
                "price",
                "volume_shares",
                "notional_cny",
                "buyer",
                "seller",
            ],
            keep="last",
        )
        .sort(["trade_date", "symbol", "price", "volume_shares", "buyer", "seller"])
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
        raise ValueError(f"block_trade returned no rows for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "event_dates": frame.get_column("trade_date").n_unique(),
        "first_trade_date": frame.get_column("trade_date").min(),
        "last_trade_date": frame.get_column("trade_date").max(),
        "fallback_months": fallback_months,
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    root = data_dir / "event_data" / "block_trade"
    try:
        results = [
            collect_year(client, root, year)
            for year in range(start_year, end_year + 1)
        ]
    finally:
        client.close()
    payload = {
        "dataset": "block_trade",
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
