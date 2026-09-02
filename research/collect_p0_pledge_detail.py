"""Collect point-in-time equity-pledge disclosures into yearly partitions."""

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
    "pledge_amount",
    "start_date",
    "end_date",
    "is_release",
    "release_date",
    "pledgor",
    "holding_amount",
    "pledged_amount",
    "p_total_ratio",
    "h_total_ratio",
    "is_buyback",
)
DOCUMENTED_ROW_LIMIT = 1000


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


def fetch_month(client: Any, year: int, month: int) -> tuple[list[dict[str, Any]], str]:
    start, end = _month_bounds(year, month)
    rows = client.query(
        "pledge_detail",
        {
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        },
        FIELDS,
    )
    if len(rows) < DOCUMENTED_ROW_LIMIT:
        return rows, "monthly_range"
    daily_rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        daily = client.query(
            "pledge_detail", {"ann_date": current.strftime("%Y%m%d")}, FIELDS
        )
        if len(daily) >= DOCUMENTED_ROW_LIMIT:
            raise ValueError(f"pledge_detail hit row limit on {current}")
        daily_rows.extend(daily)
        current += timedelta(days=1)
    return daily_rows, "daily_fallback"


def normalize(rows: list[dict[str, Any]], year: int) -> tuple[pl.DataFrame, int]:
    if not rows:
        return pl.DataFrame(), 0
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for field in FIELDS:
        if field not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(field))
    work = (
        frame.select(FIELDS)
        .rename({"ts_code": "symbol", "p_total_ratio": "reported_pledge_ratio"})
        .with_columns(
            pl.col("ann_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("start_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("end_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("release_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("is_release").cast(pl.Utf8).str.strip_chars(),
            *[
                pl.col(field).cast(pl.Float64, strict=False)
                for field in (
                    "pledge_amount",
                    "holding_amount",
                    "pledged_amount",
                    "reported_pledge_ratio",
                    "h_total_ratio",
                )
            ],
        )
        .filter(pl.col("ann_date").dt.year() == year)
        .with_columns(
            pl.when(pl.col("reported_pledge_ratio").is_not_null())
            .then(pl.col("reported_pledge_ratio"))
            .when(
                (pl.col("pledge_amount") >= 0)
                & (pl.col("holding_amount") > 0)
                & (pl.col("h_total_ratio") >= 0)
            )
            .then(
                pl.col("pledge_amount")
                / pl.col("holding_amount")
                * pl.col("h_total_ratio")
            )
            .otherwise(None)
            .alias("pledge_ratio"),
            pl.when(pl.col("reported_pledge_ratio").is_not_null())
            .then(pl.lit("reported"))
            .when(
                (pl.col("pledge_amount") >= 0)
                & (pl.col("holding_amount") > 0)
                & (pl.col("h_total_ratio") >= 0)
            )
            .then(pl.lit("derived_same_row"))
            .otherwise(pl.lit("missing"))
            .alias("pledge_ratio_source"),
        )
    )
    invalid = work.filter(
        pl.col("symbol").is_null()
        | pl.col("ann_date").is_null()
        | ~pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
    ).height
    valid = (
        work.drop_nulls(["symbol", "ann_date"])
        .filter(pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$"))
        .unique(keep="last")
        .sort(["ann_date", "symbol", "holder_name"])
    )
    return valid, invalid


def collect_year(client: Any, root: Path, year: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fallback_months: list[int] = []
    for month in range(1, 13):
        month_rows, source = fetch_month(client, year, month)
        rows.extend(month_rows)
        if source == "daily_fallback":
            fallback_months.append(month)
    frame, invalid_rows = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(f"pledge_detail returned no valid rows for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "raw_rows": len(rows),
        "valid_rows": frame.height,
        "invalid_rows": invalid_rows,
        "symbols": frame["symbol"].n_unique(),
        "announcement_dates": frame["ann_date"].n_unique(),
        "usable_ratio_rows": frame.filter(pl.col("pledge_ratio").is_not_null()).height,
        "material_new_pledges": frame.filter(
            ~pl.col("is_release").is_in(["1", "Y", "是"])
            & (pl.col("pledge_ratio") >= 5.0)
        ).height,
        "fallback_months": fallback_months,
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    root = data_dir / "event_data" / "pledge_detail"
    try:
        results = [collect_year(client, root, year) for year in range(start_year, end_year + 1)]
    finally:
        client.close()
    payload = {
        "dataset": "pledge_detail",
        "start_year": start_year,
        "end_year": end_year,
        "future_returns_read": False,
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
