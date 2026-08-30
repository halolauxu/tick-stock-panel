"""Collect point-in-time earnings-forecast events into yearly partitions."""
from __future__ import annotations

import argparse
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
    "end_date",
    "type",
    "p_change_min",
    "p_change_max",
    "net_profit_min",
    "net_profit_max",
    "last_parent_net",
    "first_ann_date",
    "summary",
    "change_reason",
)
NUMERIC_FIELDS = (
    "p_change_min",
    "p_change_max",
    "net_profit_min",
    "net_profit_max",
    "last_parent_net",
)


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


def _calendar_days(year: int) -> list[date]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _vip_periods_for_announcement_year(year: int) -> tuple[str, ...]:
    return (
        f"{year - 1}1231",
        f"{year}0331",
        f"{year}0630",
        f"{year}0930",
        f"{year}1231",
    )


def fetch_year(client: Any, year: int) -> tuple[list[dict[str, Any]], str]:
    """Prefer the bounded VIP endpoint; fail closed to announcement-day calls."""
    vip_rows: list[dict[str, Any]] = []
    try:
        for period in _vip_periods_for_announcement_year(year):
            vip_rows.extend(
                client.query("forecast_vip", {"period": period}, FIELDS)
            )
    except Exception:
        rows = []
        for day in _calendar_days(year):
            rows.extend(
                client.query(
                    "forecast", {"ann_date": day.strftime("%Y%m%d")}, FIELDS
                )
            )
        return rows, "forecast_by_announcement_day"
    return vip_rows, "forecast_vip_by_period"


def normalize(rows: list[dict[str, Any]], year: int, source: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for field in FIELDS:
        if field not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(field))
    frame = (
        frame.select(FIELDS)
        .rename({"ts_code": "symbol", "end_date": "period_end"})
        .with_columns(
            pl.col("ann_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("first_ann_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(field).cast(pl.Float64, strict=False)
                for field in NUMERIC_FIELDS
            ],
            pl.lit(source).alias("collection_source"),
        )
        .filter(pl.col("ann_date").dt.year() == year)
        .unique(
            subset=[
                "symbol",
                "ann_date",
                "period_end",
                "type",
                "p_change_min",
                "p_change_max",
                "net_profit_min",
                "net_profit_max",
            ],
            keep="last",
        )
        .sort(["ann_date", "symbol", "period_end", "type"])
    )
    if frame.get_column("symbol").null_count() or frame.get_column(
        "ann_date"
    ).null_count():
        raise ValueError("forecast result has missing symbol or announcement date")
    return frame


def collect_year(client: Any, root: Path, year: int) -> dict[str, Any]:
    rows, source = fetch_year(client, year)
    frame = normalize(rows, year, source)
    if frame.is_empty():
        raise ValueError(f"forecast returned no rows for announcement year {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "source": source,
        "path": str(path),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_ann_date": frame.get_column("ann_date").min(),
        "last_ann_date": frame.get_column("ann_date").max(),
        "first_announcements": frame.filter(
            pl.col("first_ann_date").is_null()
            | (pl.col("first_ann_date") == pl.col("ann_date"))
        ).height,
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    root = data_dir / "event_data" / "forecast"
    try:
        results = [collect_year(client, root, year) for year in range(start_year, end_year + 1)]
    finally:
        client.close()
    payload = {
        "dataset": "forecast",
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
