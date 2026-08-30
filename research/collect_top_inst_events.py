"""Collect one quarter of daily Dragon-Tiger institutional seat details."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

FIELDS = (
    "trade_date",
    "ts_code",
    "exalter",
    "side",
    "buy",
    "buy_rate",
    "sell",
    "sell_rate",
    "net_buy",
    "reason",
)
NUMERIC_FIELDS = ("buy", "buy_rate", "sell", "sell_rate", "net_buy")
ROW_LIMIT = 10000


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


def trading_dates(data_dir: Path, year: int, quarter: int) -> list[date]:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    dates = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*"):
        try:
            value = date.fromisoformat(path.name.removeprefix("date="))
        except ValueError:
            continue
        if value.year == year and start_month <= value.month <= end_month:
            dates.append(value)
    return sorted(set(dates))


def normalize(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for field in FIELDS:
        if field not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(field))
    return (
        frame.select(FIELDS)
        .rename({"ts_code": "symbol", "exalter": "seat_name"})
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(field).cast(pl.Float64, strict=False)
                for field in NUMERIC_FIELDS
            ],
        )
        .drop_nulls(["trade_date", "symbol", "seat_name", "net_buy"])
        .unique(
            subset=[
                "trade_date",
                "symbol",
                "seat_name",
                "side",
                "buy",
                "sell",
                "net_buy",
                "reason",
            ],
            keep="last",
        )
        .sort(["trade_date", "symbol", "seat_name", "reason", "side"])
    )


def collect_quarter(
    client: Any,
    data_dir: Path,
    year: int,
    quarter: int,
) -> dict[str, Any]:
    dates = trading_dates(data_dir, year, quarter)
    if not dates:
        raise ValueError(f"no trading dates for {year}Q{quarter}")
    rows: list[dict[str, Any]] = []
    empty_dates = 0
    for trade_date in dates:
        daily = client.query(
            "top_inst", {"trade_date": trade_date.strftime("%Y%m%d")}, FIELDS
        )
        if len(daily) >= ROW_LIMIT:
            raise ValueError(f"top_inst hit row limit on {trade_date}")
        rows.extend(daily)
        empty_dates += int(not daily)
    frame = normalize(rows)
    if frame.is_empty():
        raise ValueError(f"top_inst returned no rows for {year}Q{quarter}")
    path = (
        data_dir
        / "event_data"
        / "top_inst"
        / f"year={year}"
        / f"quarter={quarter}"
        / "part.parquet"
    )
    _atomic_write(frame, path)
    special = frame.get_column("seat_name").str.contains("机构专用", literal=True)
    northbound = frame.get_column("seat_name").str.contains(
        r"(?:沪股通专用|深股通专用)"
    )
    return {
        "year": year,
        "quarter": quarter,
        "path": str(path),
        "trading_dates": len(dates),
        "empty_dates": empty_dates,
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "institution_special_rows": int(special.sum()),
        "northbound_rows": int(northbound.sum()),
        "first_trade_date": frame.get_column("trade_date").min(),
        "last_trade_date": frame.get_column("trade_date").max(),
    }


def run(data_dir: Path, year: int, quarter: int) -> dict[str, Any]:
    if year < 2014 or year > 2026 or quarter not in {1, 2, 3, 4}:
        raise ValueError("collection must be one valid 2014-2026 quarter")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    try:
        result = collect_quarter(client, data_dir, year, quarter)
    finally:
        client.close()
    payload = {"dataset": "top_inst", "result": result}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--quarter", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year, args.quarter)


if __name__ == "__main__":
    main()
