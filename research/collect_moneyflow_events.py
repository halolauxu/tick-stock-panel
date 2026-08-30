"""Collect full-market daily large-order money-flow records by year."""
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
    "ts_code",
    "trade_date",
    "buy_sm_amount",
    "sell_sm_amount",
    "buy_md_amount",
    "sell_md_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
    "net_mf_amount",
)
AMOUNT_FIELDS = FIELDS[2:]
ROW_LIMIT = 6000


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


def trading_dates(data_dir: Path, year: int) -> list[date]:
    dates = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if value.year == year:
            dates.append(value)
    return sorted(set(dates))


def normalize(rows: list[dict[str, Any]], trade_date: date) -> pl.DataFrame:
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
            pl.col("trade_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                (pl.col(field).cast(pl.Float64, strict=False) * 10_000.0).alias(
                    field.removesuffix("_amount") + "_cny"
                )
                for field in AMOUNT_FIELDS
            ],
        )
        .drop(*AMOUNT_FIELDS)
        .filter(pl.col("trade_date") == trade_date)
        .drop_nulls(["symbol", "trade_date", "net_mf_cny"])
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort("symbol")
    )


def collect_year(client: Any, data_dir: Path, root: Path, year: int) -> dict[str, Any]:
    dates = trading_dates(data_dir, year)
    if not dates:
        raise ValueError(f"no local trading dates for {year}")
    frames = []
    empty_dates = []
    for trade_date in dates:
        rows = client.query(
            "moneyflow", {"trade_date": trade_date.strftime("%Y%m%d")}, FIELDS
        )
        if len(rows) >= ROW_LIMIT:
            raise ValueError(f"moneyflow hit row limit on {trade_date}")
        frame = normalize(rows, trade_date)
        if frame.is_empty():
            empty_dates.append(trade_date)
        else:
            frames.append(frame)
    if empty_dates:
        raise ValueError(f"moneyflow empty on {len(empty_dates)} trading dates: {empty_dates[:5]}")
    frame = pl.concat(frames, how="diagonal_relaxed").sort(["trade_date", "symbol"])
    if frame.get_column("trade_date").n_unique() != len(dates):
        raise ValueError(f"moneyflow date coverage mismatch for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "trading_dates": len(dates),
        "first_trade_date": frame.get_column("trade_date").min(),
        "last_trade_date": frame.get_column("trade_date").max(),
        "duplicate_keys": frame.height
        - frame.select("symbol", "trade_date").unique().height,
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    root = data_dir / "event_data" / "moneyflow"
    try:
        results = [
            collect_year(client, data_dir, root, year)
            for year in range(start_year, end_year + 1)
        ]
    finally:
        client.close()
    payload = {
        "dataset": "moneyflow",
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
