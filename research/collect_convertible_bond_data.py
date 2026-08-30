"""Collect bounded convertible-bond basic, daily, and minute research data."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "backend"))

from probe_convertible_bond_data import (  # noqa: E402
    CB_BASIC_FIELDS,
    CB_DAILY_FIELDS,
)

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

MINUTE_FIELDS = (
    "ts_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)


def normalize_basic(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "stk_code": "stock_symbol"})
        .with_columns(
            pl.col("list_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("delist_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("conv_start_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("conv_end_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("issue_size").cast(pl.Float64, strict=False),
            pl.col("remain_size").cast(pl.Float64, strict=False),
            pl.col("first_conv_price").cast(pl.Float64, strict=False),
            pl.col("conv_price").cast(pl.Float64, strict=False),
        )
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
    )


def normalize_daily(rows: list[dict], trade_date: date) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "vol": "volume_hands"})
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False)
            .alias("date"),
            (pl.col("amount").cast(pl.Float64, strict=False) * 10_000.0).alias(
                "amount_cny"
            ),
            pl.col("volume_hands").cast(pl.Float64, strict=False),
        )
        .drop("trade_date", "amount")
        .filter(pl.col("date") == pl.lit(trade_date))
        .unique(subset=["symbol", "date"], keep="last")
        .sort("symbol")
    )


def normalize_minute(rows: list[dict], start: date, end: date) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "symbol",
                "trade_time": "datetime",
                "vol": "volume_hands",
                "amount": "amount_cny",
            }
        )
        .with_columns(
            pl.col("datetime")
            .cast(pl.Utf8)
            .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.when(pl.col("symbol").str.ends_with(".SZ"))
            .then(pl.col("volume_hands").cast(pl.Float64, strict=False) / 10.0)
            .otherwise(pl.col("volume_hands").cast(pl.Float64, strict=False))
            .alias("volume_hands"),
            pl.col("amount_cny").cast(pl.Float64, strict=False),
        )
        .filter(
            (pl.col("datetime").dt.date() >= pl.lit(start))
            & (pl.col("datetime").dt.date() <= pl.lit(end))
        )
        .unique(subset=["symbol", "datetime"], keep="last")
        .sort(["datetime", "symbol"])
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


def _date_partitions(data_dir: Path, start: date, end: date) -> list[date]:
    root = data_dir / "kline_daily_enriched"
    dates = []
    for path in root.glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end:
            dates.append(value)
    return sorted(set(dates))


def collect_basic(client: TushareClient, root: Path) -> dict[str, Any]:
    rows = client.query("cb_basic", {}, CB_BASIC_FIELDS)
    frame = normalize_basic(rows)
    if frame.is_empty() or frame.get_column("symbol").n_unique() != frame.height:
        raise ValueError("invalid cb_basic result")
    path = root / "basic" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "path": str(path),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_list_date": frame.get_column("list_date").min(),
        "last_list_date": frame.get_column("list_date").max(),
    }


def collect_daily(
    client: TushareClient,
    data_dir: Path,
    root: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    dates = _date_partitions(data_dir, start, end)
    if not dates:
        raise ValueError("no local A-share trading dates for requested range")
    rows = 0
    symbols: set[str] = set()
    empty_dates = []
    for trade_date in dates:
        raw = client.query(
            "cb_daily",
            {"trade_date": trade_date.strftime("%Y%m%d")},
            CB_DAILY_FIELDS,
        )
        frame = normalize_daily(raw, trade_date)
        if frame.is_empty():
            empty_dates.append(trade_date)
            continue
        path = root / "daily" / f"date={trade_date.isoformat()}" / "part.parquet"
        _atomic_write(frame, path)
        rows += frame.height
        symbols.update(frame.get_column("symbol").to_list())
    return {
        "first_date": dates[0],
        "last_date": dates[-1],
        "requested_dates": len(dates),
        "written_dates": len(dates) - len(empty_dates),
        "empty_dates": empty_dates,
        "rows": rows,
        "symbols": len(symbols),
    }


def _symbols_from_daily(root: Path, start: date, end: date) -> list[str]:
    paths = []
    for path in (root / "daily").glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end:
            paths.append(path)
    if not paths:
        raise ValueError("collect daily convertible-bond data before minute data")
    return (
        pl.scan_parquet(paths)
        .select("symbol")
        .unique()
        .collect(engine="streaming")
        .get_column("symbol")
        .sort()
        .to_list()
    )


def collect_minute(
    client: TushareClient,
    root: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    if (end - start).days > 35:
        raise ValueError("minute collection is bounded to at most 36 calendar days")
    symbols = _symbols_from_daily(root, start, end)
    by_date: dict[date, list[pl.DataFrame]] = {}
    empty_symbols = []
    for symbol in symbols:
        raw = client.query(
            "stk_mins",
            {
                "ts_code": symbol,
                "freq": "1min",
                "start_date": datetime.combine(start, time(9, 25)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "end_date": datetime.combine(end, time(15, 5)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
            MINUTE_FIELDS,
        )
        frame = normalize_minute(raw, start, end)
        if frame.is_empty():
            empty_symbols.append(symbol)
            continue
        for key, group in frame.with_columns(
            pl.col("datetime").dt.date().alias("date")
        ).partition_by("date", as_dict=True).items():
            trade_date = key[0] if isinstance(key, tuple) else key
            by_date.setdefault(trade_date, []).append(group.drop("date"))

    total_rows = 0
    duplicate_keys = 0
    for trade_date, frames in sorted(by_date.items()):
        frame = pl.concat(frames, how="diagonal_relaxed")
        unique = frame.unique(subset=["symbol", "datetime"], keep="last").sort(
            ["datetime", "symbol"]
        )
        duplicate_keys += frame.height - unique.height
        path = root / "minute" / f"date={trade_date.isoformat()}" / "part.parquet"
        _atomic_write(unique, path)
        total_rows += unique.height
    return {
        "first_date": start,
        "last_date": end,
        "requested_symbols": len(symbols),
        "empty_symbols": empty_symbols,
        "written_dates": len(by_date),
        "rows": total_rows,
        "duplicate_keys_removed": duplicate_keys,
    }


def run(data_dir: Path, dataset: str, start: date, end: date) -> dict[str, Any]:
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    root = data_dir / "convertible_bond"
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    try:
        if dataset == "basic":
            result = collect_basic(client, root)
        elif dataset == "daily":
            result = collect_daily(client, data_dir, root, start, end)
        elif dataset == "minute":
            result = collect_minute(client, root, start, end)
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
    finally:
        client.close()
    payload = {
        "dataset": dataset,
        "start": start,
        "end": end,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--dataset", choices=("basic", "daily", "minute"), required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 28))
    args = parser.parse_args()
    run(args.data_dir, args.dataset, args.start, args.end)


if __name__ == "__main__":
    main()
