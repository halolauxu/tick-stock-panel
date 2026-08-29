"""Backfill sparse ex-date adjustment events from Tushare cumulative factors.

This is a bounded research/data-repair utility.  It queries by actual trading
date so a full market history takes one request per session instead of one
request per symbol.  The canonical file is published atomically only after the
whole requested range succeeds.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import polars as pl

from app import secrets_store
from app.plugins.tushare.client import TushareClient

DATA_DIR = Path(os.environ.get("TICKFLOW_DATA_DIR", "/app/data"))
FIELDS = ("ts_code", "trade_date", "adj_factor")


def _trading_dates(start_date: str | None, end_date: str | None) -> list[str]:
    root = DATA_DIR / "kline_daily"
    dates = []
    for path in root.glob("date=*"):
        raw = path.name.removeprefix("date=")
        if (
            len(raw) == 10
            and (start_date is None or raw >= start_date)
            and (end_date is None or raw <= end_date)
        ):
            dates.append(raw)
    return sorted(set(dates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    dates = _trading_dates(args.start_date, args.end_date)
    if not dates:
        raise RuntimeError("no kline_daily partitions found")

    previous: dict[str, float] = {}
    events: list[dict[str, object]] = []
    client = TushareClient(token)
    try:
        for index, trade_date in enumerate(dates, start=1):
            compact = trade_date.replace("-", "")
            rows = client.query("adj_factor", {"trade_date": compact}, FIELDS)
            for row in rows:
                symbol = str(row.get("ts_code") or "").strip()
                try:
                    current = float(row.get("adj_factor"))
                except (TypeError, ValueError):
                    continue
                if not symbol or not math.isfinite(current) or current <= 0:
                    continue
                prior = previous.get(symbol)
                if prior is not None:
                    ratio = current / prior
                    if math.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) > 1e-10:
                        events.append(
                            {
                                "symbol": symbol,
                                "asset_type": "stock",
                                "source": "tushare",
                                "trade_date": trade_date,
                                "ex_factor": ratio,
                            }
                        )
                previous[symbol] = current
            if index == 1 or index % 100 == 0 or index == len(dates):
                print(
                    f"progress={index}/{len(dates)} symbols={len(previous)} events={len(events)}",
                    flush=True,
                )
    finally:
        client.close()

    frame = (
        pl.DataFrame(events, infer_schema_length=None)
        .with_columns(pl.col("trade_date").str.to_date("%Y-%m-%d"))
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort(["symbol", "trade_date"])
    )
    target = DATA_DIR / "adj_factor" / "all.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_existing and target.exists():
        frame = (
            pl.concat([pl.read_parquet(target), frame], how="diagonal_relaxed")
            .unique(subset=["symbol", "trade_date"], keep="last")
            .sort(["symbol", "trade_date"])
        )
    temporary = target.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    print(
        f"published={target} rows={frame.height} symbols={frame['symbol'].n_unique()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
