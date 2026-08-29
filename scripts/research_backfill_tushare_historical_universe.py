"""Backfill delisted A-share bars, share capital, names and instruments.

The utility is intentionally research-scoped. Existing provider files are never
rewritten: it publishes dedicated parquet shards atomically, and the normal
pipeline can then rebuild enriched bars from the expanded raw universe.
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import date
from pathlib import Path

import polars as pl

from app import secrets_store
from app.plugins.tushare.client import TushareClient

DATA_DIR = Path(os.environ.get("TICKFLOW_DATA_DIR", "/app/data"))
STOCK_BASIC_FIELDS = (
    "ts_code",
    "symbol",
    "name",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
)
DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
SHARE_FIELDS = (
    "ts_code",
    "trade_date",
    "total_share",
    "float_share",
)
NAME_FIELDS = (
    "ts_code",
    "name",
    "start_date",
    "end_date",
    "ann_date",
    "change_reason",
)


def _canonical_date(value: object) -> date | None:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


def _number(value: object, *, scale: float = 1.0) -> float | None:
    try:
        parsed = float(value) * scale
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_main_board(symbol: str) -> bool:
    return (
        symbol.endswith(".SH")
        and symbol.startswith("60")
        or symbol.endswith(".SZ")
        and symbol.startswith(("000", "001", "002", "003"))
    )


def _overlaps(row: dict, start: date, end: date) -> bool:
    listed = _canonical_date(row.get("list_date"))
    delisted = _canonical_date(row.get("delist_date"))
    return listed is not None and listed <= end and (delisted is None or delisted >= start)


def _normalize_daily(rows: list[dict], start: date, end: date) -> pl.DataFrame:
    records = []
    for row in rows:
        symbol = str(row.get("ts_code") or "").strip()
        trade_date = _canonical_date(row.get("trade_date"))
        if not symbol or trade_date is None or not start <= trade_date <= end:
            continue
        records.append(
            {
                "symbol": symbol,
                "date": trade_date,
                "open": _number(row.get("open")),
                "high": _number(row.get("high")),
                "low": _number(row.get("low")),
                "close": _number(row.get("close")),
                "volume": _number(row.get("vol")),
                "amount": _number(row.get("amount"), scale=1_000.0),
            }
        )
    if not records:
        return pl.DataFrame()
    return (
        pl.DataFrame(records, infer_schema_length=None)
        .drop_nulls(["symbol", "date", "open", "high", "low", "close"])
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["date", "symbol"])
    )


def _normalize_shares(rows: list[dict], start: date, end: date) -> pl.DataFrame:
    records = []
    for row in rows:
        symbol = str(row.get("ts_code") or "").strip()
        trade_date = _canonical_date(row.get("trade_date"))
        total_shares = _number(row.get("total_share"), scale=10_000.0)
        float_shares = _number(row.get("float_share"), scale=10_000.0)
        if (
            not symbol
            or trade_date is None
            or not start <= trade_date <= end
            or total_shares is None
            or total_shares <= 0
            or float_shares is None
            or float_shares <= 0
        ):
            continue
        records.append(
            {
                "symbol": symbol,
                "period_end": trade_date.isoformat(),
                "announce_date": trade_date.isoformat(),
                "total_shares": total_shares,
                "float_shares": float_shares,
            }
        )
    if not records:
        return pl.DataFrame()
    frame = (
        pl.DataFrame(records, infer_schema_length=None)
        .unique(subset=["symbol", "period_end"], keep="last")
        .sort(["symbol", "period_end"])
    )
    # Store change points instead of duplicating unchanged daily_basic values.
    return frame.filter(
        (pl.col("total_shares") != pl.col("total_shares").shift(1).over("symbol"))
        | (pl.col("float_shares") != pl.col("float_shares").shift(1).over("symbol"))
        | pl.col("total_shares").shift(1).over("symbol").is_null()
    )


def _normalize_names(rows: list[dict]) -> pl.DataFrame:
    records = []
    for row in rows:
        symbol = str(row.get("ts_code") or "").strip()
        name = str(row.get("name") or "").strip()
        start_date = _canonical_date(row.get("start_date"))
        end_date = _canonical_date(row.get("end_date"))
        announce_date = _canonical_date(row.get("ann_date"))
        if not symbol or not name or start_date is None:
            continue
        records.append(
            {
                "symbol": symbol,
                "name": name,
                "start_date": start_date,
                "end_date": end_date,
                "announce_date": announce_date,
                "change_reason": str(row.get("change_reason") or "").strip(),
            }
        )
    if not records:
        return pl.DataFrame()
    return (
        pl.DataFrame(records, infer_schema_length=None)
        .unique(subset=["symbol", "name", "start_date", "end_date"], keep="last")
        .sort(["symbol", "start_date", "end_date"])
    )


def _instrument_frame(universe: pl.DataFrame, shares: pl.DataFrame, as_of: date) -> pl.DataFrame:
    latest_shares = (
        shares.sort(["symbol", "period_end"])
        .group_by("symbol")
        .agg(
            pl.col("total_shares").last(),
            pl.col("float_shares").last(),
        )
    )
    return (
        universe.join(latest_shares, on="symbol", how="left")
        .select(
            "symbol",
            "name",
            pl.col("symbol").str.split(".").list.first().alias("code"),
            pl.when(pl.col("exchange") == "SSE")
            .then(pl.lit("SH"))
            .otherwise(pl.lit("SZ"))
            .alias("exchange"),
            pl.lit("CN").alias("region"),
            pl.lit("stock").alias("type"),
            pl.col("list_date").dt.strftime("%Y-%m-%d").alias("listing_date"),
            "total_shares",
            "float_shares",
            pl.lit(0.01).alias("tick_size"),
            pl.lit(None, dtype=pl.Float64).alias("limit_up"),
            pl.lit(None, dtype=pl.Float64).alias("limit_down"),
            pl.lit(as_of).alias("as_of"),
        )
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def _publish_daily_shards(frame: pl.DataFrame, data_dir: Path) -> None:
    for key, partition in frame.partition_by("date", as_dict=True).items():
        trade_date = key[0] if isinstance(key, tuple) else key
        target = data_dir / "kline_daily" / f"date={trade_date}" / "tushare_delisted.parquet"
        if target.exists():
            partition = pl.concat(
                [pl.read_parquet(target), partition],
                how="diagonal_relaxed",
            )
        _atomic_parquet(
            partition.unique(subset=["symbol", "date"], keep="last").sort("symbol"),
            target,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2013-08-29")
    parser.add_argument("--end-date", default="2026-08-27")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise ValueError("start date must not exceed end date")

    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    client = TushareClient(token)
    try:
        basics = []
        for status in ("L", "D", "P"):
            basics.extend(
                client.query(
                    "stock_basic",
                    {"exchange": "", "list_status": status},
                    STOCK_BASIC_FIELDS,
                )
            )
        universe_records = []
        for row in basics:
            symbol = str(row.get("ts_code") or "").strip()
            if not _is_main_board(symbol) or not _overlaps(row, start, end):
                continue
            listed = _canonical_date(row.get("list_date"))
            universe_records.append(
                {
                    "symbol": symbol,
                    "name": str(row.get("name") or "").strip(),
                    "market": str(row.get("market") or "").strip(),
                    "exchange": str(row.get("exchange") or "").strip(),
                    "list_status": str(row.get("list_status") or "").strip(),
                    "list_date": listed,
                    "delist_date": _canonical_date(row.get("delist_date")),
                }
            )
        universe = (
            pl.DataFrame(universe_records, infer_schema_length=None)
            .unique(subset=["symbol"], keep="last")
            .sort("symbol")
        )

        name_rows = []
        for year in range(1990, end.year + 1):
            rows = client.query(
                "namechange",
                {"start_date": f"{year}0101", "end_date": f"{year}1231"},
                NAME_FIELDS,
            )
            if len(rows) >= 10_000:
                raise RuntimeError(f"namechange {year} reached the 10000-row response cap")
            name_rows.extend(rows)
        names = _normalize_names(name_rows).filter(
            pl.col("symbol").is_in(universe["symbol"].to_list())
        )

        existing_symbols = set()
        raw_root = DATA_DIR / "kline_daily"
        if raw_root.exists():
            existing_symbols = set(
                pl.scan_parquet(
                    raw_root / "**" / "*.parquet",
                    extra_columns="ignore",
                )
                .select("symbol")
                .unique()
                .collect()["symbol"]
                .to_list()
            )
        delisted = universe.filter(pl.col("list_status") == "D")
        missing = sorted(set(delisted["symbol"].to_list()) - existing_symbols)
        print(
            f"universe={universe.height} delisted={delisted.height} "
            f"missing_raw={len(missing)}",
            flush=True,
        )

        daily_frames = []
        share_frames = []
        failures = []
        universe_by_symbol = {row["symbol"]: row for row in universe.iter_rows(named=True)}
        for index, symbol in enumerate(missing, start=1):
            row = universe_by_symbol[symbol]
            symbol_start = max(start, row["list_date"])
            symbol_end = min(end, row["delist_date"] or end)
            params = {
                "ts_code": symbol,
                "start_date": symbol_start.strftime("%Y%m%d"),
                "end_date": symbol_end.strftime("%Y%m%d"),
            }
            try:
                daily_rows = client.query("daily", params, DAILY_FIELDS)
                share_rows = client.query("daily_basic", params, SHARE_FIELDS)
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")
                continue
            daily = _normalize_daily(daily_rows, symbol_start, symbol_end)
            shares = _normalize_shares(share_rows, symbol_start, symbol_end)
            if daily.is_empty():
                print(
                    f"no_trade_rows={symbol} range={symbol_start}/{symbol_end}",
                    flush=True,
                )
                continue
            daily_frames.append(daily)
            if not shares.is_empty():
                share_frames.append(shares)
            if index == 1 or index % 25 == 0 or index == len(missing):
                print(
                    f"progress={index}/{len(missing)} daily_rows="
                    f"{sum(frame.height for frame in daily_frames)} share_symbols="
                    f"{len(share_frames)} failures={len(failures)}",
                    flush=True,
                )
        if failures:
            raise RuntimeError("historical universe collection failed: " + "; ".join(failures[:20]))
    finally:
        client.close()

    daily = (
        pl.concat(daily_frames, how="diagonal_relaxed")
        if daily_frames
        else pl.DataFrame()
    )
    shares = (
        pl.concat(share_frames, how="diagonal_relaxed")
        if share_frames
        else pl.DataFrame()
    )
    research_root = DATA_DIR / "research"
    _atomic_parquet(universe, research_root / "historical_stock_universe.parquet")
    _atomic_parquet(names, research_root / "historical_stock_names.parquet")
    if not daily.is_empty():
        _atomic_parquet(daily, research_root / "delisted_kline_daily.parquet")
    if not shares.is_empty():
        _atomic_parquet(shares, research_root / "delisted_share_history.parquet")

    if args.publish and not daily.is_empty():
        _publish_daily_shards(daily, DATA_DIR)
        _atomic_parquet(
            shares,
            DATA_DIR / "financials" / "shares" / "tushare_delisted.parquet",
        )
        historical_instruments = _instrument_frame(delisted, shares, end)
        _atomic_parquet(
            historical_instruments,
            DATA_DIR / "instruments" / "tushare_delisted.parquet",
        )
    print(
        f"published={args.publish} daily_rows={daily.height} "
        f"daily_symbols={daily['symbol'].n_unique() if not daily.is_empty() else 0} "
        f"share_rows={shares.height} names={names.height}",
        flush=True,
    )


if __name__ == "__main__":
    main()
