"""Collect a point-in-time full-ETF mirror through the frozen study end."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2013, 1, 1)
LEGACY_END = date(2020, 12, 31)
END = date(2026, 8, 28)
CHECKPOINT_EVERY = 25
MASTER_FIELDS = (
    "ts_code",
    "name",
    "fund_type",
    "found_date",
    "list_date",
    "delist_date",
    "market",
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
ADJ_FIELDS = ("ts_code", "trade_date", "adj_factor")


def empty_daily() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "amount": pl.Float64,
            "source": pl.Utf8,
        }
    )


def empty_adjustments() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "trade_date": pl.Date,
            "adj_factor": pl.Float64,
        }
    )


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("name").cast(pl.Utf8).str.strip_chars(),
            pl.col("found_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("list_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("delist_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
        )
        .filter(
            pl.col("name").str.to_uppercase().str.contains("ETF", literal=True)
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("list_date").is_not_null()
            & (pl.col("list_date") <= pl.lit(END))
            & (pl.col("delist_date").is_null() | (pl.col("delist_date") >= pl.lit(START)))
            & (pl.col("delist_date").is_null() | (pl.col("delist_date") >= pl.col("list_date")))
        )
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
    )


def normalize_daily(rows: list[dict[str, Any]], *, start: date, end: date) -> pl.DataFrame:
    if not rows:
        return empty_daily()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            (pl.col("amount").cast(pl.Float64, strict=False) * 1_000.0).alias("amount"),
            pl.lit("tushare").alias("source"),
        )
        .filter(pl.col("date").is_between(start, end, closed="both"))
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )


def normalize_adjustments(rows: list[dict[str, Any]], *, start: date, end: date) -> pl.DataFrame:
    if not rows:
        return empty_adjustments()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("trade_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("adj_factor").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("trade_date").is_between(start, end, closed="both") & (pl.col("adj_factor") > 0)
        )
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort(["symbol", "trade_date"])
    )


def request_range(row: dict[str, Any], legacy_symbols: set[str]) -> tuple[date, date] | None:
    start = max(
        row["list_date"],
        date(2021, 1, 1) if row["symbol"] in legacy_symbols else START,
    )
    end = min(row.get("delist_date") or END, END)
    return (start, end) if start <= end else None


def adjustment_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Keep each response safely below Tushare's 2,600-row cap."""
    result: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year + 7, 12, 31))
        result.append((cursor, chunk_end))
        cursor = date(chunk_end.year + 1, 1, 1)
    return result


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def _atomic_json(payload: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(target)


def _merge(frames: Iterable[pl.DataFrame], keys: list[str]) -> pl.DataFrame:
    materialized = [frame for frame in frames if not frame.is_empty()]
    if not materialized:
        return empty_daily() if keys == ["symbol", "date"] else empty_adjustments()
    return (
        pl.concat(materialized, how="vertical_relaxed").unique(subset=keys, keep="last").sort(keys)
    )


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    legacy_root = data_dir / "research" / "etf_cross_asset_v2"
    legacy_daily = pl.read_parquet(legacy_root / "daily_raw.parquet")
    legacy_adjustments = pl.read_parquet(legacy_root / "adjustments.parquet")
    legacy_symbols = set(legacy_daily.get_column("symbol").unique().to_list())
    root = data_dir / "research" / "dynamic_etf_rotation_v1"
    checkpoint_daily_path = root / "extension_daily_checkpoint.parquet"
    checkpoint_adj_path = root / "extension_adj_checkpoint.parquet"
    checkpoint_symbols_path = root / "completed_symbols.json"
    extension_daily = (
        pl.read_parquet(checkpoint_daily_path) if checkpoint_daily_path.is_file() else empty_daily()
    )
    extension_adjustments = (
        pl.read_parquet(checkpoint_adj_path)
        if checkpoint_adj_path.is_file()
        else empty_adjustments()
    )
    completed = (
        set(json.loads(checkpoint_symbols_path.read_text(encoding="utf-8")))
        if checkpoint_symbols_path.is_file()
        else set()
    )

    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        master_rows: list[dict[str, Any]] = []
        for status in ("L", "D"):
            master_rows.extend(
                client.query(
                    "fund_basic",
                    {"market": "E", "status": status},
                    MASTER_FIELDS,
                )
            )
        master = normalize_master(master_rows)
        if master.height < 1_700:
            raise RuntimeError(f"ETF master unexpectedly small: {master.height}")
        rows = master.to_dicts()
        pending = [row for row in rows if row["symbol"] not in completed]
        for index, row in enumerate(pending, start=1):
            span = request_range(row, legacy_symbols)
            if span is not None:
                fetch_start, fetch_end = span
                daily_rows = client.query(
                    "fund_daily",
                    {
                        "ts_code": row["symbol"],
                        "start_date": fetch_start.strftime("%Y%m%d"),
                        "end_date": fetch_end.strftime("%Y%m%d"),
                    },
                    DAILY_FIELDS,
                )
                adjustment_rows: list[dict[str, Any]] = []
                for chunk_start, chunk_end in adjustment_ranges(fetch_start, fetch_end):
                    adjustment_rows.extend(
                        client.query(
                            "fund_adj",
                            {
                                "ts_code": row["symbol"],
                                "start_date": chunk_start.strftime("%Y%m%d"),
                                "end_date": chunk_end.strftime("%Y%m%d"),
                            },
                            ADJ_FIELDS,
                        )
                    )
                extension_daily = _merge(
                    (
                        extension_daily,
                        normalize_daily(daily_rows, start=fetch_start, end=fetch_end),
                    ),
                    ["symbol", "date"],
                )
                extension_adjustments = _merge(
                    (
                        extension_adjustments,
                        normalize_adjustments(adjustment_rows, start=fetch_start, end=fetch_end),
                    ),
                    ["symbol", "trade_date"],
                )
            completed.add(row["symbol"])
            if index == 1 or index % CHECKPOINT_EVERY == 0 or index == len(pending):
                _atomic_parquet(extension_daily, checkpoint_daily_path)
                _atomic_parquet(extension_adjustments, checkpoint_adj_path)
                _atomic_json(sorted(completed), checkpoint_symbols_path)
                print(
                    "collection_progress="
                    f"{len(completed)}/{master.height} "
                    f"daily_rows={extension_daily.height} "
                    f"adj_rows={extension_adjustments.height}",
                    flush=True,
                )
    finally:
        client.close()

    master_symbols = set(master.get_column("symbol").to_list())
    daily = (
        _merge(
            (
                legacy_daily.filter(pl.col("symbol").is_in(master_symbols)),
                extension_daily,
            ),
            ["symbol", "date"],
        )
        .join(
            master.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("date") >= pl.col("list_date"))
            & (pl.col("date") <= pl.col("delist_date").fill_null(END))
            & pl.col("date").is_between(START, END, closed="both")
        )
        .drop("list_date", "delist_date")
    )
    adjustments = _merge(
        (
            legacy_adjustments.filter(pl.col("symbol").is_in(master_symbols)),
            extension_adjustments,
        ),
        ["symbol", "trade_date"],
    ).filter(pl.col("trade_date").is_between(START, END, closed="both"))
    joined = daily.join(
        adjustments,
        left_on=["symbol", "date"],
        right_on=["symbol", "trade_date"],
        how="left",
    )
    invalid = daily.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("amount") < 0)
    ).height
    daily_symbols = daily.get_column("symbol").n_unique()
    adj_coverage = joined.get_column("adj_factor").is_not_null().mean()
    checks = {
        "master_at_least_1700": master.height >= 1_700,
        "master_unique": master.get_column("symbol").n_unique() == master.height,
        "daily_unique": daily.unique(["symbol", "date"]).height == daily.height,
        "adjustments_unique": (
            adjustments.unique(["symbol", "trade_date"]).height == adjustments.height
        ),
        "daily_symbol_coverage_at_least_95pct": (daily_symbols / master.height >= 0.95),
        "adjustment_row_coverage_at_least_99pct": adj_coverage >= 0.99,
        "valid_ohlcv": invalid == 0,
        "reaches_frozen_end": daily.get_column("date").max() >= END,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(daily, root / "daily_raw.parquet")
    _atomic_parquet(adjustments, root / "adjustments.parquet")
    payload = {
        "schema_version": "p0-dynamic-etf-rotation-data-v1",
        "contract_frozen": "2026-09-03",
        "period": {"start": START, "end": END},
        "status": status,
        "counts": {
            "master_symbols": master.height,
            "daily_symbols": daily_symbols,
            "daily_rows": daily.height,
            "adjustment_rows": adjustments.height,
            "post_2020_symbols": daily.filter(pl.col("date") > LEGACY_END)
            .get_column("symbol")
            .n_unique(),
            "invalid_ohlcv_rows": invalid,
        },
        "coverage": {"adjustment_rows": adj_coverage},
        "checks": checks,
        "artifacts": {
            "master": str(root / "master.parquet"),
            "daily": str(root / "daily_raw.parquet"),
            "adjustments": str(root / "adjustments.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    payload["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_dynamic_etf_rotation_data_v1.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
