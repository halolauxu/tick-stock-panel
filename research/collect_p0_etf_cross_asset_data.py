"""Collect and audit the frozen ETF cross-asset development data mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.services import kline_sync  # noqa: E402

START = date(2013, 1, 1)
END = date(2020, 12, 31)
MASTER_FIELDS = (
    "ts_code",
    "name",
    "fund_type",
    "found_date",
    "list_date",
    "delist_date",
    "market",
)
ADJ_FIELDS = ("ts_code", "trade_date", "adj_factor")


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("name").cast(pl.Utf8).str.strip_chars(),
            pl.col("list_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("delist_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("found_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
        )
        .filter(
            pl.col("name").str.to_uppercase().str.contains("ETF", literal=True)
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("list_date").is_not_null()
            & (pl.col("list_date") <= pl.lit(END))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("delist_date") >= pl.lit(START))
            )
            & (
                pl.col("delist_date").is_null()
                | (pl.col("delist_date") >= pl.col("list_date"))
            )
        )
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
    )


def normalize_adjustments(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "trade_date": pl.Date,
                "adj_factor": pl.Float64,
            }
        )
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("adj_factor").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("trade_date").is_between(START, END, closed="both")
            & (pl.col("adj_factor") > 0)
        )
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort(["symbol", "trade_date"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        master_rows = []
        for status in ("L", "D"):
            master_rows.extend(
                client.query(
                    "fund_basic",
                    {"market": "E", "status": status},
                    MASTER_FIELDS,
                )
            )
        master = normalize_master(master_rows)
        if master.is_empty():
            raise RuntimeError("ETF historical master is empty")
        symbols = master.get_column("symbol").to_list()
        failed: list[str] = []
        daily = kline_sync.sync_daily_batch(
            symbols,
            batch_size=50,
            rpm=60,
            start_time=datetime.combine(START, datetime.min.time()),
            end_time=datetime.combine(END, datetime.max.time()),
            failed_out=failed,
        )
        if failed:
            raise RuntimeError(
                f"ETF daily batch failure: {len(failed)} symbols; {failed[:10]}"
            )
        daily = (
            daily.join(
                master.select("symbol", "list_date", "delist_date"),
                on="symbol",
                how="inner",
            )
            .filter(
                (pl.col("date") >= pl.col("list_date"))
                & (
                    pl.col("delist_date").is_null()
                    | (pl.col("date") <= pl.col("delist_date"))
                )
            )
            .drop("list_date", "delist_date")
            .unique(subset=["symbol", "date"], keep="last")
            .sort(["symbol", "date"])
        )
        adjustment_rows = []
        for index, symbol in enumerate(symbols, start=1):
            adjustment_rows.extend(
                client.query(
                    "fund_adj",
                    {
                        "ts_code": symbol,
                        "start_date": START.strftime("%Y%m%d"),
                        "end_date": END.strftime("%Y%m%d"),
                    },
                    ADJ_FIELDS,
                )
            )
            if index == 1 or index % 50 == 0 or index == len(symbols):
                print(
                    f"adjustment_progress={index}/{len(symbols)} rows={len(adjustment_rows)}",
                    flush=True,
                )
        adjustments = normalize_adjustments(adjustment_rows)
    finally:
        client.close()

    daily_symbol_values = (
        set(daily.get_column("symbol").unique().to_list()) if daily.height else set()
    )
    adjustment_symbol_values = (
        set(adjustments.get_column("symbol").unique().to_list())
        if adjustments.height
        else set()
    )
    daily_symbols = len(daily_symbol_values)
    adj_symbols = len(adjustment_symbol_values)
    adjusted_daily_symbols = len(daily_symbol_values & adjustment_symbol_values)
    daily_coverage = daily_symbols / master.height
    adjustment_coverage = (
        adjusted_daily_symbols / daily_symbols if daily_symbols else 0.0
    )
    invalid_ohlc = daily.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("amount") < 0)
    ).height
    checks = {
        "master_unique": master.get_column("symbol").n_unique() == master.height,
        "daily_unique": daily.unique(["symbol", "date"]).height == daily.height,
        "adjustments_unique": (
            adjustments.unique(["symbol", "trade_date"]).height
            == adjustments.height
        ),
        "daily_coverage_at_least_95pct": daily_coverage >= 0.95,
        "adjustment_coverage_at_least_95pct": adjustment_coverage >= 0.95,
        "valid_ohlcv": invalid_ohlc == 0,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    root = data_dir / "research" / "etf_cross_asset"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(daily, root / "daily_raw.parquet")
    _atomic_parquet(adjustments, root / "adjustments.parquet")
    payload = {
        "schema_version": "p0-etf-cross-asset-data-v1",
        "contract_frozen": "2026-08-30",
        "period": {"start": START, "end": END, "validation_returns_read": False},
        "status": status,
        "counts": {
            "master_symbols": master.height,
            "delisted_symbols": master.filter(pl.col("delist_date").is_not_null()).height,
            "daily_rows": daily.height,
            "daily_symbols": daily_symbols,
            "adjustment_rows": adjustments.height,
            "adjustment_symbols": adj_symbols,
            "adjusted_daily_symbols": adjusted_daily_symbols,
            "daily_coverage": daily_coverage,
            "adjustment_coverage": adjustment_coverage,
            "invalid_ohlcv_rows": invalid_ohlc,
        },
        "checks": checks,
        "artifacts": {
            "master": str(root / "master.parquet"),
            "daily_raw": str(root / "daily_raw.parquet"),
            "adjustments": str(root / "adjustments.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    payload["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_etf_cross_asset_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
