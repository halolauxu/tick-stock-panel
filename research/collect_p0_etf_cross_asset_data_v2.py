"""Repair the frozen ETF research mirror without dropping delisted funds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2013, 1, 1)
END = date(2020, 12, 31)
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


def normalize_tushare_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
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
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "symbol",
                "trade_date": "date",
                "vol": "volume",
            }
        )
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            (pl.col("amount").cast(pl.Float64, strict=False) * 1_000.0).alias(
                "amount"
            ),
            pl.lit("tushare_gap_fill").alias("source"),
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    v1_root = data_dir / "research" / "etf_cross_asset"
    master = pl.read_parquet(v1_root / "master.parquet")
    primary = pl.read_parquet(v1_root / "daily_raw.parquet")
    adjustments = pl.read_parquet(v1_root / "adjustments.parquet")
    primary_symbols = set(primary.get_column("symbol").unique().to_list())
    missing_symbols = [
        symbol
        for symbol in master.get_column("symbol").to_list()
        if symbol not in primary_symbols
    ]

    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        supplement_rows: list[dict[str, Any]] = []
        for index, symbol in enumerate(missing_symbols, start=1):
            supplement_rows.extend(
                client.query(
                    "fund_daily",
                    {
                        "ts_code": symbol,
                        "start_date": START.strftime("%Y%m%d"),
                        "end_date": END.strftime("%Y%m%d"),
                    },
                    DAILY_FIELDS,
                )
            )
            if index == 1 or index % 25 == 0 or index == len(missing_symbols):
                print(
                    f"daily_gap_progress={index}/{len(missing_symbols)} "
                    f"rows={len(supplement_rows)}",
                    flush=True,
                )
    finally:
        client.close()

    supplement = normalize_tushare_daily(supplement_rows)
    supplement = (
        supplement.join(
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
    )
    daily = (
        pl.concat(
            [
                primary.with_columns(pl.lit("tickflow").alias("source")),
                supplement,
            ],
            how="vertical_relaxed",
        )
        .unique(subset=["symbol", "date"], keep="first")
        .sort(["symbol", "date"])
    )

    daily_symbols = set(daily.get_column("symbol").unique().to_list())
    adjustment_symbols = set(
        adjustments.get_column("symbol").unique().to_list()
    )
    daily_coverage = len(daily_symbols) / master.height
    adjusted_daily_symbols = len(daily_symbols & adjustment_symbols)
    adjustment_coverage = adjusted_daily_symbols / len(daily_symbols)
    invalid_ohlcv = daily.filter(
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
        "valid_ohlcv": invalid_ohlcv == 0,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    root = data_dir / "research" / "etf_cross_asset_v2"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(daily, root / "daily_raw.parquet")
    _atomic_parquet(adjustments, root / "adjustments.parquet")
    remaining_missing = sorted(set(master["symbol"].to_list()) - daily_symbols)
    payload = {
        "schema_version": "p0-etf-cross-asset-data-v2",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": START,
            "end": END,
            "validation_returns_read": False,
        },
        "status": status,
        "counts": {
            "master_symbols": master.height,
            "delisted_symbols": master.filter(
                pl.col("delist_date").is_not_null()
            ).height,
            "primary_daily_symbols": len(primary_symbols),
            "requested_gap_symbols": len(missing_symbols),
            "supplement_rows": supplement.height,
            "supplement_symbols": supplement["symbol"].n_unique(),
            "daily_rows": daily.height,
            "daily_symbols": len(daily_symbols),
            "daily_coverage": daily_coverage,
            "adjustment_rows": adjustments.height,
            "adjustment_symbols": len(adjustment_symbols),
            "adjusted_daily_symbols": adjusted_daily_symbols,
            "adjustment_coverage": adjustment_coverage,
            "invalid_ohlcv_rows": invalid_ohlcv,
        },
        "checks": checks,
        "remaining_missing_symbols": remaining_missing,
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
        default=Path(
            "/app/data/research/p0_etf_cross_asset_data_v2_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
