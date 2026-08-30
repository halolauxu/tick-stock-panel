"""Collect and audit the frozen convertible-bond development mirror."""
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

START = date(2017, 1, 1)
END = date(2020, 12, 31)
MASTER_FIELDS = (
    "ts_code",
    "bond_short_name",
    "stk_code",
    "list_date",
    "delist_date",
    "conv_start_date",
    "conv_end_date",
    "maturity_date",
    "conv_price",
    "issue_size",
    "remain_size",
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
    "bond_value",
    "bond_over_rate",
    "cb_value",
    "cb_over_rate",
)


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows, infer_schema_length=None).rename(
        {"ts_code": "symbol"}
    )
    date_columns = (
        "list_date",
        "delist_date",
        "conv_start_date",
        "conv_end_date",
        "maturity_date",
    )
    return (
        frame.with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            *[
                pl.col(column)
                .cast(pl.Utf8)
                .str.to_date("%Y%m%d", strict=False)
                .alias(column)
                for column in date_columns
            ],
        )
        .filter(
            pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
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


def normalize_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
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
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "bond_value",
                    "bond_over_rate",
                    "cb_value",
                    "cb_over_rate",
                )
            ],
            (pl.col("amount").cast(pl.Float64, strict=False) * 10_000.0).alias(
                "amount"
            ),
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
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        master = normalize_master(client.query("cb_basic", {}, MASTER_FIELDS))
        if master.is_empty():
            raise RuntimeError("convertible-bond master is empty")
        symbols = master.get_column("symbol").to_list()
        rows: list[dict[str, Any]] = []
        for index, symbol in enumerate(symbols, start=1):
            rows.extend(
                client.query(
                    "cb_daily",
                    {
                        "ts_code": symbol,
                        "start_date": START.strftime("%Y%m%d"),
                        "end_date": END.strftime("%Y%m%d"),
                    },
                    DAILY_FIELDS,
                )
            )
            if index == 1 or index % 50 == 0 or index == len(symbols):
                print(
                    f"bond_daily_progress={index}/{len(symbols)} rows={len(rows)}",
                    flush=True,
                )
        daily = normalize_daily(rows)
    finally:
        client.close()

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
        .sort(["symbol", "date"])
    )
    active = daily.filter((pl.col("volume") > 0) & (pl.col("amount") > 0))
    invalid_active_ohlc = active.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    ).height
    negative_market_rows = daily.filter(
        (pl.col("volume") < 0) | (pl.col("amount") < 0)
    ).height
    value_rows = active.filter(
        pl.col("cb_value").is_not_null()
        & pl.col("cb_over_rate").is_not_null()
        & (pl.col("cb_value") > 0)
    ).height
    daily_symbols = daily.get_column("symbol").n_unique()
    coverage = daily_symbols / master.height
    value_coverage = value_rows / active.height if active.height else 0.0
    checks = {
        "master_unique": master["symbol"].n_unique() == master.height,
        "daily_unique": daily.unique(["symbol", "date"]).height == daily.height,
        "daily_coverage_at_least_95pct": coverage >= 0.95,
        "active_ohlc_valid": invalid_active_ohlc == 0,
        "market_values_nonnegative": negative_market_rows == 0,
        "conversion_values_at_least_95pct": value_coverage >= 0.95,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    root = data_dir / "research" / "convertible_bond"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(daily, root / "daily.parquet")
    missing = sorted(set(master["symbol"].to_list()) - set(daily["symbol"].to_list()))
    payload = {
        "schema_version": "p0-convertible-bond-data-v1",
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
            "daily_rows": daily.height,
            "daily_symbols": daily_symbols,
            "daily_coverage": coverage,
            "active_rows": active.height,
            "conversion_value_rows": value_rows,
            "conversion_value_coverage": value_coverage,
            "invalid_active_ohlc_rows": invalid_active_ohlc,
            "negative_market_rows": negative_market_rows,
        },
        "checks": checks,
        "missing_symbols": missing,
        "artifacts": {
            "master": str(root / "master.parquet"),
            "daily": str(root / "daily.parquet"),
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
            "/app/data/research/p0_convertible_bond_data_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
