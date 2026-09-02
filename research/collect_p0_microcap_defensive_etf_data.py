"""Collect the frozen ETF mirror for the micro-cap defensive rotation study."""

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
END = date(2026, 8, 28)
SYMBOLS = (
    "510300.SH",
    "513100.SH",
    "518880.SH",
    "511010.SH",
    "511880.SH",
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


def normalize_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {"ts_code": "symbol", "trade_date": "date", "vol": "volume"}
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
        )
        .filter(
            pl.col("symbol").is_in(SYMBOLS)
            & pl.col("date").is_between(START, END, closed="both")
        )
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )


def normalize_adjustments(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "trade_date": "date"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("adj_factor").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("symbol").is_in(SYMBOLS)
            & pl.col("date").is_between(START, END, closed="both")
            & (pl.col("adj_factor") > 0)
        )
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def year_ranges(start: date, end: date) -> list[tuple[date, date]]:
    return [
        (
            max(start, date(year, 1, 1)),
            min(end, date(year, 12, 31)),
        )
        for year in range(start.year, end.year + 1)
    ]


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    daily_rows: list[dict[str, Any]] = []
    adjustment_rows: list[dict[str, Any]] = []
    try:
        for symbol in SYMBOLS:
            daily_rows.extend(
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
            for chunk_start, chunk_end in year_ranges(START, END):
                adjustment_rows.extend(
                    client.query(
                        "fund_adj",
                        {
                            "ts_code": symbol,
                            "start_date": chunk_start.strftime("%Y%m%d"),
                            "end_date": chunk_end.strftime("%Y%m%d"),
                        },
                        ADJ_FIELDS,
                    )
                )
            print(f"collected={symbol}", flush=True)
    finally:
        client.close()

    daily = normalize_daily(daily_rows)
    adjustments = normalize_adjustments(adjustment_rows)
    root = data_dir / "research" / "microcap_defensive_etf_v1"
    existing_master = pl.read_parquet(
        data_dir / "research" / "etf_cross_asset_v2" / "master.parquet"
    )
    master = existing_master.filter(pl.col("symbol").is_in(SYMBOLS)).sort(
        "symbol"
    )
    coverage = daily.join(
        adjustments, on=["symbol", "date"], how="left"
    ).group_by("symbol").agg(
        pl.len().alias("daily_rows"),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
        pl.col("adj_factor").is_not_null().mean().alias("adjustment_coverage"),
        pl.col("amount").median().alias("median_amount"),
    ).sort("symbol")
    invalid = daily.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("amount") < 0)
    ).height
    checks = {
        "all_symbols_in_master": master.height == len(SYMBOLS),
        "all_symbols_have_daily": daily.get_column("symbol").n_unique()
        == len(SYMBOLS),
        "all_symbols_have_adjustments": adjustments.get_column(
            "symbol"
        ).n_unique()
        == len(SYMBOLS),
        "daily_unique": daily.unique(["symbol", "date"]).height
        == daily.height,
        "adjustments_unique": adjustments.unique(
            ["symbol", "date"]
        ).height
        == adjustments.height,
        "valid_ohlcv": invalid == 0,
        "adjustment_coverage_at_least_99pct": (
            coverage.get_column("adjustment_coverage").min() >= 0.99
        ),
        "all_series_reach_end": coverage.get_column("last_date").min()
        >= END,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(daily, root / "daily_raw.parquet")
    _atomic_parquet(adjustments, root / "adjustments.parquet")
    payload = {
        "schema_version": "p0-microcap-defensive-etf-data-v1",
        "contract_frozen": "2026-09-02",
        "period": {"start": START, "end": END},
        "symbols": SYMBOLS,
        "status": status,
        "coverage": coverage.to_dicts(),
        "counts": {
            "daily_rows": daily.height,
            "adjustment_rows": adjustments.height,
            "invalid_ohlcv_rows": invalid,
        },
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
        default=Path(
            "/app/data/research/p0_microcap_defensive_etf_data_v1.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
