"""Collect and audit one year of four-gold-ETF minute data without returns."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from calendar import monthrange
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2025, 8, 1)
END = date(2026, 8, 28)
SYMBOLS = ("518880.SH", "518800.SH", "159934.SZ", "159937.SZ")
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
DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
)
EXPECTED_BARS = 241
MINIMUM_COMMON_COMPLETE_DAYS = 240


def _month_ranges(start: date, end: date) -> list[tuple[datetime, datetime]]:
    ranges = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        left = max(start, cursor)
        right = min(end, last)
        ranges.append((datetime.combine(left, time(9, 20)), datetime.combine(right, time(15, 5))))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return ranges


def normalize_minutes(rows: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
    schema = {
        "symbol": pl.Utf8,
        "datetime": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "trade_time": "datetime", "vol": "volume"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
            pl.col("datetime").cast(pl.Utf8).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ("open", "high", "low", "close", "volume", "amount")
            ],
        )
        .filter(
            (pl.col("symbol") == symbol)
            & pl.col("datetime").dt.date().is_between(START, END, closed="both")
            & (
                pl.col("datetime").dt.time().is_between(time(9, 30), time(11, 30), closed="both")
                | pl.col("datetime").dt.time().is_between(time(13, 1), time(15, 0), closed="both")
            )
        )
        .select(*schema)
        .unique(["symbol", "datetime"], keep="last")
        .sort("datetime")
    )


def normalize_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ("open", "high", "low", "close", "pre_close", "volume", "amount")
            ],
        )
        .filter(
            pl.col("symbol").is_in(SYMBOLS)
            & pl.col("date").is_between(START, END, closed="both")
        )
        .unique(["symbol", "date"], keep="last")
        .sort(["date", "symbol"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def audit(
    minutes: pl.DataFrame,
    daily: pl.DataFrame,
    minimum_common_days: int = MINIMUM_COMMON_COMPLETE_DAYS,
) -> dict[str, Any]:
    with_dates = minutes.with_columns(pl.col("datetime").dt.date().alias("date"))
    counts = with_dates.group_by("symbol", "date").len().rename({"len": "bars"})
    complete = counts.filter(pl.col("bars") == EXPECTED_BARS)
    common = (
        complete.group_by("date")
        .agg(pl.col("symbol").n_unique().alias("symbols"))
        .filter(pl.col("symbols") == len(SYMBOLS))
    )
    daily_keys = daily.select("symbol", "date").unique() if not daily.is_empty() else pl.DataFrame()
    minute_keys = counts.select("symbol", "date")
    missing_daily = (
        minute_keys.join(daily_keys, on=["symbol", "date"], how="anti").height
        if not daily_keys.is_empty()
        else minute_keys.height
    )
    invalid_prices = minutes.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("amount") < 0)
    )
    checks = {
        "four_symbols_exact": set(minutes["symbol"].unique().to_list()) == set(SYMBOLS),
        "minute_keys_unique": minutes.unique(["symbol", "datetime"]).height == minutes.height,
        "at_least_240_common_complete_days": common.height >= minimum_common_days,
        "all_minute_dates_have_daily_rows": missing_daily == 0,
        "prices_and_activity_valid": invalid_prices.is_empty(),
        "all_symbol_days_are_241_bars": counts.filter(pl.col("bars") != EXPECTED_BARS).is_empty(),
    }
    per_symbol = {
        symbol: {
            "rows": minutes.filter(pl.col("symbol") == symbol).height,
            "days": counts.filter(pl.col("symbol") == symbol).height,
            "complete_days": complete.filter(pl.col("symbol") == symbol).height,
        }
        for symbol in SYMBOLS
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "returns_evaluated": False,
        "strategy_metrics_computed": False,
        "counts": {
            "minute_rows": minutes.height,
            "common_complete_days": common.height,
            "daily_rows": daily.height,
            "per_symbol": per_symbol,
        },
        "checks": checks,
    }


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    root = data_dir / "research" / "gold_etf_dispersion"
    try:
        minute_frames = []
        for symbol in SYMBOLS:
            target = root / "minute" / f"symbol={symbol}" / "part.parquet"
            if target.exists():
                frame = pl.read_parquet(target)
            else:
                rows: list[dict[str, Any]] = []
                for index, (left, right) in enumerate(_month_ranges(START, END), start=1):
                    rows.extend(
                        client.query(
                            "stk_mins",
                            {
                                "ts_code": symbol,
                                "start_date": left.strftime("%Y-%m-%d %H:%M:%S"),
                                "end_date": right.strftime("%Y-%m-%d %H:%M:%S"),
                                "freq": "1min",
                            },
                            MINUTE_FIELDS,
                        )
                    )
                    print(f"minute_progress={symbol}:{index}/13", flush=True)
                frame = normalize_minutes(rows, symbol)
                _atomic_parquet(frame, target)
            minute_frames.append(frame)
        daily_rows: list[dict[str, Any]] = []
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
        daily = normalize_daily(daily_rows)
        _atomic_parquet(daily, root / "daily.parquet")
    finally:
        client.close()
    minutes = pl.concat(minute_frames, how="vertical_relaxed")
    payload = {
        "schema_version": "p0-gold-etf-minute-data-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": START, "end": END},
        "symbols": list(SYMBOLS),
        **audit(minutes, daily),
        "artifacts": {
            "minute_root": str(root / "minute"),
            "daily": str(root / "daily.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({**payload, "sha256": digest}, ensure_ascii=False, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_gold_etf_minute_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
