"""Audit persisted convertible-bond daily/minute data and reconcile units."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

CRITICAL_DAILY = ("open", "high", "low", "close", "volume_hands", "amount_cny")
CRITICAL_MINUTE = (
    "open",
    "high",
    "low",
    "close",
    "volume_hands",
    "amount_cny",
)


def _paths(root: Path, dataset: str, start: date, end: date) -> list[Path]:
    output = []
    for path in (root / dataset).glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end:
            output.append(path)
    return sorted(output)


def _null_rates(frame: pl.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    return {
        column: frame.get_column(column).null_count() / frame.height
        if frame.height
        else 1.0
        for column in columns
    }


def audit_frames(daily: pl.DataFrame, minute: pl.DataFrame) -> dict[str, Any]:
    daily_keys = daily.select("symbol", "date").n_unique()
    minute_with_date = minute.with_columns(
        pl.col("datetime").dt.date().alias("date")
    )
    minute_keys = minute.select("symbol", "datetime").n_unique()
    aggregated = minute_with_date.group_by("symbol", "date").agg(
        pl.col("volume_hands").sum().alias("minute_volume_hands"),
        pl.col("amount_cny").sum().alias("minute_amount_cny"),
        pl.len().alias("minute_rows"),
        pl.col("datetime").min().alias("first_time"),
        pl.col("datetime").max().alias("last_time"),
    )
    reconciliation = daily.select(
        "symbol", "date", "volume_hands", "amount_cny"
    ).join(aggregated, on=["symbol", "date"], how="left")
    traded = reconciliation.filter(
        (pl.col("volume_hands") > 0) | (pl.col("amount_cny") > 0)
    ).with_columns(
        (
            (pl.col("minute_volume_hands") - pl.col("volume_hands")).abs()
            / pl.col("volume_hands").clip(lower_bound=1.0)
        ).alias("volume_relative_error"),
        (
            (pl.col("minute_amount_cny") - pl.col("amount_cny")).abs()
            / pl.col("amount_cny").clip(lower_bound=1.0)
        ).alias("amount_relative_error"),
    )
    missing_minute = traded.filter(pl.col("minute_rows").is_null())
    matched = traded.filter(pl.col("minute_rows").is_not_null())
    max_volume_error = (
        matched.get_column("volume_relative_error").max() if matched.height else None
    )
    max_amount_error = (
        matched.get_column("amount_relative_error").max() if matched.height else None
    )
    session = aggregated.select(
        pl.col("minute_rows").min().alias("min_rows"),
        pl.col("minute_rows").median().alias("median_rows"),
        pl.col("minute_rows").max().alias("max_rows"),
        (pl.col("minute_rows") == 241).sum().alias("complete_241_sessions"),
        pl.len().alias("symbol_sessions"),
    ).to_dicts()[0]
    daily_dates = set(daily.get_column("date").unique().to_list())
    minute_dates = set(minute_with_date.get_column("date").unique().to_list())
    checks = {
        "daily_keys_unique": daily.height == daily_keys,
        "minute_keys_unique": minute.height == minute_keys,
        "daily_critical_complete": all(
            value == 0.0 for value in _null_rates(daily, CRITICAL_DAILY).values()
        ),
        "minute_critical_complete": all(
            value == 0.0 for value in _null_rates(minute, CRITICAL_MINUTE).values()
        ),
        "date_sets_equal": daily_dates == minute_dates,
        "all_traded_daily_rows_have_minute": missing_minute.is_empty(),
        "volume_reconciles": (
            max_volume_error is not None and max_volume_error <= 1e-9
        ),
        "amount_reconciles": (
            max_amount_error is not None and max_amount_error <= 1e-6
        ),
    }
    return {
        "daily": {
            "rows": daily.height,
            "symbols": daily.get_column("symbol").n_unique(),
            "dates": len(daily_dates),
            "first_date": min(daily_dates),
            "last_date": max(daily_dates),
            "duplicate_keys": daily.height - daily_keys,
            "null_rates": _null_rates(daily, CRITICAL_DAILY),
        },
        "minute": {
            "rows": minute.height,
            "symbols": minute.get_column("symbol").n_unique(),
            "dates": len(minute_dates),
            "first_date": min(minute_dates),
            "last_date": max(minute_dates),
            "duplicate_keys": minute.height - minute_keys,
            "null_rates": _null_rates(minute, CRITICAL_MINUTE),
            "session_rows": session,
        },
        "reconciliation": {
            "daily_symbol_sessions": reconciliation.height,
            "traded_daily_symbol_sessions": traded.height,
            "matched_traded_symbol_sessions": matched.height,
            "missing_minute_symbol_sessions": missing_minute.select(
                "symbol", "date", "volume_hands", "amount_cny"
            ).to_dicts(),
            "max_volume_relative_error": max_volume_error,
            "max_amount_relative_error": max_amount_error,
        },
        "decision": {
            "passed": all(checks.values()),
            "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed],
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, start: date, end: date, output: Path) -> dict[str, Any]:
    root = data_dir / "convertible_bond"
    daily_paths = _paths(root, "daily", start, end)
    minute_paths = _paths(root, "minute", start, end)
    if not daily_paths or not minute_paths:
        raise ValueError("daily and minute convertible-bond partitions are required")
    daily = pl.read_parquet(daily_paths).sort(["date", "symbol"])
    minute = pl.read_parquet(minute_paths).sort(["datetime", "symbol"])
    payload = {
        "schema_version": "p0-convertible-bond-storage-audit-v1",
        "start": start,
        "end": end,
        **audit_frames(daily, minute),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
