"""Collect the frozen monthly warmup panel for return seasonality."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

START_DATE = date(2007, 12, 1)
END_DATE = date(2013, 8, 31)
CALENDAR_FIELDS = ("exchange", "cal_date", "is_open", "pretrade_date")
MONTHLY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "pct_chg",
)
FACTOR_FIELDS = ("ts_code", "trade_date", "adj_factor")
A_SHARE_PATTERN = r"^((60|68)\d{4}\.SH|(00|30)\d{4}\.SZ|[489]\d{5}\.BJ)$"
_A_SHARE_RE = re.compile(A_SHARE_PATTERN)


def _atomic_write(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def expected_months(start: date = START_DATE, end: date = END_DATE) -> list[str]:
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def complete_factor_rows(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
    monthly_rows: list[dict[str, Any]],
    exact_factor_rows: list[dict[str, Any]],
    month_end: date,
) -> tuple[list[dict[str, Any]], int]:
    monthly_symbols = {
        str(row.get("ts_code") or "").strip()
        for row in monthly_rows
        if _A_SHARE_RE.fullmatch(str(row.get("ts_code") or "").strip())
    }
    exact_symbols = {str(row.get("ts_code") or "").strip() for row in exact_factor_rows}
    completed = list(exact_factor_rows)
    fallback_count = 0
    for symbol in sorted(monthly_symbols - exact_symbols):
        rows = fetch(
            "adj_factor",
            {
                "ts_code": symbol,
                "start_date": date(month_end.year - 1, month_end.month, 1).strftime(
                    "%Y%m%d"
                ),
                "end_date": month_end.strftime("%Y%m%d"),
            },
            FACTOR_FIELDS,
        )
        valid = [
            row
            for row in rows
            if str(row.get("trade_date") or "") <= month_end.strftime("%Y%m%d")
            and row.get("adj_factor") is not None
        ]
        if not valid:
            continue
        completed.append(max(valid, key=lambda row: str(row["trade_date"])))
        fallback_count += 1
    return completed, fallback_count


def month_end_trading_dates(rows: list[dict[str, Any]]) -> list[date]:
    open_dates = sorted(
        datetime.strptime(str(row["cal_date"]), "%Y%m%d").date()
        for row in rows
        if int(row.get("is_open") or 0) == 1
    )
    by_month: dict[str, date] = {}
    for value in open_dates:
        if START_DATE <= value <= END_DATE:
            by_month[value.strftime("%Y-%m")] = value
    missing = sorted(set(expected_months()) - set(by_month))
    if missing:
        raise ValueError(f"trading calendar is missing months: {missing}")
    return [by_month[month] for month in expected_months()]


def normalize_month(
    monthly_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    month_end: date,
    fallback_factors: int = 0,
) -> tuple[pl.DataFrame, dict[str, int]]:
    monthly = pl.DataFrame(monthly_rows, infer_schema_length=None)
    factors = pl.DataFrame(factor_rows, infer_schema_length=None)
    if monthly.is_empty():
        raise ValueError(f"monthly endpoint returned no rows for {month_end}")
    if factors.is_empty():
        raise ValueError(f"adjustment factors returned no rows for {month_end}")
    numeric = ("open", "high", "low", "close", "vol", "amount", "pct_chg")
    monthly = monthly.with_columns(
        pl.col("ts_code").cast(pl.String).str.strip_chars().alias("symbol"),
        pl.col("trade_date").cast(pl.String).str.to_date("%Y%m%d").alias("month_end"),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in numeric),
    )
    excluded_non_a_rows = monthly.filter(
        ~pl.col("symbol").str.contains(A_SHARE_PATTERN)
    ).height
    monthly = monthly.filter(pl.col("symbol").str.contains(A_SHARE_PATTERN))
    factors = factors.with_columns(
        pl.col("ts_code").cast(pl.String).str.strip_chars().alias("symbol"),
        pl.col("trade_date")
        .cast(pl.String)
        .str.to_date("%Y%m%d")
        .alias("adj_factor_date"),
        pl.col("adj_factor").cast(pl.Float64, strict=False),
    )
    factors = factors.filter(pl.col("symbol").str.contains(A_SHARE_PATTERN))
    duplicate_factors = (
        factors.height - factors.unique(subset=["symbol", "adj_factor_date"]).height
    )
    factors = (
        factors.sort(["symbol", "adj_factor_date"])
        .unique(subset=["symbol"], keep="last")
        .with_columns(pl.lit(month_end).alias("month_end"))
        .select("symbol", "month_end", "adj_factor_date", "adj_factor")
    )
    duplicate_prices = (
        monthly.height - monthly.unique(subset=["symbol", "month_end"]).height
    )
    if duplicate_prices or duplicate_factors:
        raise ValueError(f"duplicate monthly keys for {month_end}")
    joined = monthly.join(
        factors, on=["symbol", "month_end"], how="left", validate="1:1"
    )
    missing_factors = joined.filter(pl.col("adj_factor").is_null()).height
    invalid_dates = joined.filter(pl.col("month_end") != pl.lit(month_end)).height
    invalid_symbols = joined.filter(
        ~pl.col("symbol").str.contains(r"^\d{6}\.(SH|SZ|BJ)$")
    ).height
    future_factors = joined.filter(
        pl.col("adj_factor_date") > pl.col("month_end")
    ).height
    invalid_ohlc = joined.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        | (pl.col("adj_factor") <= 0)
    ).height
    audit = {
        "duplicate_prices": duplicate_prices,
        "duplicate_factors": duplicate_factors,
        "missing_factors": missing_factors,
        "invalid_dates": invalid_dates,
        "invalid_symbols": invalid_symbols,
        "invalid_ohlc": invalid_ohlc,
        "future_factors": future_factors,
        "excluded_non_a_rows": excluded_non_a_rows,
        "fallback_factors": fallback_factors,
        "maximum_factor_lag_days": (
            joined.select(
                (pl.col("month_end") - pl.col("adj_factor_date")).dt.total_days().max()
            ).item()
            or 0
        ),
    }
    informational = {
        "excluded_non_a_rows",
        "fallback_factors",
        "maximum_factor_lag_days",
    }
    failure_fields = {
        key: value for key, value in audit.items() if key not in informational
    }
    if any(failure_fields.values()):
        raise ValueError(f"monthly data failed audit for {month_end}: {audit}")
    return (
        joined.select(
            "symbol",
            "month_end",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "pct_chg",
            "adj_factor",
            "adj_factor_date",
            (pl.col("month_end") - pl.col("adj_factor_date"))
            .dt.total_days()
            .alias("adj_factor_lag_days"),
            (pl.col("close") * pl.col("adj_factor")).alias("adjusted_close"),
            pl.lit("tushare_monthly_adj_factor").alias("source"),
        ).sort(["month_end", "symbol"]),
        audit,
    )


def collect(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    calendar = fetch(
        "trade_cal",
        {
            "exchange": "SSE",
            "start_date": START_DATE.strftime("%Y%m%d"),
            "end_date": END_DATE.strftime("%Y%m%d"),
        },
        CALENDAR_FIELDS,
    )
    frames = []
    audits = []
    for month_end in month_end_trading_dates(calendar):
        day = month_end.strftime("%Y%m%d")
        monthly_rows = fetch("monthly", {"trade_date": day}, MONTHLY_FIELDS)
        exact_factor_rows = fetch("adj_factor", {"trade_date": day}, FACTOR_FIELDS)
        factor_rows, fallback_count = complete_factor_rows(
            fetch, monthly_rows, exact_factor_rows, month_end
        )
        frame, audit = normalize_month(
            monthly_rows, factor_rows, month_end, fallback_count
        )
        frames.append(frame)
        audits.append(
            {
                "month": month_end.strftime("%Y-%m"),
                "month_end": month_end,
                "rows": frame.height,
                "symbols": frame.get_column("symbol").n_unique(),
                **audit,
            }
        )
    return pl.concat(frames).sort(["month_end", "symbol"]), audits


def run(data_dir: Path, output: Path, audit_output: Path) -> dict[str, Any]:
    token = get_api_key()
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        frame, monthly_audits = collect(client.query)
    finally:
        client.close()
    duplicate_keys = frame.height - frame.unique(subset=["symbol", "month_end"]).height
    observed_months = frame.get_column("month_end").dt.strftime("%Y-%m").n_unique()
    if observed_months != len(expected_months()) or duplicate_keys:
        raise ValueError("warmup panel failed final continuity audit")
    _atomic_write(frame, output)
    data_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    payload = {
        "schema_version": "p0-return-seasonality-warmup-v1",
        "source": "tushare_monthly_and_adj_factor",
        "outcome_strategy_read": False,
        "period": {"start": START_DATE, "end": END_DATE},
        "data": {
            "rows": frame.height,
            "symbols": frame.get_column("symbol").n_unique(),
            "months": observed_months,
            "first_month_end": frame.get_column("month_end").min(),
            "last_month_end": frame.get_column("month_end").max(),
            "duplicate_keys": duplicate_keys,
            "monthly_audits": monthly_audits,
        },
        "output": str(output),
        "data_sha256": data_sha,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    audit_sha = hashlib.sha256(audit_output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "audit_output": str(audit_output), "audit_sha256": audit_sha},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/return_seasonality_warmup/monthly.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("/app/data/research/p0_return_seasonality_warmup_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output, args.audit_output)


if __name__ == "__main__":
    main()
