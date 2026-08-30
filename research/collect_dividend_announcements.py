"""Collect one month of point-in-time cash-dividend announcement metadata."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from calendar import monthrange
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START_YEAR = 2012
END_YEAR = 2020
SOURCE_ROW_LIMIT = 3_000
FIELDS = (
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
)
EVENT_SCHEMA = {
    "symbol": pl.String,
    "period_end": pl.Date,
    "ann_date": pl.Date,
    "dividend_stage": pl.String,
    "stock_dividend_per_share": pl.Float64,
    "bonus_share_per_share": pl.Float64,
    "capitalization_share_per_share": pl.Float64,
    "cash_dividend_pre_tax_per_share": pl.Float64,
    "cash_dividend_after_tax_per_share": pl.Float64,
}


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def validate_period(year: int, month: int) -> None:
    if year not in range(START_YEAR, END_YEAR + 1) or month not in range(1, 13):
        raise ValueError("dividend collection must be one valid 2012-2020 month")


def normalize(rows: list[dict[str, Any]], year: int, month: int) -> pl.DataFrame:
    if not rows:
        return _empty_events()
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "symbol",
                "end_date": "period_end",
                "div_proc": "dividend_stage",
                "stk_div": "stock_dividend_per_share",
                "stk_bo_rate": "bonus_share_per_share",
                "stk_co_rate": "capitalization_share_per_share",
                "cash_div": "cash_dividend_pre_tax_per_share",
                "cash_div_tax": "cash_dividend_after_tax_per_share",
            }
        )
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("ann_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("dividend_stage").cast(pl.Utf8).str.strip_chars(),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in EVENT_SCHEMA
                if column.endswith("_per_share")
            ],
        )
        .filter(
            pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
            & (pl.col("ann_date").dt.year() == year)
            & (pl.col("ann_date").dt.month() == month)
            & pl.col("period_end").is_not_null()
        )
        .unique(
            subset=["symbol", "period_end", "ann_date", "dividend_stage"],
            keep="last",
        )
        .sort(["ann_date", "symbol", "period_end", "dividend_stage"])
    )
    return frame.select(*EVENT_SCHEMA) if not frame.is_empty() else _empty_events()


def fetch_month(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
    year: int,
    month: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    maximum_daily_rows = 0
    for day_number in range(1, monthrange(year, month)[1] + 1):
        current = date(year, month, day_number)
        daily = fetch(
            "dividend",
            {"ann_date": current.strftime("%Y%m%d")},
            FIELDS,
        )
        maximum_daily_rows = max(maximum_daily_rows, len(daily))
        if len(daily) >= SOURCE_ROW_LIMIT:
            raise ValueError(
                f"Tushare dividend response may be truncated on {current}: {len(daily)}"
            )
        rows.extend(daily)
    return rows, maximum_daily_rows


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


def collect_month(
    fetch: Callable[[str, dict[str, str], tuple[str, ...]], list[dict[str, Any]]],
    root: Path,
    year: int,
    month: int,
) -> dict[str, Any]:
    validate_period(year, month)
    rows, maximum_daily_rows = fetch_month(fetch, year, month)
    frame = normalize(rows, year, month)
    target = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
    _atomic_write(frame, target)
    return {
        "year": year,
        "month": month,
        "path": str(target),
        "raw_rows": len(rows),
        "events": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_announcement": frame.get_column("ann_date").min(),
        "last_announcement": frame.get_column("ann_date").max(),
        "maximum_daily_rows": maximum_daily_rows,
        "source_row_limit": SOURCE_ROW_LIMIT,
    }


def run(data_dir: Path, year: int, month: int) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        result = collect_month(
            client.query,
            data_dir / "event_data" / "dividend_announcements",
            year,
            month,
        )
    finally:
        client.close()
    payload = {
        "dataset": "tushare_dividend_announcement_metadata",
        "outcome_fields_persisted": False,
        "future_corporate_action_dates_persisted": False,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year, args.month)


if __name__ == "__main__":
    main()
