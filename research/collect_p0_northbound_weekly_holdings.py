"""Collect and audit frozen weekly northbound stock-holding snapshots."""
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

START = date(2017, 3, 17)
END = date(2024, 8, 16)
FIELDS = ("code", "trade_date", "ts_code", "name", "vol", "ratio", "exchange")


def weekly_candidate_dates(data_dir: Path) -> list[list[date]]:
    dates = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if START <= day <= END:
            dates.append(day)
    if not dates:
        raise ValueError("daily partitions are required to build weekly dates")
    weeks: dict[tuple[int, int], list[date]] = {}
    for day in sorted(set(dates)):
        iso = day.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(day)
    return [sorted(weeks[key], reverse=True) for key in sorted(weeks)]


def weekly_dates(data_dir: Path) -> list[date]:
    return [candidates[0] for candidates in weekly_candidate_dates(data_dir)]


def normalize_holdings(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "trade_date": "date",
                "ts_code": "symbol",
                "vol": "holding_shares",
                "ratio": "holding_ratio_pct",
            }
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("holding_shares").cast(pl.Float64, strict=False),
            pl.col("holding_ratio_pct").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("exchange").is_in(["SH", "SZ"])
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("date").is_between(START, END, closed="both")
        )
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["date", "symbol"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    candidate_groups = weekly_candidate_dates(data_dir)
    planned_dates = [candidates[0] for candidates in candidate_groups]
    target = data_dir / "research" / "northbound_weekly_holdings.parquet"
    existing = pl.read_parquet(target) if target.is_file() else pl.DataFrame()
    covered_weeks = {
        (day.isocalendar().year, day.isocalendar().week)
        for day in existing["date"].unique().to_list()
    } if not existing.is_empty() else set()
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    new_frames: list[pl.DataFrame] = []
    client = TushareClient(token)
    try:
        for index, candidates in enumerate(candidate_groups, start=1):
            iso = candidates[0].isocalendar()
            week_key = (iso.year, iso.week)
            if week_key not in covered_weeks:
                for day in candidates:
                    fetched = normalize_holdings(
                        client.query(
                            "hk_hold",
                            {"trade_date": day.strftime("%Y%m%d")},
                            FIELDS,
                        )
                    )
                    if not fetched.is_empty():
                        new_frames.append(fetched)
                        covered_weeks.add(week_key)
                        break
            if index == 1 or index % 25 == 0 or index == len(candidate_groups):
                print(
                    f"northbound_progress={index}/{len(candidate_groups)} "
                    f"covered_weeks={len(covered_weeks)}",
                    flush=True,
                )
    finally:
        client.close()

    frames = ([existing] if not existing.is_empty() else []) + new_frames
    holdings = (
        pl.concat(frames, how="diagonal_relaxed")
        .unique(["symbol", "date"], keep="last")
        .sort(["date", "symbol"])
    )
    counts = holdings.group_by("date").agg(pl.len().alias("symbols")).sort("date")
    returned_weeks = {
        (day.isocalendar().year, day.isocalendar().week)
        for day in holdings["date"].unique().to_list()
    }
    planned_weeks = {
        (day.isocalendar().year, day.isocalendar().week) for day in planned_dates
    }
    missing_dates = [
        candidates[0]
        for candidates in candidate_groups
        if (
            candidates[0].isocalendar().year,
            candidates[0].isocalendar().week,
        )
        not in returned_weeks
    ]
    invalid_shares = holdings.filter(
        pl.col("holding_shares").is_null()
        | (pl.col("holding_shares") < 0)
    ).height
    missing_ratios = holdings.filter(pl.col("holding_ratio_pct").is_null()).height
    invalid_ratios = holdings.filter(pl.col("holding_ratio_pct") < 0).height
    ratio_missing_rate = missing_ratios / holdings.height
    minimum_week_symbols = counts["symbols"].min() if counts.height else 0
    date_coverage = len(returned_weeks) / len(planned_weeks)
    checks = {
        "weekly_date_coverage_at_least_98pct": date_coverage >= 0.98,
        "symbol_date_unique": holdings.unique(["symbol", "date"]).height
        == holdings.height,
        "holding_shares_complete_and_nonnegative": invalid_shares == 0,
        "vendor_ratio_missing_rate_at_most_0_01pct": ratio_missing_rate <= 0.0001,
        "reported_vendor_ratios_nonnegative": invalid_ratios == 0,
        "every_returned_week_at_least_100_symbols": minimum_week_symbols >= 100,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    _atomic_parquet(holdings, target)
    payload = {
        "schema_version": "p0-northbound-weekly-holdings-v2",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": START,
            "end": END,
            "future_returns_read": False,
        },
        "status": status,
        "counts": {
            "planned_weeks": len(planned_dates),
            "returned_weeks": len(returned_weeks),
            "date_coverage": date_coverage,
            "rows": holdings.height,
            "symbols": holdings["symbol"].n_unique(),
            "minimum_week_symbols": minimum_week_symbols,
            "invalid_share_rows": invalid_shares,
            "missing_vendor_ratio_rows": missing_ratios,
            "vendor_ratio_missing_rate": ratio_missing_rate,
            "invalid_vendor_ratio_rows": invalid_ratios,
        },
        "checks": checks,
        "missing_dates": missing_dates,
        "artifact": str(target),
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
            "/app/data/research/p0_northbound_weekly_holdings_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
