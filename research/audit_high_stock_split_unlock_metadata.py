"""Audit high-stock-split proposals with pre-announced unlock pressure.

This stage is metadata-only.  It deliberately does not load price, return, or
backtest data.  Its only purpose is to decide whether the proposed mechanism
has enough point-in-time events to justify freezing a development contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

DIVIDEND_START_YEAR = 2012
DIVIDEND_END_YEAR = 2020
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
UNLOCK_END_YEAR = 2021
HIGH_SPLIT_RATIO = 1.0
UPCOMING_UNLOCK_DAYS = 180
MIN_UPCOMING_UNLOCK_RATIO_PCT = 5.0
COOLDOWN_DAYS = 365
MIN_MATCHED_EVENTS = 40
MIN_SIGNAL_DAYS = 30
MIN_SYMBOLS = 30
MIN_YEARS = 3


def expected_dividend_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "dividend_announcements"
    return [
        root / f"year={year}" / f"month={month:02d}" / "part.parquet"
        for year in range(DIVIDEND_START_YEAR, DIVIDEND_END_YEAR + 1)
        for month in range(1, 13)
    ]


def expected_unlock_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "share_float"
    return [
        root / f"year={year}" / "part.parquet"
        for year in range(DEVELOPMENT_START.year, UNLOCK_END_YEAR + 1)
    ]


def _high_split_proposals(dividends: pl.DataFrame) -> pl.DataFrame:
    proposals = (
        dividends.filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & (pl.col("dividend_stage") == "预案")
            & pl.col("symbol").str.contains(
                r"^(?:(?:0|3)\d{5}\.SZ|6\d{5}\.SH)$"
            )
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("stock_dividend_per_share").fill_null(0.0),
                pl.col("bonus_share_per_share").fill_null(0.0)
                + pl.col("capitalization_share_per_share").fill_null(0.0),
            ).alias("split_ratio")
        )
        .filter(pl.col("split_ratio") >= HIGH_SPLIT_RATIO)
        .sort(["symbol", "period_end", "ann_date"])
        .unique(["symbol", "period_end"], keep="first", maintain_order=True)
        .select("symbol", "period_end", "ann_date", "split_ratio")
        .sort(["symbol", "ann_date", "period_end"])
    )
    if proposals.is_empty():
        return proposals
    keep: list[bool] = []
    last_kept: dict[str, date] = {}
    for row in proposals.iter_rows(named=True):
        previous = last_kept.get(row["symbol"])
        accepted = previous is None or (
            row["ann_date"] - previous
        ).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[row["symbol"]] = row["ann_date"]
    return proposals.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def _attach_upcoming_unlocks(
    proposals: pl.DataFrame, unlock_details: pl.DataFrame
) -> pl.DataFrame:
    if proposals.is_empty() or unlock_details.is_empty():
        return proposals.head(0).with_columns(
            pl.lit(None, dtype=pl.Date).alias("unlock_date"),
            pl.lit(None, dtype=pl.Float64).alias("unlock_ratio_pct"),
        )
    candidates = (
        proposals.rename({"ann_date": "split_ann_date"})
        .join(unlock_details, on="symbol", how="inner")
        .filter(
            (pl.col("ann_date") <= pl.col("split_ann_date"))
            & (pl.col("float_date") > pl.col("split_ann_date"))
            & (
                pl.col("float_date")
                <= pl.col("split_ann_date")
                + pl.duration(days=UPCOMING_UNLOCK_DAYS)
            )
        )
        .group_by(
            "symbol", "period_end", "split_ann_date", "split_ratio", "float_date"
        )
        .agg(
            pl.col("float_ratio").sum().alias("unlock_ratio_pct"),
            pl.col("ann_date").max().alias("latest_unlock_announcement"),
        )
        .filter(pl.col("unlock_ratio_pct") >= MIN_UPCOMING_UNLOCK_RATIO_PCT)
        .sort(
            ["symbol", "split_ann_date", "unlock_ratio_pct", "float_date"],
            descending=[False, False, True, False],
        )
        .unique(["symbol", "period_end", "split_ann_date"], keep="first")
        .rename(
            {
                "split_ann_date": "ann_date",
                "float_date": "unlock_date",
            }
        )
        .sort(["ann_date", "symbol"])
    )
    return candidates


def _yearly(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return (
        frame.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("events"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("ann_date").n_unique().alias("signal_days"),
        )
        .sort("year")
        .to_dicts()
    )


def audit(data_dir: Path) -> dict[str, Any]:
    dividend_paths = expected_dividend_paths(data_dir)
    unlock_paths = expected_unlock_paths(data_dir)
    missing_dividends = [str(path) for path in dividend_paths if not path.is_file()]
    missing_unlocks = [str(path) for path in unlock_paths if not path.is_file()]
    if missing_dividends or missing_unlocks:
        return {
            "status": "DATA_INCOMPLETE",
            "future_returns_read": False,
            "price_data_read": False,
            "missing_dividend_partitions": missing_dividends,
            "missing_unlock_partitions": missing_unlocks,
        }

    dividends = pl.read_parquet(dividend_paths, hive_partitioning=False)
    unlocks = pl.read_parquet(unlock_paths, hive_partitioning=False)
    proposals = _high_split_proposals(dividends)
    matched = _attach_upcoming_unlocks(proposals, unlocks)
    years = matched.get_column("ann_date").dt.year().n_unique() if matched.height else 0
    checks = {
        "all_required_metadata_present": True,
        "matched_events_at_least_40": matched.height >= MIN_MATCHED_EVENTS,
        "signal_days_at_least_30": (
            matched.get_column("ann_date").n_unique() >= MIN_SIGNAL_DAYS
            if matched.height
            else False
        ),
        "symbols_at_least_30": (
            matched.get_column("symbol").n_unique() >= MIN_SYMBOLS
            if matched.height
            else False
        ),
        "at_least_three_event_years": years >= MIN_YEARS,
        "unlock_announced_no_later_than_split": (
            matched.filter(
                pl.col("latest_unlock_announcement") > pl.col("ann_date")
            ).is_empty()
            if matched.height
            else True
        ),
        "unlock_after_split_within_180_days": (
            matched.filter(
                (pl.col("unlock_date") <= pl.col("ann_date"))
                | (
                    pl.col("unlock_date")
                    > pl.col("ann_date") + pl.duration(days=UPCOMING_UNLOCK_DAYS)
                )
            ).is_empty()
            if matched.height
            else True
        ),
    }
    sample_checks = [
        "matched_events_at_least_40",
        "signal_days_at_least_30",
        "symbols_at_least_30",
        "at_least_three_event_years",
    ]
    integrity_checks = [name for name in checks if name not in sample_checks]
    if not all(checks[name] for name in integrity_checks):
        status = "DATA_GAP"
    elif all(checks[name] for name in sample_checks):
        status = "SAMPLE_SUFFICIENT"
    else:
        status = "SAMPLE_SPARSE"
    return {
        "status": status,
        "future_returns_read": False,
        "price_data_read": False,
        "period": {
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "latest_required_unlock_year": UNLOCK_END_YEAR,
        },
        "assumptions": {
            "minimum_split_ratio_per_share": HIGH_SPLIT_RATIO,
            "upcoming_unlock_days": UPCOMING_UNLOCK_DAYS,
            "minimum_upcoming_unlock_ratio_pct": MIN_UPCOMING_UNLOCK_RATIO_PCT,
            "same_symbol_cooldown_days": COOLDOWN_DAYS,
        },
        "rows": {
            "dividend_metadata": dividends.height,
            "unlock_details": unlocks.height,
            "high_split_proposals": proposals.height,
            "high_split_with_upcoming_unlock": matched.height,
        },
        "coverage": {
            "matched_symbols": matched["symbol"].n_unique() if matched.height else 0,
            "matched_signal_days": (
                matched["ann_date"].n_unique() if matched.height else 0
            ),
            "matched_years": years,
            "yearly": _yearly(matched),
        },
        "checks": checks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "p0-high-stock-split-unlock-metadata-audit-v1",
        "contract_frozen": "2026-08-31",
        **audit(data_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": digest},
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_high_stock_split_unlock_metadata_audit.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
