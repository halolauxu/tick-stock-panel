"""Run the frozen development-only shareholder-increase drift event study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.run_p0_forecast_drift_development import (  # noqa: E402
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
COOLDOWN_DAYS = 180

CATEGORIES = (
    "management_increase",
    "corporate_increase",
    "personal_increase",
    "decrease_control",
)
POSITIVE_CATEGORIES = CATEGORIES[:-1]


def load_holder_trades(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "holder_trade").glob(
        "year=*/part.parquet"
    ):
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if DEVELOPMENT_START.year <= year <= DEVELOPMENT_END.year:
            paths.append(path)
    expected = DEVELOPMENT_END.year - DEVELOPMENT_START.year + 1
    if len(paths) != expected:
        raise ValueError("all 2014-2020 holder-trade yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "direction", "holder_type"])
    )


def aggregate_events(details: pl.DataFrame) -> pl.DataFrame:
    is_increase = pl.col("direction") == "IN"
    is_decrease = pl.col("direction") == "DE"
    grouped = (
        details.group_by("symbol", "ann_date")
        .agg(
            is_increase.any().alias("has_increase"),
            is_decrease.any().alias("has_decrease"),
            (is_increase & (pl.col("holder_type") == "G"))
            .any()
            .alias("has_management_increase"),
            (is_increase & (pl.col("holder_type") == "C"))
            .any()
            .alias("has_corporate_increase"),
            (is_increase & (pl.col("holder_type") == "P"))
            .any()
            .alias("has_personal_increase"),
            pl.when(is_increase)
            .then(pl.col("change_vol").abs())
            .otherwise(0.0)
            .sum()
            .alias("increase_shares"),
            pl.when(is_increase)
            .then(pl.col("change_ratio").abs())
            .otherwise(0.0)
            .sum()
            .alias("increase_float_ratio_pct"),
            pl.when(is_increase)
            .then(pl.col("change_vol").abs() * pl.col("avg_price"))
            .otherwise(0.0)
            .sum()
            .alias("increase_notional_cny"),
            pl.len().alias("detail_rows"),
        )
        .filter(pl.col("has_increase") != pl.col("has_decrease"))
        .with_columns(
            pl.when(pl.col("has_increase") & pl.col("has_management_increase"))
            .then(pl.lit("management_increase"))
            .when(pl.col("has_increase") & pl.col("has_corporate_increase"))
            .then(pl.lit("corporate_increase"))
            .when(pl.col("has_increase") & pl.col("has_personal_increase"))
            .then(pl.lit("personal_increase"))
            .when(pl.col("has_decrease"))
            .then(pl.lit("decrease_control"))
            .otherwise(None)
            .alias("category")
        )
        .filter(pl.col("category").is_not_null())
        .sort(["symbol", "category", "ann_date"])
    )
    last_kept: dict[tuple[str, str], date] = {}
    keep = []
    for row in grouped.iter_rows(named=True):
        key = (row["symbol"], row["category"])
        event_date = row["ann_date"]
        previous = last_kept.get(key)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[key] = event_date
    return grouped.filter(pl.Series("_keep", keep)).sort(
        ["ann_date", "symbol", "category"]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_details = load_holder_trades(data_dir)
    events = aggregate_events(raw_details)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    trades = build_trades(events, panel, HOLD_TRADING_DAYS)
    benchmark = build_market_benchmark(panel)
    trades = attach_market_excess(trades, benchmark)
    summaries = {
        category: summarize_category(
            trades, category, positive_categories=POSITIVE_CATEGORIES
        )
        for category in CATEGORIES
    }
    promoted = [
        category
        for category in POSITIVE_CATEGORIES
        if summaries[category]["promotion_passed"]
    ]
    selected = (
        max(
            promoted,
            key=lambda category: summaries[category]["excess_daily_cluster_t"],
        )
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-holder-increase-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 20-day median return",
        },
        "data": {
            "raw_holder_trade_rows": raw_details.height,
            "aggregated_unique_events": events.height,
            "mixed_direction_days_excluded": raw_details.select(
                "symbol", "ann_date", "direction"
            )
            .unique()
            .group_by("symbol", "ann_date")
            .agg(pl.col("direction").n_unique().alias("directions"))
            .filter(pl.col("directions") > 1)
            .height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "categories": summaries,
        "decision": {
            "promoted_categories": promoted,
            "selected_candidate": selected,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_validation"
                if selected
                else "terminate_holder_increase_mechanism"
            ),
        },
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_holder_increase_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
