"""Run the frozen development-only share-unlock absorption event study."""
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

START = date(2013, 10, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 10
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 180
MIN_FLOAT_RATIO_PCT = 5.0
MIN_EVENT_RETURN = 0.01
MIN_AMOUNT_MULTIPLE = 1.5

CATEGORIES = ("absorbed_unlock", "unabsorbed_control")
POSITIVE_CATEGORIES = ("absorbed_unlock",)


def load_share_float(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "share_float").glob(
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
        raise ValueError("all 2014-2020 share-float yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("float_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["float_date", "symbol", "ann_date"])
    )


def build_event_day_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("close").shift(1).over("symbol").alias("previous_close"),
            pl.col("amount")
            .shift(1)
            .rolling_median(window_size=20, min_samples=20)
            .over("symbol")
            .alias("previous_20d_median_amount"),
        )
        .with_columns(
            (pl.col("close") / pl.col("previous_close") - 1.0).alias(
                "event_return"
            ),
            (pl.col("amount") / pl.col("previous_20d_median_amount")).alias(
                "amount_multiple"
            ),
        )
        .select(
            "symbol",
            pl.col("date").alias("float_date"),
            "event_return",
            "amount_multiple",
        )
    )


def aggregate_events(
    details: pl.DataFrame, event_day_panel: pl.DataFrame
) -> pl.DataFrame:
    grouped = (
        details.group_by("symbol", "float_date")
        .agg(
            pl.col("ann_date").min().alias("first_ann_date"),
            pl.col("ann_date").max().alias("last_ann_date"),
            pl.col("float_shares").sum().alias("float_shares"),
            pl.col("float_ratio").sum().alias("float_ratio_pct"),
            pl.len().alias("detail_rows"),
        )
        .filter(
            (pl.col("float_ratio_pct") >= MIN_FLOAT_RATIO_PCT)
            & (pl.col("float_shares") > 0)
            & (pl.col("last_ann_date") <= pl.col("float_date"))
        )
        .join(event_day_panel, on=["symbol", "float_date"], how="left")
        .drop_nulls(["event_return", "amount_multiple"])
        .with_columns(
            pl.when(
                (pl.col("event_return") >= MIN_EVENT_RETURN)
                & (pl.col("amount_multiple") >= MIN_AMOUNT_MULTIPLE)
            )
            .then(pl.lit("absorbed_unlock"))
            .otherwise(pl.lit("unabsorbed_control"))
            .alias("category")
        )
        .rename({"float_date": "ann_date"})
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep = []
    for row in grouped.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    return grouped.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_details = load_share_float(data_dir)
    raw_panel = load_panel(data_dir, START, PANEL_END)
    events = aggregate_events(raw_details, build_event_day_panel(raw_panel))
    panel = prepare_panel(raw_panel)
    trades = build_trades(
        events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    summaries = {
        category: summarize_category(
            trades,
            category,
            positive_categories=POSITIVE_CATEGORIES,
            min_tradable_events=300,
            min_announcement_days=150,
        )
        for category in CATEGORIES
    }
    candidate = summaries["absorbed_unlock"]
    control = summaries["unabsorbed_control"]
    direction_specific = (
        candidate["mean_excess_return"] is not None
        and control["mean_excess_return"] is not None
        and candidate["mean_excess_return"] > control["mean_excess_return"]
    )
    candidate["direction_specific_vs_control"] = direction_specific
    candidate["promotion_passed"] = bool(
        candidate["promotion_passed"] and direction_specific
    )
    selected = "absorbed_unlock" if candidate["promotion_passed"] else None
    payload = {
        "schema_version": "p0-share-unlock-absorption-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "minimum_float_ratio_pct": MIN_FLOAT_RATIO_PCT,
            "minimum_event_return": MIN_EVENT_RETURN,
            "minimum_amount_multiple": MIN_AMOUNT_MULTIPLE,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 10-day median return",
        },
        "data": {
            "raw_share_float_rows": raw_details.height,
            "categorized_events": events.height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "categories": summaries,
        "decision": {
            "selected_candidate": selected,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_validation"
                if selected
                else "terminate_share_unlock_absorption_mechanism"
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
        default=Path(
            "/app/data/research/p0_share_unlock_absorption_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
