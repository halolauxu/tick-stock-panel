"""Run the frozen development-only block-trade price event study."""
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
HOLD_TRADING_DAYS = 10
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 30
MIN_NOTIONAL_CNY = 10_000_000.0
MIN_DAILY_AMOUNT_SHARE = 0.05
PREMIUM_THRESHOLD = 0.01

CATEGORIES = ("premium_block", "discount_block", "near_close_control")
POSITIVE_CATEGORIES = CATEGORIES[:2]


def load_block_trades(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "block_trade").glob(
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
        raise ValueError("all 2014-2020 block-trade yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("trade_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["trade_date", "symbol", "price", "volume_shares"])
    )


def aggregate_events(
    details: pl.DataFrame,
    event_day_panel: pl.DataFrame,
) -> pl.DataFrame:
    grouped = (
        details.group_by("symbol", "trade_date")
        .agg(
            pl.col("notional_cny").sum().alias("block_notional_cny"),
            pl.col("volume_shares").sum().alias("block_volume_shares"),
            (
                (pl.col("price") * pl.col("volume_shares")).sum()
                / pl.col("volume_shares").sum()
            ).alias("block_vwap"),
            pl.len().alias("detail_rows"),
        )
        .join(
            event_day_panel.rename({"date": "trade_date"}),
            on=["symbol", "trade_date"],
            how="left",
        )
        .with_columns(
            (pl.col("block_vwap") / pl.col("event_raw_close") - 1.0).alias(
                "discount_premium"
            ),
            (pl.col("block_notional_cny") / pl.col("event_daily_amount")).alias(
                "daily_amount_share"
            ),
        )
        .filter(
            (pl.col("block_notional_cny") >= MIN_NOTIONAL_CNY)
            & (pl.col("daily_amount_share") >= MIN_DAILY_AMOUNT_SHARE)
            & pl.col("event_raw_close").is_not_null()
            & (pl.col("event_raw_close") > 0)
        )
        .with_columns(
            pl.when(pl.col("discount_premium") >= PREMIUM_THRESHOLD)
            .then(pl.lit("premium_block"))
            .when(pl.col("discount_premium") <= -PREMIUM_THRESHOLD)
            .then(pl.lit("discount_block"))
            .otherwise(pl.lit("near_close_control"))
            .alias("category")
        )
        .rename({"trade_date": "ann_date"})
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
    return grouped.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol", "category"]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_details = load_block_trades(data_dir)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    event_day_panel = panel.select(
        "symbol",
        "date",
        pl.col("raw_close").alias("event_raw_close"),
        pl.col("amount").alias("event_daily_amount"),
    )
    events = aggregate_events(raw_details, event_day_panel)
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
            min_tradable_events=500,
            min_announcement_days=200,
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
        "schema_version": "p0-block-trade-price-development-v1",
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
            "minimum_block_notional_cny": MIN_NOTIONAL_CNY,
            "minimum_daily_amount_share": MIN_DAILY_AMOUNT_SHARE,
            "premium_discount_threshold": PREMIUM_THRESHOLD,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 10-day median return",
        },
        "data": {
            "raw_block_trade_rows": raw_details.height,
            "categorized_events": events.height,
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
                else "terminate_block_trade_price_mechanism"
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
        default=Path("/app/data/research/p0_block_trade_price_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
