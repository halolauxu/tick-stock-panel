"""Run the frozen development-only Dragon-Tiger institutional-flow study."""
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
PANEL_END = date(2021, 2, 28)
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 20

CATEGORIES = (
    "institution_buy",
    "northbound_buy",
    "institution_sell_control",
    "northbound_sell_control",
)
POSITIVE_CATEGORIES = CATEGORIES[:2]


def load_top_inst(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "top_inst").glob(
        "year=*/quarter=*/part.parquet"
    ):
        try:
            year = int(path.parents[1].name.removeprefix("year="))
        except ValueError:
            continue
        if DEVELOPMENT_START.year <= year <= DEVELOPMENT_END.year:
            paths.append(path)
    expected = (DEVELOPMENT_END.year - DEVELOPMENT_START.year + 1) * 4
    if len(paths) != expected:
        raise ValueError("all 2014-2020 top_inst quarterly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("trade_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["trade_date", "symbol", "seat_name", "reason", "side"])
    )


def deduplicate_seat_amounts(details: pl.DataFrame) -> pl.DataFrame:
    return details.unique(
        subset=[
            "trade_date",
            "symbol",
            "seat_name",
            "buy",
            "sell",
            "net_buy",
        ],
        keep="first",
        maintain_order=True,
    )


def aggregate_events(details: pl.DataFrame) -> pl.DataFrame:
    unique_details = deduplicate_seat_amounts(details)
    institution = pl.col("seat_name").str.contains("机构专用", literal=True)
    northbound = pl.col("seat_name").str.contains(r"(?:沪股通专用|深股通专用)")
    grouped = unique_details.group_by("symbol", "trade_date").agg(
        pl.when(institution)
        .then(pl.col("net_buy"))
        .otherwise(0.0)
        .sum()
        .alias("institution_net_buy"),
        institution.sum().alias("institution_rows"),
        pl.when(northbound)
        .then(pl.col("net_buy"))
        .otherwise(0.0)
        .sum()
        .alias("northbound_net_buy"),
        northbound.sum().alias("northbound_rows"),
    )
    institution_events = (
        grouped.filter(
            (pl.col("institution_rows") > 0)
            & (pl.col("institution_net_buy") != 0)
        )
        .with_columns(
            pl.when(pl.col("institution_net_buy") > 0)
            .then(pl.lit("institution_buy"))
            .otherwise(pl.lit("institution_sell_control"))
            .alias("category"),
            pl.col("institution_net_buy").alias("category_net_buy"),
            pl.col("institution_rows").alias("category_seat_rows"),
        )
    )
    northbound_events = (
        grouped.filter(
            (pl.col("northbound_rows") > 0) & (pl.col("northbound_net_buy") != 0)
        )
        .with_columns(
            pl.when(pl.col("northbound_net_buy") > 0)
            .then(pl.lit("northbound_buy"))
            .otherwise(pl.lit("northbound_sell_control"))
            .alias("category"),
            pl.col("northbound_net_buy").alias("category_net_buy"),
            pl.col("northbound_rows").alias("category_seat_rows"),
        )
    )
    combined = (
        pl.concat([institution_events, northbound_events], how="diagonal_relaxed")
        .rename({"trade_date": "ann_date"})
        .sort(["symbol", "category", "ann_date"])
    )
    last_kept: dict[tuple[str, str], date] = {}
    keep = []
    for row in combined.iter_rows(named=True):
        key = (row["symbol"], row["category"])
        event_date = row["ann_date"]
        previous = last_kept.get(key)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[key] = event_date
    return combined.filter(pl.Series("_keep", keep)).sort(
        ["ann_date", "symbol", "category"]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_details = load_top_inst(data_dir)
    unique_details = deduplicate_seat_amounts(raw_details)
    events = aggregate_events(raw_details)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
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
        "schema_version": "p0-top-inst-flow-development-v1",
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
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 5-day median return",
        },
        "data": {
            "raw_seat_rows": raw_details.height,
            "unique_seat_amount_rows": unique_details.height,
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
                else "terminate_top_inst_flow_mechanism"
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
        default=Path("/app/data/research/p0_top_inst_flow_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
