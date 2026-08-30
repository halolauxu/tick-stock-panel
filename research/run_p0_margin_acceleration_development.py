"""Run the frozen development-only margin-acceleration event study."""
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
from research.run_p0_large_order_flow_development import (  # noqa: E402
    MAX_ABS_EVENT_RETURN,
    MAX_EXIT_DELAY,
    MIN_ABS_FLOW,
    MIN_DAILY_AMOUNT,
    _promotion,
    build_event_day_panel,
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
HOLD_TRADING_DAYS = 5
COOLDOWN_DAYS = 20
MIN_BALANCE_GROWTH = 0.05
MIN_BUY_INTENSITY = 0.10

CANDIDATES = ("margin_price_divergence", "margin_price_continuation")
CONTROL = "deleverage_control"
CATEGORIES = (*CANDIDATES, CONTROL)


def load_margin_detail(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "margin_detail").glob(
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
        raise ValueError("all 2014-2020 margin-detail yearly partitions are required")
    frame = (
        pl.read_parquet(sorted(paths), hive_partitioning=False)
        .filter(
            pl.col("trade_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )
    calendar = frame.select("trade_date").unique().sort("trade_date").with_row_index(
        "margin_trade_index"
    )
    return (
        frame.join(calendar, on="trade_date", how="left")
        .sort(["symbol", "trade_date"])
        .with_columns(
            pl.col("rzye").shift(1).over("symbol").alias("previous_rzye"),
            pl.col("margin_trade_index")
            .shift(1)
            .over("symbol")
            .alias("previous_margin_trade_index"),
        )
        .with_columns(
            (pl.col("rzye") / pl.col("previous_rzye") - 1.0).alias(
                "margin_balance_change"
            ),
            (
                pl.col("margin_trade_index")
                == pl.col("previous_margin_trade_index") + 1
            ).alias("margin_dates_adjacent"),
        )
        .sort(["trade_date", "symbol"])
    )


def categorize_events(
    margin: pl.DataFrame, event_day_panel: pl.DataFrame
) -> pl.DataFrame:
    work = (
        margin.join(event_day_panel, on=["symbol", "trade_date"], how="left")
        .with_columns(
            (pl.col("rzmre") / pl.col("event_daily_amount")).alias(
                "margin_buy_intensity"
            )
        )
        .filter(
            (pl.col("previous_rzye") > 0)
            & pl.col("margin_dates_adjacent").fill_null(False)
            & (pl.col("event_daily_amount") >= MIN_DAILY_AMOUNT)
            & pl.col("event_return").is_between(
                -MAX_ABS_EVENT_RETURN, MAX_ABS_EVENT_RETURN, closed="both"
            )
        )
        .with_columns(
            pl.when(
                (pl.col("margin_balance_change") >= MIN_BALANCE_GROWTH)
                & (pl.col("rzmre") >= MIN_ABS_FLOW)
                & (pl.col("margin_buy_intensity") >= MIN_BUY_INTENSITY)
                & (pl.col("event_return") <= 0)
            )
            .then(pl.lit("margin_price_divergence"))
            .when(
                (pl.col("margin_balance_change") >= MIN_BALANCE_GROWTH)
                & (pl.col("rzmre") >= MIN_ABS_FLOW)
                & (pl.col("margin_buy_intensity") >= MIN_BUY_INTENSITY)
                & (pl.col("event_return") > 0)
            )
            .then(pl.lit("margin_price_continuation"))
            .when(pl.col("margin_balance_change") <= -MIN_BALANCE_GROWTH)
            .then(pl.lit(CONTROL))
            .otherwise(None)
            .alias("category")
        )
        .filter(pl.col("category").is_not_null())
        .rename({"trade_date": "ann_date"})
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep = []
    for row in work.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    return work.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_margin = load_margin_detail(data_dir)
    raw_panel = load_panel(data_dir, START, PANEL_END)
    events = categorize_events(raw_margin, build_event_day_panel(raw_panel))
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
            positive_categories=CANDIDATES,
            min_tradable_events=500,
            min_announcement_days=200,
        )
        for category in CATEGORIES
    }
    control_mean = summaries[CONTROL]["mean_excess_return"]
    for candidate in CANDIDATES:
        candidate_mean = summaries[candidate]["mean_excess_return"]
        direction_specific = (
            candidate_mean is not None
            and control_mean is not None
            and candidate_mean > control_mean
        )
        summaries[candidate]["direction_specific_vs_deleverage"] = direction_specific
        summaries[candidate]["promotion_passed"] = _promotion(
            summaries[candidate], direction_specific
        )
    promoted = [
        candidate for candidate in CANDIDATES if summaries[candidate]["promotion_passed"]
    ]
    selected = (
        max(promoted, key=lambda name: summaries[name]["excess_daily_cluster_t"])
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-margin-acceleration-development-v1",
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
            "minimum_balance_growth": MIN_BALANCE_GROWTH,
            "minimum_buy_intensity": MIN_BUY_INTENSITY,
            "minimum_daily_amount_cny": MIN_DAILY_AMOUNT,
            "minimum_financing_buy_cny": MIN_ABS_FLOW,
            "maximum_absolute_event_return": MAX_ABS_EVENT_RETURN,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 5-day median return",
        },
        "data": {
            "raw_margin_rows": raw_margin.height,
            "categorized_events": events.height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "categories": summaries,
        "decision": {
            "promoted_candidates": promoted,
            "selected_candidate": selected,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_validation"
                if selected
                else "terminate_margin_acceleration_mechanism"
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
        default=Path("/app/data/research/p0_margin_acceleration_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
