"""Run the frozen development-only large-order money-flow event study."""
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
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 20
MIN_DAILY_AMOUNT = 50_000_000.0
MIN_ABS_FLOW = 10_000_000.0
FLOW_RATIO = 0.10
MAX_ABS_EVENT_RETURN = 0.05

CANDIDATES = ("flow_price_divergence", "flow_price_continuation")
CONTROL = "large_outflow_control"
CATEGORIES = (*CANDIDATES, CONTROL)


def load_moneyflow(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "moneyflow").glob(
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
        raise ValueError("all 2014-2020 moneyflow yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths), hive_partitioning=False)
        .filter(
            pl.col("trade_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["trade_date", "symbol"])
    )


def build_event_day_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("close").shift(1).over("symbol").alias("previous_close")
        )
        .with_columns(
            (pl.col("close") / pl.col("previous_close") - 1.0).alias(
                "event_return"
            )
        )
        .select(
            "symbol",
            pl.col("date").alias("trade_date"),
            pl.col("amount").alias("event_daily_amount"),
            "event_return",
        )
    )


def categorize_events(
    moneyflow: pl.DataFrame, event_day_panel: pl.DataFrame
) -> pl.DataFrame:
    large_net = (
        pl.col("buy_lg_cny").fill_null(0)
        + pl.col("buy_elg_cny").fill_null(0)
        - pl.col("sell_lg_cny").fill_null(0)
        - pl.col("sell_elg_cny").fill_null(0)
    )
    work = (
        moneyflow.join(event_day_panel, on=["symbol", "trade_date"], how="left")
        .with_columns(large_net.alias("large_net_flow_cny"))
        .with_columns(
            (pl.col("large_net_flow_cny") / pl.col("event_daily_amount")).alias(
                "large_net_flow_ratio"
            )
        )
        .filter(
            (pl.col("event_daily_amount") >= MIN_DAILY_AMOUNT)
            & (pl.col("large_net_flow_cny").abs() >= MIN_ABS_FLOW)
            & pl.col("event_return").is_between(
                -MAX_ABS_EVENT_RETURN, MAX_ABS_EVENT_RETURN, closed="both"
            )
        )
        .with_columns(
            pl.when(
                (pl.col("large_net_flow_ratio") >= FLOW_RATIO)
                & (pl.col("event_return") <= 0)
            )
            .then(pl.lit("flow_price_divergence"))
            .when(
                (pl.col("large_net_flow_ratio") >= FLOW_RATIO)
                & (pl.col("event_return") > 0)
            )
            .then(pl.lit("flow_price_continuation"))
            .when(pl.col("large_net_flow_ratio") <= -FLOW_RATIO)
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


def _promotion(metrics: dict[str, Any], direction_specific: bool) -> bool:
    return bool(
        metrics["tradable_events"] >= 500
        and metrics["announcement_days"] >= 200
        and metrics["tradable_rate"] >= 0.90
        and metrics["benchmark_coverage"] >= 0.99
        and metrics["entry_capacity_feasible_rate"] >= 0.95
        and metrics["unresolved_exits"] == 0
        and (metrics["mean_net_return"] or -math.inf) >= 0.0075
        and (metrics["mean_excess_return"] or -math.inf) >= 0.005
        and (metrics["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and metrics["positive_excess_years"] >= 5
        and (metrics["max_year_positive_excess_share"] or math.inf) <= 0.50
        and direction_specific
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_flow = load_moneyflow(data_dir)
    raw_panel = load_panel(data_dir, START, PANEL_END)
    events = categorize_events(raw_flow, build_event_day_panel(raw_panel))
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
        summaries[candidate]["direction_specific_vs_outflow"] = direction_specific
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
        "schema_version": "p0-large-order-flow-development-v1",
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
            "minimum_daily_amount_cny": MIN_DAILY_AMOUNT,
            "minimum_absolute_large_flow_cny": MIN_ABS_FLOW,
            "large_flow_ratio": FLOW_RATIO,
            "maximum_absolute_event_return": MAX_ABS_EVENT_RETURN,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 5-day median return",
        },
        "data": {
            "raw_moneyflow_rows": raw_flow.height,
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
                else "terminate_large_order_flow_mechanism"
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
        default=Path("/app/data/research/p0_large_order_flow_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
