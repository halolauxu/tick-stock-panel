"""Run the frozen CB call-condition approach development event study."""

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
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH))

import run_p0_convertible_bond_momentum_development as cb_momentum  # noqa: E402
from research.run_p0_forecast_drift_development import (  # noqa: E402
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_large_order_flow_development import (  # noqa: E402
    MAX_EXIT_DELAY,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

CONTEXT_START = date(2016, 1, 1)
DEVELOPMENT_START = date(2017, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
CALL_WINDOW_DAYS = 30
ENTRY_COUNT = 10
CALL_COUNT = 15
MAX_HOLD_DAYS = 20
COOLDOWN_DAYS = 90
CALL_VALUE_THRESHOLD = 130.0
CATEGORY = "cb_call_condition_approach"


def prepare_call_panel(daily: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "global_index"
    )
    return (
        daily.join(
            master.select(
                pl.col("symbol").alias("bond_symbol"),
                pl.col("stk_code").alias("stock_symbol"),
            ),
            left_on="symbol",
            right_on="bond_symbol",
            how="inner",
        )
        .filter(
            pl.col("symbol")
            .str.slice(0, 3)
            .is_in(cb_momentum.ORDINARY_CB_PREFIXES)
            & pl.col("stock_symbol").is_not_null()
        )
        .join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("global_index")
            .shift(CALL_WINDOW_DAYS - 1)
            .over("symbol")
            .alias("window_start_index"),
            (pl.col("cb_value") >= CALL_VALUE_THRESHOLD)
            .cast(pl.Int64)
            .rolling_sum(
                window_size=CALL_WINDOW_DAYS,
                min_samples=CALL_WINDOW_DAYS,
            )
            .over("symbol")
            .alias("call_count_30"),
        )
        .with_columns(
            pl.when(
                pl.col("global_index")
                == pl.col("window_start_index") + CALL_WINDOW_DAYS - 1
            )
            .then(pl.col("call_count_30"))
            .otherwise(None)
            .alias("call_count_30")
        )
        .select(
            pl.col("symbol").alias("bond_symbol"),
            "stock_symbol",
            "date",
            "global_index",
            "call_count_30",
        )
    )


def build_events(call_panel: pl.DataFrame) -> pl.DataFrame:
    work = (
        call_panel.sort(["bond_symbol", "date"])
        .with_columns(
            pl.col("call_count_30")
            .shift(1)
            .over("bond_symbol")
            .alias("previous_call_count")
        )
        .filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END))
    )
    panel_by_bond = {
        key[0] if isinstance(key, tuple) else key: group
        for key, group in call_panel.partition_by("bond_symbol", as_dict=True).items()
    }
    raw_signals = work.filter(
        (pl.col("call_count_30") >= ENTRY_COUNT)
        & (pl.col("call_count_30") < CALL_COUNT)
        & pl.col("previous_call_count").is_not_null()
        & (pl.col("previous_call_count") < ENTRY_COUNT)
    ).sort(["stock_symbol", "date", "bond_symbol"])
    last_kept: dict[str, date] = {}
    events: list[dict[str, Any]] = []
    for row in raw_signals.iter_rows(named=True):
        stock = row["stock_symbol"]
        signal_date = row["date"]
        previous = last_kept.get(stock)
        if previous is not None and (signal_date - previous).days < COOLDOWN_DAYS:
            continue
        signal_index = int(row["global_index"])
        future = panel_by_bond[row["bond_symbol"]].filter(
            (pl.col("global_index") > signal_index)
            & (pl.col("global_index") <= signal_index + MAX_HOLD_DAYS)
            & (pl.col("call_count_30") >= CALL_COUNT)
        )
        hold_days = (
            int(future["global_index"].min()) - signal_index
            if future.height
            else MAX_HOLD_DAYS
        )
        events.append(
            {
                "symbol": stock,
                "bond_symbol": row["bond_symbol"],
                "ann_date": signal_date,
                "entry_call_count": int(row["call_count_30"]),
                "holding_trading_days": hold_days,
                "category": CATEGORY,
            }
        )
        last_kept[stock] = signal_date
    return (
        pl.DataFrame(events, infer_schema_length=None).sort(["ann_date", "symbol"])
        if events
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "bond_symbol": pl.String,
                "ann_date": pl.Date,
                "entry_call_count": pl.Int64,
                "holding_trading_days": pl.Int64,
                "category": pl.String,
            }
        )
    )


def build_event_trades(events: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    groups = []
    for hold_days in range(1, MAX_HOLD_DAYS + 1):
        scoped = events.filter(pl.col("holding_trading_days") == hold_days)
        if scoped.is_empty():
            continue
        trades = build_trades(
            scoped,
            panel,
            holding_trading_days=hold_days,
            max_exit_delay=MAX_EXIT_DELAY,
        )
        benchmark = build_market_benchmark(panel, hold_days)
        groups.append(attach_market_excess(trades, benchmark))
    if not groups:
        raise ValueError("call-condition approach produced no event trades")
    return pl.concat(groups, how="diagonal_relaxed").sort(["ann_date", "symbol"])


def promotion_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["tradable_events"] >= 80
        and metrics["announcement_days"] >= 60
        and metrics["tradable_rate"] >= 0.90
        and metrics["benchmark_coverage"] >= 0.99
        and metrics["entry_capacity_feasible_rate"] >= 0.95
        and metrics["unresolved_exits"] == 0
        and (metrics["mean_net_return"] or -math.inf) >= 0.03
        and (metrics["mean_excess_return"] or -math.inf) >= 0.02
        and (metrics["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and metrics["positive_excess_years"] >= 3
        and (metrics["max_year_positive_excess_share"] or math.inf) <= 0.50
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    cb_root = data_dir / "research" / "convertible_bond"
    master = pl.read_parquet(cb_root / "master.parquet")
    daily = pl.read_parquet(cb_root / "daily.parquet")
    call_panel = prepare_call_panel(daily, master)
    events = build_events(call_panel)
    panel = prepare_panel(load_panel(data_dir, CONTEXT_START, PANEL_END))
    trades = build_event_trades(events, panel)
    metrics = summarize_category(
        trades,
        CATEGORY,
        positive_categories=(CATEGORY,),
        min_tradable_events=80,
        min_announcement_days=60,
    )
    metrics["promotion_passed"] = promotion_passed(metrics)
    passed = metrics["promotion_passed"]
    hold_distribution = (
        events.group_by("holding_trading_days")
        .len()
        .sort("holding_trading_days")
        .to_dicts()
    )
    payload = {
        "schema_version": "p0-cb-call-condition-approach-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "call_window_trading_days": CALL_WINDOW_DAYS,
            "entry_count": ENTRY_COUNT,
            "call_count": CALL_COUNT,
            "call_value_threshold": CALL_VALUE_THRESHOLD,
            "maximum_holding_trading_days": MAX_HOLD_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation": DAILY_PARTICIPATION,
            "max_exit_delay": MAX_EXIT_DELAY,
        },
        "data": {
            "ordinary_bonds": call_panel["bond_symbol"].n_unique(),
            "events": events.height,
            "event_stocks": events["symbol"].n_unique(),
            "signal_days": events["ann_date"].n_unique(),
            "holding_days_distribution": hold_distribution,
            "panel_rows": panel.height,
        },
        "metrics": metrics,
        "decision": {
            "development_passed": passed,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_rule_before_independent_validation"
                if passed
                else "terminate_cb_call_condition_approach"
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
            "/app/data/research/p0_cb_call_condition_approach_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
