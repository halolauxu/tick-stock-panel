"""Run the frozen development-only ST-removal repricing event study."""
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
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
CATEGORY = "st_removal"


def load_events(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "research" / "historical_stock_names_all_a.parquet"
    if not path.is_file():
        raise ValueError("point-in-time stock name history is required")
    return build_events(pl.read_parquet(path))


def build_events(names: pl.DataFrame) -> pl.DataFrame:
    return (
        names.with_columns(
            pl.col("start_date").cast(pl.Date, strict=False).alias("ann_date"),
            pl.col("name").fill_null("").str.strip_chars().alias("name"),
        )
        .sort(["symbol", "ann_date"])
        .with_columns(
            pl.col("name").shift(1).over("symbol").alias("previous_name")
        )
        .with_columns(
            pl.col("previous_name")
            .fill_null("")
            .str.to_uppercase()
            .str.contains("ST", literal=True)
            .alias("previous_is_st"),
            pl.col("name")
            .str.to_uppercase()
            .str.contains("ST", literal=True)
            .alias("current_is_st"),
        )
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & pl.col("previous_is_st")
            & ~pl.col("current_is_st")
            & ~pl.col("name").str.contains("退", literal=True)
        )
        .with_columns(pl.lit(CATEGORY).alias("category"))
        .unique(subset=["symbol", "ann_date"], keep="first")
        .sort(["ann_date", "symbol"])
    )


def _cluster_t(frame: pl.DataFrame, column: str) -> float | None:
    daily = (
        frame.drop_nulls(column)
        .group_by("ann_date")
        .agg(pl.col(column).mean().alias("daily_return"))
    )
    if daily.height <= 1:
        return None
    mean = daily["daily_return"].mean()
    deviation = daily["daily_return"].std(ddof=1)
    if mean is None or deviation in (None, 0.0):
        return None
    return mean / (deviation / math.sqrt(daily.height))


def summarize(trades: pl.DataFrame) -> dict[str, Any]:
    eligible = trades.filter(pl.col("universe_eligible"))
    tradable = trades.filter(pl.col("tradable"))
    benchmarked = tradable.filter(pl.col("excess_return").is_not_null())
    capacity_base = eligible.filter(
        pl.col("entry_date").is_not_null()
        & (pl.col("entry_volume").fill_null(0) > 0)
        & (pl.col("entry_open").fill_null(0) > 0)
    )
    capacity_feasible = capacity_base.filter(
        pl.col("entry_amount").fill_null(0) * DAILY_PARTICIPATION
        >= POSITION_NOTIONAL
    )
    yearly = (
        benchmarked.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("net_return").mean().alias("mean_net_return"),
            pl.col("excess_return").mean().alias("mean_excess_return"),
            pl.col("excess_return").sum().alias("sum_excess_return"),
        )
        .sort("year")
    )
    positive_years = yearly.filter(pl.col("mean_excess_return") > 0).height
    positive_sums = [
        float(value)
        for value in yearly["sum_excess_return"].to_list()
        if value is not None and value > 0
    ]
    maximum_year_share = (
        max(positive_sums) / sum(positive_sums) if positive_sums else None
    )
    result = {
        "events": trades.height,
        "universe_eligible_events": eligible.height,
        "tradable_events": tradable.height,
        "signal_days": benchmarked["ann_date"].n_unique()
        if benchmarked.height
        else 0,
        "tradable_rate": tradable.height / eligible.height if eligible.height else 0.0,
        "benchmark_coverage": benchmarked.height / tradable.height
        if tradable.height
        else 0.0,
        "entry_capacity_feasible_rate": capacity_feasible.height
        / capacity_base.height
        if capacity_base.height
        else 0.0,
        "unresolved_exits": eligible.filter(
            pl.col("entry_valid") & pl.col("exit_delay").is_null()
        ).height,
        "mean_net_return": benchmarked["net_return"].mean()
        if benchmarked.height
        else None,
        "mean_excess_return": benchmarked["excess_return"].mean()
        if benchmarked.height
        else None,
        "median_excess_return": benchmarked["excess_return"].median()
        if benchmarked.height
        else None,
        "absolute_daily_cluster_t": _cluster_t(benchmarked, "net_return"),
        "excess_daily_cluster_t": _cluster_t(benchmarked, "excess_return"),
        "positive_excess_years": positive_years,
        "max_year_positive_excess_share": maximum_year_share,
        "yearly": yearly.to_dicts(),
    }
    checks = {
        "tradable_events_at_least_150": result["tradable_events"] >= 150,
        "signal_days_at_least_120": result["signal_days"] >= 120,
        "tradable_rate_at_least_90pct": result["tradable_rate"] >= 0.90,
        "benchmark_coverage_at_least_99pct": result["benchmark_coverage"]
        >= 0.99,
        "entry_capacity_feasible_at_least_95pct": result[
            "entry_capacity_feasible_rate"
        ]
        >= 0.95,
        "unresolved_exits_zero": result["unresolved_exits"] == 0,
        "mean_net_return_at_least_1pct": (
            result["mean_net_return"] or -math.inf
        )
        >= 0.01,
        "mean_excess_at_least_0_75pct": (
            result["mean_excess_return"] or -math.inf
        )
        >= 0.0075,
        "excess_cluster_t_at_least_2": (
            result["excess_daily_cluster_t"] or -math.inf
        )
        >= 2.0,
        "positive_excess_years_at_least_5": positive_years >= 5,
        "max_positive_year_share_at_most_50pct": (
            maximum_year_share or math.inf
        )
        <= 0.50,
    }
    result["checks"] = checks
    result["promotion_passed"] = all(checks.values())
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    events = load_events(data_dir)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    trades = build_trades(events, panel, HOLD_TRADING_DAYS)
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    result = summarize(trades)
    payload = {
        "schema_version": "p0-st-removal-repricing-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "signal": "effective name transition from ST to non-ST and non-delisting",
            "entry": "next trading day open after effective name date",
            "benchmark": "same-entry-date eligible A-share 20-day median return",
        },
        "data": {
            "events": events.height,
            "event_symbols": events["symbol"].n_unique(),
            "panel_rows": panel.height,
            "panel_symbols": panel["symbol"].n_unique(),
        },
        "result": result,
        "decision": {
            "verdict": (
                "FREEZE_ACCOUNT_RULES"
                if result["promotion_passed"]
                else "TERMINATE"
            ),
            "counts_toward_50pct_goal": False,
            "validation_read": False,
        },
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
            "/app/data/research/p0_st_removal_repricing_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
