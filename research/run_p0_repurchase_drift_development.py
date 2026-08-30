"""Run the frozen development-only stock-repurchase drift event study."""
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

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
COOLDOWN_DAYS = 365

CATEGORIES = (
    "proposal_approved",
    "implementation",
    "completion",
    "termination_control",
)
POSITIVE_CATEGORIES = CATEGORIES[:-1]
PROPOSAL_LABELS = ("预案", "董事会预案", "董事会通过", "股东大会通过")


def load_repurchase_events(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "repurchase").glob(
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
        raise ValueError("all 2014-2020 repurchase yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "proc"])
    )


def _category_expression() -> pl.Expr:
    process = pl.col("proc").fill_null("").str.strip_chars()
    return (
        pl.when(process.str.contains(r"(?:停止|终止)"))
        .then(pl.lit("termination_control"))
        .when(process.is_in(PROPOSAL_LABELS))
        .then(pl.lit("proposal_approved"))
        .when(process == "实施")
        .then(pl.lit("implementation"))
        .when(process == "完成")
        .then(pl.lit("completion"))
        .otherwise(None)
    )


def categorize_events(events: pl.DataFrame) -> pl.DataFrame:
    categorized = (
        events.with_columns(
            _category_expression().alias("category"),
            pl.col("repurchase_amount_cny").is_not_null().alias("_amount_known"),
        )
        .filter(pl.col("category").is_not_null())
        .sort(
            [
                "symbol",
                "category",
                "ann_date",
                "_amount_known",
                "repurchase_amount_cny",
                "end_date",
            ],
            descending=[False, False, False, True, True, True],
            nulls_last=True,
        )
        .unique(
            subset=["symbol", "category", "ann_date"],
            keep="first",
            maintain_order=True,
        )
        .drop("_amount_known")
    )
    last_kept: dict[tuple[str, str], date] = {}
    keep = []
    for row in categorized.iter_rows(named=True):
        key = (row["symbol"], row["category"])
        event_date = row["ann_date"]
        previous = last_kept.get(key)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[key] = event_date
    return categorized.filter(pl.Series("_keep", keep)).sort(
        ["ann_date", "symbol", "category"]
    )


def build_market_benchmark(panel: pl.DataFrame) -> pl.DataFrame:
    entry = panel.select(
        "symbol",
        "trade_index",
        pl.col("open").alias("benchmark_entry_open"),
        pl.col("raw_open").alias("benchmark_entry_raw_open"),
        pl.col("amount").alias("benchmark_entry_amount"),
        pl.col("volume").alias("benchmark_entry_volume"),
        pl.col("excluded_name").alias("benchmark_entry_excluded"),
    )
    exit_prices = panel.select(
        "symbol",
        (pl.col("trade_index") - HOLD_TRADING_DAYS).alias("trade_index"),
        pl.col("open").alias("benchmark_exit_open"),
    )
    return (
        entry.join(exit_prices, on=["symbol", "trade_index"], how="inner")
        .filter(
            ~pl.col("benchmark_entry_excluded").fill_null(True)
            & pl.col("benchmark_entry_raw_open").is_between(
                3.0, 300.0, closed="both"
            )
            & (pl.col("benchmark_entry_amount").fill_null(0) >= 20_000_000.0)
            & (pl.col("benchmark_entry_volume").fill_null(0) > 0)
            & (pl.col("benchmark_entry_open").fill_null(0) > 0)
            & (pl.col("benchmark_exit_open").fill_null(0) > 0)
        )
        .with_columns(
            (
                pl.col("benchmark_exit_open")
                / pl.col("benchmark_entry_open")
                - 1.0
            ).alias("market_return")
        )
        .group_by("trade_index")
        .agg(
            pl.col("market_return").median().alias("market_median_return"),
            pl.len().alias("market_symbols"),
        )
        .sort("trade_index")
    )


def attach_market_excess(
    trades: pl.DataFrame, benchmark: pl.DataFrame
) -> pl.DataFrame:
    return trades.join(benchmark, on="trade_index", how="left").with_columns(
        pl.when(pl.col("tradable") & pl.col("market_median_return").is_not_null())
        .then(pl.col("net_return") - pl.col("market_median_return"))
        .otherwise(None)
        .alias("excess_return")
    )


def _cluster_t(frame: pl.DataFrame, column: str) -> float | None:
    daily = (
        frame.drop_nulls(column)
        .group_by("ann_date")
        .agg(pl.col(column).mean().alias("daily_return"))
    )
    if daily.height <= 1:
        return None
    mean = daily.get_column("daily_return").mean()
    standard_deviation = daily.get_column("daily_return").std(ddof=1)
    if mean is None or standard_deviation in (None, 0.0):
        return None
    return mean / (standard_deviation / math.sqrt(daily.height))


def summarize_category(
    trades: pl.DataFrame,
    category: str,
    positive_categories: tuple[str, ...] = POSITIVE_CATEGORIES,
) -> dict[str, Any]:
    scoped = trades.filter(pl.col("category") == category)
    eligible = scoped.filter(pl.col("universe_eligible"))
    tradable = scoped.filter(pl.col("tradable"))
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
        for value in yearly.get_column("sum_excess_return").to_list()
        if value is not None and value > 0
    ]
    maximum_year_share = (
        max(positive_sums) / sum(positive_sums) if positive_sums else None
    )
    result = {
        "events": scoped.height,
        "universe_eligible_events": eligible.height,
        "tradable_events": tradable.height,
        "benchmarked_events": benchmarked.height,
        "announcement_days": benchmarked.get_column("ann_date").n_unique()
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
        "mean_net_return": benchmarked.get_column("net_return").mean()
        if benchmarked.height
        else None,
        "mean_excess_return": benchmarked.get_column("excess_return").mean()
        if benchmarked.height
        else None,
        "median_excess_return": benchmarked.get_column("excess_return").median()
        if benchmarked.height
        else None,
        "absolute_daily_cluster_t": _cluster_t(benchmarked, "net_return"),
        "excess_daily_cluster_t": _cluster_t(benchmarked, "excess_return"),
        "positive_excess_years": positive_years,
        "max_year_positive_excess_share": maximum_year_share,
        "yearly": yearly.to_dicts(),
    }
    result["promotion_passed"] = bool(
        category in positive_categories
        and result["tradable_events"] >= 300
        and result["announcement_days"] >= 150
        and result["tradable_rate"] >= 0.90
        and result["benchmark_coverage"] >= 0.99
        and result["entry_capacity_feasible_rate"] >= 0.95
        and result["unresolved_exits"] == 0
        and (result["mean_net_return"] or -math.inf) >= 0.01
        and (result["mean_excess_return"] or -math.inf) >= 0.0075
        and (result["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and result["positive_excess_years"] >= 5
        and (result["max_year_positive_excess_share"] or math.inf) <= 0.50
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_events = load_repurchase_events(data_dir)
    events = categorize_events(raw_events)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    trades = build_trades(events, panel, HOLD_TRADING_DAYS)
    benchmark = build_market_benchmark(panel)
    trades = attach_market_excess(trades, benchmark)
    summaries = {
        category: summarize_category(trades, category) for category in CATEGORIES
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
    raw_process_counts = (
        raw_events.group_by("proc").len().sort("len", descending=True).to_dicts()
    )
    payload = {
        "schema_version": "p0-repurchase-drift-development-v1",
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
            "raw_repurchase_rows": raw_events.height,
            "raw_process_counts": raw_process_counts,
            "categorized_unique_events": events.height,
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
                else "terminate_repurchase_drift_mechanism"
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
        default=Path("/app/data/research/p0_repurchase_drift_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
