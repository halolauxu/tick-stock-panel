"""Run the frozen 2021-2023 micro-cap shareholder-concentration validation."""

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

import run_p0_microcap_baseline as microcap  # noqa: E402
from research.run_p0_forecast_drift_development import (  # noqa: E402
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_holder_concentration_development import (  # noqa: E402
    COOLDOWN_DAYS,
    MAX_MEASUREMENT_GAP_DAYS,
    MIN_CONTRACTION,
    MIN_MEASUREMENT_GAP_DAYS,
)
from research.run_p0_large_order_flow_development import MAX_EXIT_DELAY  # noqa: E402
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

CONTEXT_START = date(2020, 1, 1)
VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = date(2023, 12, 31)
PANEL_END = date(2024, 5, 31)
HOLD_TRADING_DAYS = 20
MAX_CAP_SNAPSHOT_AGE_DAYS = 10
CATEGORY = "microcap_holder_concentration"


def load_holder_numbers(data_dir: Path) -> pl.DataFrame:
    paths: list[Path] = []
    for path in (data_dir / "event_data" / "holder_number").glob("year=*/part.parquet"):
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if CONTEXT_START.year <= year <= VALIDATION_END.year:
            paths.append(path)
    expected = VALIDATION_END.year - CONTEXT_START.year + 1
    if len(paths) != expected:
        raise ValueError("all 2020-2023 holder-number yearly partitions are required")
    return pl.read_parquet(sorted(paths), hive_partitioning=False).filter(
        pl.col("ann_date").is_between(CONTEXT_START, VALIDATION_END, closed="both")
    )


def build_holder_events(holder_numbers: pl.DataFrame) -> pl.DataFrame:
    disclosure = (
        holder_numbers.drop_nulls(["symbol", "ann_date", "end_date", "holder_num"])
        .filter(pl.col("holder_num") > 0)
        .sort(["symbol", "ann_date", "end_date"], descending=[False, False, True])
        .unique(subset=["symbol", "ann_date"], keep="first", maintain_order=True)
        .sort(["symbol", "ann_date"])
        .with_columns(
            pl.col("holder_num").shift(1).over("symbol").alias("previous_holder_num"),
            pl.col("end_date").shift(1).over("symbol").alias("previous_end_date"),
        )
        .with_columns(
            (pl.col("holder_num") / pl.col("previous_holder_num") - 1.0).alias(
                "holder_count_change"
            ),
            (pl.col("end_date") - pl.col("previous_end_date"))
            .dt.total_days()
            .alias("measurement_gap_days"),
        )
        .filter(
            pl.col("ann_date").is_between(VALIDATION_START, VALIDATION_END, closed="both")
            & (pl.col("end_date") > pl.col("previous_end_date"))
            & pl.col("measurement_gap_days").is_between(
                MIN_MEASUREMENT_GAP_DAYS,
                MAX_MEASUREMENT_GAP_DAYS,
                closed="both",
            )
            & (pl.col("holder_count_change") <= -MIN_CONTRACTION)
        )
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep: list[bool] = []
    for row in disclosure.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    return disclosure.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def build_cap_snapshots(raw_panel: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    shares = microcap.load_share_history(data_dir)
    eligible = (
        raw_panel.filter(
            pl.col("symbol").str.contains(microcap.SYMBOL_PATTERN)
            & ((pl.col("date") - pl.col("list_date")).dt.total_days() >= microcap.MIN_LISTING_DAYS)
        )
        .sort(["symbol", "date"])
        .join_asof(
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter((pl.col("total_shares") > 0) & (pl.col("raw_close") > 0))
        .with_columns(
            (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
            pl.len().over("date").alias("cap_universe_size"),
        )
        .with_columns(pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"))
        .with_columns(
            (
                ((pl.col("cap_rank") - 1) * 10 / pl.col("cap_universe_size"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile")
        )
    )
    return eligible.select(
        "symbol", "date", "market_cap", "cap_rank", "cap_universe_size", "cap_decile"
    ).sort(["symbol", "date"])


def attach_microcap_filter(
    holder_events: pl.DataFrame, cap_snapshots: pl.DataFrame
) -> pl.DataFrame:
    return (
        holder_events.sort(["symbol", "ann_date"])
        .join_asof(
            cap_snapshots,
            left_on="ann_date",
            right_on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("ann_date") - pl.col("date")).dt.total_days().alias("cap_snapshot_age_days")
        )
        .filter(
            (pl.col("cap_decile") == 0)
            & pl.col("cap_snapshot_age_days").is_between(
                0, MAX_CAP_SNAPSHOT_AGE_DAYS, closed="both"
            )
        )
        .with_columns(pl.lit(CATEGORY).alias("category"))
        .sort(["ann_date", "symbol"])
    )


def validation_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["tradable_events"] >= 150
        and metrics["announcement_days"] >= 100
        and metrics["tradable_rate"] >= 0.90
        and metrics["benchmark_coverage"] >= 0.99
        and metrics["entry_capacity_feasible_rate"] >= 0.95
        and metrics["unresolved_exits"] == 0
        and (metrics["mean_net_return"] or -math.inf) >= 0.04
        and (metrics["mean_excess_return"] or -math.inf) >= 0.025
        and (metrics["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and metrics["positive_excess_years"] >= 2
        and (metrics["max_year_positive_excess_share"] or math.inf) <= 0.60
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    holder_events = build_holder_events(load_holder_numbers(data_dir))
    raw_panel = load_panel(data_dir, CONTEXT_START, PANEL_END)
    joint_events = attach_microcap_filter(holder_events, build_cap_snapshots(raw_panel, data_dir))
    panel = prepare_panel(raw_panel)
    trades = build_trades(
        joint_events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    metrics = summarize_category(
        trades,
        CATEGORY,
        positive_categories=(CATEGORY,),
        min_tradable_events=150,
        min_announcement_days=100,
    )
    metrics["promotion_passed"] = validation_passed(metrics)
    passed = metrics["promotion_passed"]
    payload = {
        "schema_version": "p0-microcap-holder-concentration-validation-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": VALIDATION_START,
            "end": VALIDATION_END,
            "joint_development_returns_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "minimum_holder_count_contraction": MIN_CONTRACTION,
            "microcap_decile": 0,
            "maximum_cap_snapshot_age_days": MAX_CAP_SNAPSHOT_AGE_DAYS,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 20-day median return",
        },
        "data": {
            "holder_contraction_events": holder_events.height,
            "joint_events": joint_events.height,
            "joint_symbols": joint_events.get_column("symbol").n_unique()
            if joint_events.height
            else 0,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "metrics": metrics,
        "decision": {
            "validation_passed": passed,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_account_and_stress_rules_before_reading_2024_plus"
                if passed
                else "terminate_microcap_holder_concentration_mechanism"
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
        default=Path("/app/data/research/p0_microcap_holder_concentration_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
