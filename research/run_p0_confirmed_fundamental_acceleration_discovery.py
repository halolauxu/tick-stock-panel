"""Run the frozen price-confirmed fundamental-acceleration discovery."""

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

from research import (  # noqa: E402
    run_p0_forecast_drift_development as forecast,
    run_p0_fundamental_acceleration_drift_discovery as fundamental,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PRIMARY_HORIZON = 5
DIAGNOSTIC_HORIZONS = (2, 10)
MAX_EXIT_DELAY = 20

CANDIDATE = "confirmed_underreaction"
NEGATIVE_CONTROL = "negative_reaction_control"
OVERREACTION_CONTROL = "overreaction_control"
CATEGORIES = (CANDIDATE, NEGATIVE_CONTROL, OVERREACTION_CONTROL)


def build_reaction_panel(raw_panel: pl.DataFrame) -> pl.DataFrame:
    calendar = raw_panel.select("date").unique().sort("date").with_row_index(
        "trade_index"
    )
    return (
        raw_panel.select("symbol", "date", "close")
        .join(calendar, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("close").shift(1).over("symbol").alias("prior_close"),
            pl.col("trade_index")
            .shift(1)
            .over("symbol")
            .alias("prior_trade_index"),
        )
        .filter(pl.col("trade_index") == pl.col("prior_trade_index") + 1)
        .with_columns(
            (pl.col("close") / pl.col("prior_close") - 1.0).alias(
                "reaction_return"
            )
        )
        .select(
            "symbol",
            pl.col("date").alias("reaction_date"),
            "reaction_return",
        )
        .sort(["symbol", "reaction_date"])
    )


def attach_first_reaction(
    events: pl.DataFrame, raw_panel: pl.DataFrame
) -> pl.DataFrame:
    calendar = (
        raw_panel.select(pl.col("date").alias("reaction_date"))
        .unique()
        .sort("reaction_date")
    )
    mapped = (
        events.with_columns(
            (pl.col("ann_date") + pl.duration(days=1)).alias("reaction_search_date")
        )
        .sort("reaction_search_date")
        .join_asof(
            calendar,
            left_on="reaction_search_date",
            right_on="reaction_date",
            strategy="forward",
        )
        .drop_nulls("reaction_date")
    )
    return mapped.join(
        build_reaction_panel(raw_panel),
        on=["symbol", "reaction_date"],
        how="inner",
    )


def classify_reactions(reactions: pl.DataFrame) -> pl.DataFrame:
    reaction = pl.col("reaction_return")
    category = (
        pl.when((reaction > 0) & (reaction <= 0.05))
        .then(pl.lit(CANDIDATE))
        .when(reaction.is_between(-0.05, 0.0, closed="both"))
        .then(pl.lit(NEGATIVE_CONTROL))
        .when((reaction > 0.05) & (reaction <= 0.11))
        .then(pl.lit(OVERREACTION_CONTROL))
        .otherwise(None)
    )
    return (
        reactions.drop("category")
        .rename({"ann_date": "report_announce_date", "reaction_date": "ann_date"})
        .with_columns(category.alias("category"))
        .filter(pl.col("category").is_not_null())
        .sort(["ann_date", "category", "symbol"])
    )


def load_mother_events(data_dir: Path) -> pl.DataFrame:
    comparisons = fundamental.build_report_comparisons(
        fundamental.load_metrics(data_dir)
    )
    return fundamental.classify_events(comparisons).filter(
        pl.col("category") == fundamental.CANDIDATE
    )


def summaries_for_horizon(
    events: pl.DataFrame, panel: pl.DataFrame, horizon: int
) -> dict[str, dict[str, Any]]:
    trades = forecast.build_trades(
        events,
        panel,
        holding_trading_days=horizon,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    trades = attach_market_excess(
        trades, build_market_benchmark(panel, holding_trading_days=horizon)
    )
    return {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=700,
            min_announcement_days=250,
        )
        for category in CATEGORIES
    }


def evaluate(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = primary[CANDIDATE]
    negative = primary[NEGATIVE_CONTROL]
    overreaction = primary[OVERREACTION_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    negative_excess = negative.get("mean_excess_return")
    overreaction_excess = overreaction.get("mean_excess_return")
    versus_negative = (
        candidate_excess - negative_excess
        if candidate_excess is not None and negative_excess is not None
        else None
    )
    versus_overreaction = (
        candidate_excess - overreaction_excess
        if candidate_excess is not None and overreaction_excess is not None
        else None
    )
    eligible = candidate["universe_eligible_events"]
    unresolved_rate = candidate["unresolved_exits"] / eligible if eligible else math.inf
    checks = {
        "at_least_700_tradable_events": candidate["tradable_events"] >= 700,
        "at_least_250_announcement_days": candidate["announcement_days"] >= 250,
        "tradable_rate_at_least_90pct": candidate["tradable_rate"] >= 0.90,
        "benchmark_coverage_at_least_99pct": candidate["benchmark_coverage"] >= 0.99,
        "capacity_feasible_at_least_95pct": candidate[
            "entry_capacity_feasible_rate"
        ]
        >= 0.95,
        "unresolved_exit_rate_at_most_1pct": unresolved_rate <= 0.01,
        "mean_net_return_at_least_1pct": (
            candidate.get("mean_net_return") or -math.inf
        )
        >= 0.01,
        "mean_excess_at_least_75bp": (candidate_excess or -math.inf) >= 0.0075,
        "excess_cluster_t_at_least_2_5": (
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 2.5,
        "at_least_5_positive_excess_years": candidate["positive_excess_years"] >= 5,
        "max_year_positive_share_at_most_50pct": (
            candidate.get("max_year_positive_excess_share") or math.inf
        )
        <= 0.50,
        "beats_negative_control_by_50bp": (versus_negative or -math.inf) >= 0.005,
        "beats_overreaction_control_by_50bp": (
            versus_overreaction or -math.inf
        )
        >= 0.005,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "unresolved_exit_rate": unresolved_rate,
        "candidate_excess_minus_negative_control": versus_negative,
        "candidate_excess_minus_overreaction_control": versus_overreaction,
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_panel = fundamental.load_main_board_panel(data_dir)
    mother_events = load_mother_events(data_dir)
    events = classify_reactions(attach_first_reaction(mother_events, raw_panel))
    panel = forecast.prepare_panel(raw_panel)
    primary = summaries_for_horizon(events, panel, PRIMARY_HORIZON)
    diagnostics = {
        str(horizon): summaries_for_horizon(events, panel, horizon)
        for horizon in DIAGNOSTIC_HORIZONS
    }
    decision = evaluate(primary)
    payload = {
        "schema_version": "p0-confirmed-fundamental-acceleration-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "mother_event": fundamental.CANDIDATE,
            "candidate_reaction_range": [0.0, 0.05],
            "candidate_lower_bound_open": True,
            "negative_control_range": [-0.05, 0.0],
            "overreaction_control_range": [0.05, 0.11],
            "primary_holding_trading_days": PRIMARY_HORIZON,
            "diagnostic_holding_trading_days": list(DIAGNOSTIC_HORIZONS),
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "execution": "open after the first complete reaction day",
        },
        "data": {
            "mother_events": mother_events.height,
            "classified_reaction_events": events.height,
            "event_counts": {
                category: events.filter(pl.col("category") == category).height
                for category in CATEGORIES
            },
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "primary_5d": primary,
        "diagnostics_only": diagnostics,
        "decision": decision,
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
            "/app/data/research/p0_confirmed_fundamental_acceleration_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
