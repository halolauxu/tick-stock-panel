"""Run the frozen industry-confirmed earnings-forecast drift discovery."""

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

from research import run_p0_forecast_drift_development as forecast  # noqa: E402
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PRIMARY_HORIZON = 10
DIAGNOSTIC_HORIZONS = (2, 5)
MAX_EXIT_DELAY = 20
MAIN_BOARD_PATTERN = r"^(?:600|601|603|605|000|001|002|003)\d{3}\.(?:SH|SZ)$"

CANDIDATE = "industry_confirmed_positive"
NEUTRAL_CONTROL = "unconfirmed_positive_control"
NEGATIVE_CONTROL = "industry_negative_breadth_control"
CATEGORIES = (CANDIDATE, NEUTRAL_CONTROL, NEGATIVE_CONTROL)
POSITIVE_FORECAST_CATEGORIES = (
    "turnaround",
    "growth_ge_100",
    "growth_50_100",
    "growth_0_50",
)
TARGET_FORECAST_CATEGORIES = ("growth_0_50", "growth_50_100")


def load_point_in_time_membership(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "research" / "sw_l1_membership.parquet"
    if not path.is_file():
        raise ValueError("point-in-time SW L1 membership is required")
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col("in_date").cast(pl.Date, strict=False),
            pl.col("out_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "l1_code", "l1_name", "in_date", "out_date")
        .sort(["symbol", "in_date"])
    )


def attach_industry(
    events: pl.DataFrame, membership: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    main_board = events.filter(pl.col("symbol").str.contains(MAIN_BOARD_PATTERN))
    joined = main_board.sort(["symbol", "ann_date"]).join_asof(
        membership,
        left_on="ann_date",
        right_on="in_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    valid = joined.filter(
        pl.col("l1_code").is_not_null()
        & (pl.col("out_date").is_null() | (pl.col("ann_date") <= pl.col("out_date")))
    )
    audit = {
        "main_board_events": main_board.height,
        "mapped_events": valid.height,
        "mapping_rate": valid.height / main_board.height if main_board.height else 0.0,
        "mapped_symbols": valid.get_column("symbol").n_unique(),
        "industries": valid.get_column("l1_code").n_unique(),
    }
    return valid, audit


def classify_events(events: pl.DataFrame) -> pl.DataFrame:
    work = events.with_columns(
        pl.col("category")
        .is_in(POSITIVE_FORECAST_CATEGORIES)
        .cast(pl.Int64)
        .alias("_positive")
    )
    industry_day = work.group_by("ann_date", "l1_code").agg(
        pl.len().alias("_industry_count"),
        pl.col("_positive").sum().alias("_industry_positive"),
    )
    market_day = work.group_by("ann_date").agg(
        pl.len().alias("_market_count"),
        pl.col("_positive").sum().alias("_market_positive"),
    )
    scored = (
        work.join(industry_day, on=["ann_date", "l1_code"], how="left")
        .join(market_day, on="ann_date", how="left")
        .with_columns(
            (pl.col("_industry_count") - 1).alias("industry_peer_count"),
            (
                (pl.col("_industry_positive") - pl.col("_positive"))
                / (pl.col("_industry_count") - 1)
            ).alias("industry_peer_positive_share"),
            (
                (pl.col("_market_positive") - pl.col("_positive"))
                / (pl.col("_market_count") - 1)
            ).alias("market_peer_positive_share"),
        )
        .with_columns(
            (
                pl.col("industry_peer_positive_share")
                - pl.col("market_peer_positive_share")
            ).alias("industry_positive_share_excess")
        )
        .filter(pl.col("category").is_in(TARGET_FORECAST_CATEGORIES))
    )
    high = (
        (pl.col("industry_peer_count") >= 5)
        & (pl.col("industry_peer_positive_share") >= 0.70)
        & (pl.col("industry_positive_share_excess") >= 0.15)
    )
    low = (
        (pl.col("industry_peer_count") >= 5)
        & (pl.col("industry_peer_positive_share") <= 0.40)
        & (pl.col("industry_positive_share_excess") <= -0.10)
    )
    study_category = (
        pl.when(high)
        .then(pl.lit(CANDIDATE))
        .when(low)
        .then(pl.lit(NEGATIVE_CONTROL))
        .otherwise(pl.lit(NEUTRAL_CONTROL))
    )
    return (
        scored.with_columns(study_category.alias("study_category"))
        .drop("category")
        .rename({"study_category": "category"})
        .sort(["ann_date", "category", "l1_code", "symbol"])
    )


def load_main_board_panel(data_dir: Path) -> pl.DataFrame:
    return forecast.load_panel(data_dir).filter(
        pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
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
    benchmark = build_market_benchmark(panel, horizon)
    trades = attach_market_excess(trades, benchmark)
    return {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=400,
            min_announcement_days=60,
        )
        for category in CATEGORIES
    }


def evaluate(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = primary[CANDIDATE]
    neutral = primary[NEUTRAL_CONTROL]
    negative = primary[NEGATIVE_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    neutral_excess = neutral.get("mean_excess_return")
    negative_excess = negative.get("mean_excess_return")
    versus_neutral = (
        candidate_excess - neutral_excess
        if candidate_excess is not None and neutral_excess is not None
        else None
    )
    versus_negative = (
        candidate_excess - negative_excess
        if candidate_excess is not None and negative_excess is not None
        else None
    )
    eligible = candidate["universe_eligible_events"]
    unresolved_rate = candidate["unresolved_exits"] / eligible if eligible else math.inf
    checks = {
        "at_least_400_tradable_events": candidate["tradable_events"] >= 400,
        "at_least_60_announcement_days": candidate["announcement_days"] >= 60,
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
        "mean_excess_at_least_1pct": (candidate_excess or -math.inf) >= 0.01,
        "excess_cluster_t_at_least_2_5": (
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 2.5,
        "at_least_5_positive_excess_years": candidate["positive_excess_years"] >= 5,
        "max_year_positive_share_at_most_50pct": (
            candidate.get("max_year_positive_excess_share") or math.inf
        )
        <= 0.50,
        "beats_unconfirmed_by_50bp": (versus_neutral or -math.inf) >= 0.005,
        "beats_negative_breadth_by_75bp": (versus_negative or -math.inf) >= 0.0075,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "unresolved_exit_rate": unresolved_rate,
        "candidate_excess_minus_unconfirmed": versus_neutral,
        "candidate_excess_minus_negative_breadth": versus_negative,
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
    raw_events = forecast.load_forecasts(data_dir)
    categorized = forecast.categorize_events(raw_events)
    mapped, membership_audit = attach_industry(
        categorized, load_point_in_time_membership(data_dir)
    )
    events = classify_events(mapped)
    panel = forecast.prepare_panel(load_main_board_panel(data_dir))
    primary = summaries_for_horizon(events, panel, PRIMARY_HORIZON)
    diagnostics = {
        str(horizon): summaries_for_horizon(events, panel, horizon)
        for horizon in DIAGNOSTIC_HORIZONS
    }
    decision = evaluate(primary)
    payload = {
        "schema_version": "p0-industry-confirmed-forecast-drift-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "industry_membership": "point_in_time_sw_l1",
            "minimum_industry_peers": 5,
            "candidate_peer_positive_share": 0.70,
            "candidate_excess_over_market_share": 0.15,
            "negative_control_peer_positive_share": 0.40,
            "negative_control_below_market_share": -0.10,
            "primary_holding_trading_days": PRIMARY_HORIZON,
            "diagnostic_holding_trading_days": list(DIAGNOSTIC_HORIZONS),
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
        },
        "data": {
            "raw_forecast_rows": raw_events.height,
            **membership_audit,
            "study_events": events.height,
            "candidate_events_before_execution": events.filter(
                pl.col("category") == CANDIDATE
            ).height,
            "candidate_announcement_days_before_execution": events.filter(
                pl.col("category") == CANDIDATE
            )
            .get_column("ann_date")
            .n_unique(),
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "primary_10d": primary,
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
            "/app/data/research/"
            "p0_industry_confirmed_forecast_drift_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()

