"""Run the frozen main-board fundamental-acceleration drift discovery."""

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

CANDIDATE = "dual_acceleration_cash_confirmed"
LEVEL_CONTROL = "positive_level_no_dual_acceleration_control"
CASH_POOR_CONTROL = "cash_poor_acceleration_control"
CATEGORIES = (CANDIDATE, LEVEL_CONTROL, CASH_POOR_CONTROL)

MIN_REVENUE_ACCELERATION = 5.0
MIN_PROFIT_ACCELERATION = 10.0
MAX_DEBT_RATIO = 80.0

METRIC_COLUMNS = (
    "symbol",
    "period_end",
    "announce_date",
    "roe",
    "debt_to_asset_ratio",
    "revenue_yoy",
    "net_income_yoy",
    "operating_cash_to_revenue",
)


def load_metrics(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "financials" / "metrics" / "part.parquet"
    if not path.is_file():
        raise ValueError("financial metrics history is required")
    frame = pl.read_parquet(path)
    missing = set(METRIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"financial metrics missing columns: {sorted(missing)}")
    return frame.select(METRIC_COLUMNS)


def build_report_comparisons(
    metrics: pl.DataFrame,
    start: date = DEVELOPMENT_START,
    end: date = DEVELOPMENT_END,
) -> pl.DataFrame:
    ordered = (
        metrics.with_columns(
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("period_end"),
            pl.col("announce_date")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("announce_date"),
        )
        .filter(
            pl.col("announce_date").is_between(start, end, closed="both")
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
        .sort(["symbol", "period_end", "announce_date"])
        .unique(["symbol", "period_end"], keep="first", maintain_order=True)
        .sort(["symbol", "period_end"])
        .with_columns(
            pl.col("period_end").shift(1).over("symbol").alias("prior_period_end"),
            pl.col("announce_date")
            .shift(1)
            .over("symbol")
            .alias("prior_announce_date"),
            pl.col("revenue_yoy")
            .shift(1)
            .over("symbol")
            .alias("prior_revenue_yoy"),
            pl.col("net_income_yoy")
            .shift(1)
            .over("symbol")
            .alias("prior_net_income_yoy"),
        )
        .with_columns(
            (
                pl.col("period_end").dt.year() * 4
                + pl.col("period_end").dt.quarter()
                - pl.col("prior_period_end").dt.year() * 4
                - pl.col("prior_period_end").dt.quarter()
            ).alias("quarter_gap"),
            (pl.col("revenue_yoy") - pl.col("prior_revenue_yoy")).alias(
                "revenue_acceleration"
            ),
            (pl.col("net_income_yoy") - pl.col("prior_net_income_yoy")).alias(
                "profit_acceleration"
            ),
        )
        .filter(
            (pl.col("quarter_gap") == 1)
            & (pl.col("prior_announce_date") < pl.col("announce_date"))
        )
    )
    return ordered


def candidate_expression() -> pl.Expr:
    return (
        (pl.col("revenue_yoy") > 0)
        & (pl.col("net_income_yoy") > 0)
        & (pl.col("revenue_acceleration") >= MIN_REVENUE_ACCELERATION)
        & (pl.col("profit_acceleration") >= MIN_PROFIT_ACCELERATION)
        & (pl.col("roe") > 0)
        & (pl.col("operating_cash_to_revenue") > 0)
        & (pl.col("debt_to_asset_ratio") <= MAX_DEBT_RATIO)
    )


def classify_events(comparisons: pl.DataFrame) -> pl.DataFrame:
    positive_level = (
        (pl.col("revenue_yoy") > 0)
        & (pl.col("net_income_yoy") > 0)
        & (pl.col("roe") > 0)
        & (pl.col("debt_to_asset_ratio") <= MAX_DEBT_RATIO)
    )
    dual_acceleration = (
        (pl.col("revenue_acceleration") >= MIN_REVENUE_ACCELERATION)
        & (pl.col("profit_acceleration") >= MIN_PROFIT_ACCELERATION)
    )
    cash_confirmed = pl.col("operating_cash_to_revenue") > 0
    category = (
        pl.when(candidate_expression())
        .then(pl.lit(CANDIDATE))
        .when(positive_level & dual_acceleration & ~cash_confirmed)
        .then(pl.lit(CASH_POOR_CONTROL))
        .when(positive_level & ~dual_acceleration & cash_confirmed)
        .then(pl.lit(LEVEL_CONTROL))
        .otherwise(None)
    )
    return (
        comparisons.with_columns(category.alias("category"))
        .filter(pl.col("category").is_not_null())
        .sort(["symbol", "announce_date", "period_end"], descending=[False, False, True])
        .unique(["symbol", "announce_date"], keep="first", maintain_order=True)
        .rename({"announce_date": "ann_date"})
        .sort(["ann_date", "category", "symbol"])
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
    trades = attach_market_excess(
        trades, build_market_benchmark(panel, holding_trading_days=horizon)
    )
    return {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=1_000,
            min_announcement_days=300,
        )
        for category in CATEGORIES
    }


def evaluate(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = primary[CANDIDATE]
    level = primary[LEVEL_CONTROL]
    cash_poor = primary[CASH_POOR_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    level_excess = level.get("mean_excess_return")
    cash_poor_excess = cash_poor.get("mean_excess_return")
    versus_level = (
        candidate_excess - level_excess
        if candidate_excess is not None and level_excess is not None
        else None
    )
    versus_cash_poor = (
        candidate_excess - cash_poor_excess
        if candidate_excess is not None and cash_poor_excess is not None
        else None
    )
    eligible = candidate["universe_eligible_events"]
    unresolved_rate = candidate["unresolved_exits"] / eligible if eligible else math.inf
    checks = {
        "at_least_1000_tradable_events": candidate["tradable_events"] >= 1_000,
        "at_least_300_announcement_days": candidate["announcement_days"] >= 300,
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
        "beats_level_control_by_50bp": (versus_level or -math.inf) >= 0.005,
        "beats_cash_poor_control_by_50bp": (versus_cash_poor or -math.inf) >= 0.005,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "unresolved_exit_rate": unresolved_rate,
        "candidate_excess_minus_level_control": versus_level,
        "candidate_excess_minus_cash_poor_control": versus_cash_poor,
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
    metrics = load_metrics(data_dir)
    comparisons = build_report_comparisons(metrics)
    events = classify_events(comparisons)
    panel = forecast.prepare_panel(load_main_board_panel(data_dir))
    primary = summaries_for_horizon(events, panel, PRIMARY_HORIZON)
    diagnostics = {
        str(horizon): summaries_for_horizon(events, panel, horizon)
        for horizon in DIAGNOSTIC_HORIZONS
    }
    decision = evaluate(primary)
    payload = {
        "schema_version": "p0-fundamental-acceleration-drift-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "minimum_revenue_acceleration_pct_points": MIN_REVENUE_ACCELERATION,
            "minimum_profit_acceleration_pct_points": MIN_PROFIT_ACCELERATION,
            "maximum_debt_ratio_pct": MAX_DEBT_RATIO,
            "primary_holding_trading_days": PRIMARY_HORIZON,
            "diagnostic_holding_trading_days": list(DIAGNOSTIC_HORIZONS),
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "execution": "first trading-day open after announcement",
        },
        "data": {
            "metric_rows": metrics.height,
            "consecutive_report_comparisons": comparisons.height,
            "study_events": events.height,
            "event_counts": {
                category: events.filter(pl.col("category") == category).height
                for category in CATEGORIES
            },
            "candidate_symbols": events.filter(pl.col("category") == CANDIDATE)
            .get_column("symbol")
            .n_unique(),
            "candidate_announcement_days": events.filter(
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
            "/app/data/research/p0_fundamental_acceleration_drift_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
