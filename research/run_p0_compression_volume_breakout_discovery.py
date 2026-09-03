"""Run the frozen main-board compression-volume breakout discovery."""

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

WARMUP_START = date(2013, 10, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 2, 28)
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
MAIN_BOARD_PATTERN = r"^(?:600|601|603|605|000|001|002|003)\d{3}\.(?:SH|SZ)$"

CANDIDATE = "compression_volume_breakout"
LOW_SCORE_CONTROL = "low_composite_score_control"
CATEGORIES = (CANDIDATE, LOW_SCORE_CONTROL)


def load_feature_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    panel = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "raw_close",
            "raw_high",
            "raw_low",
        )
        .filter(
            pl.col("date").is_between(WARMUP_START, PANEL_END, closed="both")
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
        .collect(engine="streaming")
    )
    return forecast.attach_point_in_time_universe(panel, data_dir)


def attach_features(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("close")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("ma20"),
            pl.col("close")
            .rolling_std(window_size=20, min_samples=20, ddof=1)
            .over("symbol")
            .alias("close_std_20d"),
            pl.col("volume")
            .shift(1)
            .rolling_mean(window_size=10, min_samples=10)
            .over("symbol")
            .alias("prior_volume_mean_10d"),
        )
        .with_columns(
            (4.0 * pl.col("close_std_20d") / pl.col("ma20")).alias(
                "boll_width"
            ),
            (pl.col("volume") / pl.col("prior_volume_mean_10d")).alias(
                "vol_ratio_10d"
            ),
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(
                "close_position"
            ),
        )
    )


def weekly_signal_dates(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.select("date")
        .unique()
        .sort("date")
        .filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END))
        .with_columns(pl.col("date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(pl.col("date").max().alias("date"))
        .drop("week")
    )


def build_ranked_events(feature_panel: pl.DataFrame) -> pl.DataFrame:
    weekly = weekly_signal_dates(feature_panel)
    eligible = (
        feature_panel.join(weekly, on="date", how="inner")
        .filter(
            pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("amount") >= 20_000_000.0)
            & (pl.col("boll_width") > 0)
            & (pl.col("vol_ratio_10d") > 0)
            & pl.col("close_position").is_between(0.0, 1.0, closed="both")
        )
        .with_columns(pl.len().over("date").alias("cross_section_count"))
        .with_columns(
            (
                1.0
                - (pl.col("boll_width").rank(method="average").over("date") - 1.0)
                / (pl.col("cross_section_count") - 1.0)
            ).alias("boll_width_score"),
            (
                (pl.col("vol_ratio_10d").rank(method="average").over("date") - 1.0)
                / (pl.col("cross_section_count") - 1.0)
            ).alias("volume_score"),
            (
                (pl.col("close_position").rank(method="average").over("date") - 1.0)
                / (pl.col("cross_section_count") - 1.0)
            ).alias("close_position_score"),
        )
        .with_columns(
            (
                100.0
                * (
                    0.30 * pl.col("boll_width_score")
                    + 0.40 * pl.col("volume_score")
                    + 0.30 * pl.col("close_position_score")
                )
            ).alias("composite_score")
        )
        .with_columns(
            pl.col("composite_score")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("high_rank"),
            pl.col("composite_score")
            .rank(method="ordinal")
            .over("date")
            .alias("low_rank"),
        )
    )
    category = (
        pl.when((pl.col("composite_score") >= 75.0) & (pl.col("high_rank") <= 20))
        .then(pl.lit(CANDIDATE))
        .when((pl.col("composite_score") <= 25.0) & (pl.col("low_rank") <= 20))
        .then(pl.lit(LOW_SCORE_CONTROL))
        .otherwise(None)
    )
    return (
        eligible.with_columns(category.alias("category"))
        .filter(pl.col("category").is_not_null())
        .select(
            "symbol",
            pl.col("date").alias("ann_date"),
            "category",
            "boll_width",
            "vol_ratio_10d",
            "close_position",
            "composite_score",
            "high_rank",
            "low_rank",
        )
        .sort(["ann_date", "category", "symbol"])
    )


def evaluate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = summaries[CANDIDATE]
    control = summaries[LOW_SCORE_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    control_excess = control.get("mean_excess_return")
    versus_control = (
        candidate_excess - control_excess
        if candidate_excess is not None and control_excess is not None
        else None
    )
    eligible = candidate["universe_eligible_events"]
    unresolved_rate = candidate["unresolved_exits"] / eligible if eligible else math.inf
    checks = {
        "at_least_5000_tradable_events": candidate["tradable_events"] >= 5_000,
        "at_least_300_signal_days": candidate["announcement_days"] >= 300,
        "tradable_rate_at_least_90pct": candidate["tradable_rate"] >= 0.90,
        "benchmark_coverage_at_least_99pct": candidate["benchmark_coverage"] >= 0.99,
        "capacity_feasible_at_least_95pct": candidate[
            "entry_capacity_feasible_rate"
        ]
        >= 0.95,
        "unresolved_exit_rate_at_most_1pct": unresolved_rate <= 0.01,
        "mean_net_return_at_least_75bp": (
            candidate.get("mean_net_return") or -math.inf
        )
        >= 0.0075,
        "mean_excess_at_least_50bp": (candidate_excess or -math.inf) >= 0.005,
        "excess_cluster_t_at_least_2_5": (
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 2.5,
        "at_least_5_positive_excess_years": candidate["positive_excess_years"] >= 5,
        "max_year_positive_share_at_most_50pct": (
            candidate.get("max_year_positive_excess_share") or math.inf
        )
        <= 0.50,
        "beats_low_score_control_by_1pct": (versus_control or -math.inf) >= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "unresolved_exit_rate": unresolved_rate,
        "candidate_excess_minus_low_score_control": versus_control,
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
    raw_panel = load_feature_panel(data_dir)
    events = build_ranked_events(attach_features(raw_panel))
    panel = forecast.prepare_panel(raw_panel)
    trades = forecast.build_trades(
        events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    trades = attach_market_excess(
        trades, build_market_benchmark(panel, HOLD_TRADING_DAYS)
    )
    summaries = {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=5_000,
            min_announcement_days=300,
        )
        for category in CATEGORIES
    }
    decision = evaluate(summaries)
    payload = {
        "schema_version": "p0-compression-volume-breakout-discovery-v1",
        "contract_frozen": "2026-09-03",
        "source_hypothesis_id": "ah-ai-7f12fe019bc805c0a5e1",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "signal_frequency": "last_trading_day_of_iso_week",
            "weights": {
                "low_boll_width": 0.30,
                "high_vol_ratio_10d": 0.40,
                "high_close_position": 0.30,
            },
            "candidate_minimum_score": 75.0,
            "control_maximum_score": 25.0,
            "maximum_names_per_side": 20,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
        },
        "data": {
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "events": events.height,
            "event_counts": {
                category: events.filter(pl.col("category") == category).height
                for category in CATEGORIES
            },
            "signal_days": events.get_column("ann_date").n_unique(),
        },
        "summaries": summaries,
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
            "/app/data/research/p0_compression_volume_breakout_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
