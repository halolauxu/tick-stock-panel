"""Run the frozen development-only low-attention forecast drift study."""
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

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
PRIMARY_HORIZON = 10
DIAGNOSTIC_HORIZONS = (2, 5)
MAX_EXIT_DELAY = 20
LOW_ATTENTION_MAX = 1.0 / 3.0
HIGH_ATTENTION_MIN = 2.0 / 3.0
MAIN_BOARD_PATTERN = r"^(?:600|601|603|605|000|001|002)\d{3}\.(?:SH|SZ)$"

CANDIDATE = "low_attention_positive"
HIGH_CONTROL = "high_attention_positive_control"
NEGATIVE_CONTROL = "low_attention_negative_control"
CATEGORIES = (CANDIDATE, HIGH_CONTROL, NEGATIVE_CONTROL)


def load_main_board_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    raw = (
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
            "turnover_rate",
        )
        .filter(
            pl.col("date").is_between(START, PANEL_END, closed="both")
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
        .collect(engine="streaming")
    )
    return forecast.attach_point_in_time_universe(raw, data_dir)


def attach_prior_attention(panel: pl.DataFrame) -> pl.DataFrame:
    work = (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("turnover_rate")
            .shift(1)
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("prior_turnover_20d")
        )
        .with_columns(
            pl.col("prior_turnover_20d")
            .rank(method="average")
            .over("date")
            .alias("_attention_rank"),
            pl.col("prior_turnover_20d").count().over("date").alias("_attention_n"),
        )
        .with_columns(
            (pl.col("_attention_rank") / pl.col("_attention_n")).alias(
                "attention_percentile"
            )
        )
    )
    return work


def build_events(raw_events: pl.DataFrame, attention: pl.DataFrame) -> pl.DataFrame:
    categorized = forecast.categorize_events(raw_events).filter(
        pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        & pl.col("category").is_in(
            ["growth_0_50", "growth_50_100", "negative_control"]
        )
    )
    attention_lookup = attention.select(
        "symbol",
        pl.col("date").alias("attention_date"),
        "prior_turnover_20d",
        "attention_percentile",
    ).sort(["symbol", "attention_date"])
    joined = categorized.sort(["symbol", "ann_date"]).join_asof(
        attention_lookup,
        left_on="ann_date",
        right_on="attention_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    positive = pl.col("category").is_in(["growth_0_50", "growth_50_100"])
    category = (
        pl.when(positive & (pl.col("attention_percentile") <= LOW_ATTENTION_MAX))
        .then(pl.lit(CANDIDATE))
        .when(positive & (pl.col("attention_percentile") >= HIGH_ATTENTION_MIN))
        .then(pl.lit(HIGH_CONTROL))
        .when(
            (pl.col("category") == "negative_control")
            & (pl.col("attention_percentile") <= LOW_ATTENTION_MAX)
        )
        .then(pl.lit(NEGATIVE_CONTROL))
        .otherwise(None)
    )
    return (
        joined.with_columns(category.alias("study_category"))
        .filter(pl.col("study_category").is_not_null())
        .drop("category")
        .rename({"study_category": "category"})
        .sort(["ann_date", "category", "symbol"])
    )


def _summaries(
    events: pl.DataFrame, panel: pl.DataFrame, horizon: int
) -> dict[str, dict[str, Any]]:
    trades = forecast.build_trades(
        events,
        panel,
        holding_trading_days=horizon,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    trades = attach_market_excess(trades, build_market_benchmark(panel, horizon))
    return {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=500,
            min_announcement_days=150,
        )
        for category in CATEGORIES
    }


def evaluate(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = primary[CANDIDATE]
    high = primary[HIGH_CONTROL]
    negative = primary[NEGATIVE_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    high_excess = high.get("mean_excess_return")
    negative_excess = negative.get("mean_excess_return")
    versus_high = (
        candidate_excess - high_excess
        if candidate_excess is not None and high_excess is not None
        else None
    )
    checks = {
        "at_least_500_tradable_events": candidate["tradable_events"] >= 500,
        "at_least_150_announcement_days": candidate["announcement_days"] >= 150,
        "tradable_rate_at_least_90pct": candidate["tradable_rate"] >= 0.90,
        "no_unresolved_exits": candidate["unresolved_exits"] == 0,
        "mean_net_return_at_least_75bp": (candidate.get("mean_net_return") or -math.inf)
        >= 0.0075,
        "mean_excess_at_least_50bp": (candidate_excess or -math.inf) >= 0.005,
        "announcement_cluster_t_at_least_2_5": (
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 2.5,
        "at_least_5_positive_years": candidate["positive_excess_years"] >= 5,
        "beats_high_attention_by_25bp": (versus_high or -math.inf) >= 0.0025,
        "beats_low_attention_negative_control": (
            candidate_excess is not None
            and negative_excess is not None
            and candidate_excess > negative_excess
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "candidate_excess_minus_high_attention": versus_high,
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
    attention_panel = attach_prior_attention(load_main_board_panel(data_dir))
    events = build_events(raw_events, attention_panel)
    panel = forecast.prepare_panel(attention_panel)
    primary = _summaries(events, panel, PRIMARY_HORIZON)
    diagnostics = {
        str(horizon): _summaries(events, panel, horizon)
        for horizon in DIAGNOSTIC_HORIZONS
    }
    decision = evaluate(primary)
    payload = {
        "schema_version": "p0-low-attention-forecast-drift-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "attention": "prior_20_trading_day_mean_turnover_cross_sectional_percentile",
            "low_attention_max_percentile": LOW_ATTENTION_MAX,
            "high_attention_min_percentile": HIGH_ATTENTION_MIN,
            "primary_holding_trading_days": PRIMARY_HORIZON,
            "diagnostic_holding_trading_days": list(DIAGNOSTIC_HORIZONS),
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
        },
        "data": {
            "raw_forecast_rows": raw_events.height,
            "categorized_study_events": events.height,
            "event_symbols": events.get_column("symbol").n_unique(),
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
            "/app/data/research/p0_low_attention_forecast_drift_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()

