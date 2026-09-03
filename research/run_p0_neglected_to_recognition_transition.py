"""Run the frozen neglected-to-recognition transition account study."""

from __future__ import annotations

import argparse
import gc
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
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import fixed_horizon_account as fixed  # noqa: E402
import run_p0_main_board_neglected_liquidity_premium as base  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

TRANSITION = "neglected_to_recognition"
STILL_NEGLECTED = "still_neglected_control"
MOTHER = "low_turnover_mother"


def attach_transition_features(panel: pl.DataFrame) -> pl.DataFrame:
    featured = base.attach_turnover_features(panel).with_columns(
        pl.col("turnover_proxy")
        .rolling_mean(window_size=5, min_samples=5)
        .over("symbol")
        .alias("mean_turnover_recent_5d"),
        pl.col("turnover_proxy")
        .rolling_mean(window_size=15, min_samples=15)
        .over("symbol")
        .alias("_mean_turnover_15d_including_today"),
        pl.col("close").shift(5).over("symbol").alias("_close_5d"),
        pl.col("_global_index").shift(5).over("symbol").alias("_index_5d"),
    )
    return featured.with_columns(
        pl.col("_mean_turnover_15d_including_today")
        .shift(5)
        .over("symbol")
        .alias("mean_turnover_prior_15d"),
        pl.when(pl.col("_global_index") == pl.col("_index_5d") + 5)
        .then(pl.col("close") / pl.col("_close_5d") - 1.0)
        .otherwise(None)
        .alias("return_5d"),
    ).with_columns(
        (pl.col("mean_turnover_recent_5d") / pl.col("mean_turnover_prior_15d")).alias(
            "turnover_transition_ratio"
        )
    )


def build_transition_candidates(ranked: pl.DataFrame, direction: str) -> pl.DataFrame:
    low_turnover = ranked.filter(pl.col("turnover_percentile") <= 0.10)
    if direction == TRANSITION:
        selected = low_turnover.filter(
            pl.col("turnover_transition_ratio").is_between(1.2, 2.0, closed="both")
            & pl.col("return_5d").is_between(0.0, 0.05, closed="both")
        )
        sort_columns = [
            "date",
            "turnover_transition_ratio",
            "return_5d",
            "turnover_percentile",
            "symbol",
        ]
        descending = [False, True, True, False, False]
    elif direction == STILL_NEGLECTED:
        selected = low_turnover.filter(
            (pl.col("turnover_transition_ratio") <= 1.0)
            & (pl.col("return_5d") <= 0.0)
        )
        sort_columns = [
            "date",
            "turnover_transition_ratio",
            "return_5d",
            "turnover_percentile",
            "symbol",
        ]
        descending = [False, False, False, False, False]
    else:
        raise ValueError(f"unknown direction: {direction}")
    return (
        selected.sort(sort_columns, descending=descending)
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= base.TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "market_cap",
            "market_cap_percentile",
            "size_bin",
            "mean_turnover_20d",
            "turnover_percentile",
            "turnover_transition_ratio",
            "return_5d",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate(
    candidate: dict[str, Any],
    mother: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    mother_annualized = float(mother["metrics"].get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    drawdown = float(metrics.get("max_drawdown") or -math.inf)
    mother_drawdown = float(mother["metrics"].get("max_drawdown") or -math.inf)
    checks = {
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": drawdown >= -0.30,
        "at_least_5_positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "mean_cash_ratio_at_most_50pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.50,
        "buy_execution_at_least_90pct": candidate["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_execution_at_least_90pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "no_unresolved_positions": candidate["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciled": candidate["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 0.01,
        "at_least_100_round_trips": int(candidate["account"].get("trade_count") or 0)
        // 2
        >= 100,
        "annualized_improves_mother_by_5pp": annualized - mother_annualized >= 0.05,
        "drawdown_improves_mother_by_5pp": drawdown - mother_drawdown >= 0.05,
    }
    passed = all(checks.values())
    return {
        "verdict": "FREEZE_CAPACITY_AND_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": annualized - benchmark_annualized,
        "annualized_minus_mother": annualized - mother_annualized,
        "drawdown_improvement_vs_mother": drawdown - mother_drawdown,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
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
    raw_all = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=base.DEVELOPMENT_END)
    )
    raw_source = raw_all.filter(pl.col("date") >= base.DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = attach_transition_features(
        baseline.prepare_panel(
            baseline.attach_point_in_time_data(raw_all, data_dir)
        )
    )
    benchmark = base.shared.benchmark_metrics(base.benchmark_universe(panel))
    weekly, _action_dates = base.weekly_signal_panel(panel)
    ranked = base.rank_investable(weekly)
    del panel, weekly
    gc.collect()
    candidates = {
        TRANSITION: build_transition_candidates(ranked, TRANSITION),
        STILL_NEGLECTED: build_transition_candidates(ranked, STILL_NEGLECTED),
        MOTHER: base.build_candidates(ranked, base.LOW_TURNOVER),
    }
    del ranked
    gc.collect()
    results = {}
    for direction, frame in candidates.items():
        results[direction] = fixed.simulate(
            frame,
            fixed.prepare_quotes(frame, raw_source, data_dir),
            all_dates,
            initial_cash=base.shared.INITIAL_CASH,
            target_positions=base.TARGET_POSITIONS,
            holding_trading_days=base.HOLD_TRADING_DAYS,
            maximum_exit_delay=base.MAX_EXIT_DELAY,
            period_start=base.DEVELOPMENT_START,
            period_end=base.DEVELOPMENT_END,
        )
    decision = evaluate(results[TRANSITION], results[MOTHER], benchmark)
    payload = {
        "schema_version": "p0-neglected-to-recognition-transition-v2",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": base.DEVELOPMENT_START,
            "end": base.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "mother": "main_board_non_microcap_low_turnover",
            "recent_turnover_days": 5,
            "prior_turnover_days": 15,
            "candidate_turnover_ratio": [1.2, 2.0],
            "candidate_return_5d": [0.0, 0.05],
            "holding_trading_days": base.HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": base.MAX_EXIT_DELAY,
            "target_positions": base.TARGET_POSITIONS,
            "initial_cash_cny": base.shared.INITIAL_CASH,
        },
        "data": {
            direction: {
                "signal_rows": frame.height,
                "signal_symbols": frame.get_column("symbol").n_unique(),
                "rebalance_days": frame.get_column("entry_date").n_unique(),
            }
            for direction, frame in candidates.items()
        },
        "benchmark": benchmark,
        "results": results,
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
            "/app/data/research/p0_neglected_to_recognition_transition_v2.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
