"""Run the frozen main-board non-microcap neglected-liquidity account."""

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
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
TARGET_POSITIONS = 10
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
LOW_TURNOVER = "low_turnover_candidate"
HIGH_TURNOVER = "high_turnover_control"


def attach_turnover_features(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            (pl.col("amount") / pl.col("market_cap")).alias("turnover_proxy"),
            pl.col("_global_index").shift(19).over("symbol").alias("_index_19d"),
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_index_19d") + 19)
            .then(
                pl.col("turnover_proxy")
                .rolling_mean(window_size=20, min_samples=20)
                .over("symbol")
            )
            .otherwise(None)
            .alias("mean_turnover_20d")
        )
    )


def weekly_signal_panel(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[date]]:
    calendar = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%G-%V").alias("week"),
        )
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .filter(pl.col("signal_date") >= DEVELOPMENT_START)
    )
    signal = panel.join(
        calendar.select("signal_date", "entry_date"),
        left_on="date",
        right_on="signal_date",
        how="inner",
    )
    return signal, calendar.get_column("entry_date").to_list()


def rank_investable(signal: pl.DataFrame) -> pl.DataFrame:
    base = (
        signal.filter(
            (pl.col("market_cap") >= MIN_MARKET_CAP)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & pl.col("mean_turnover_20d").is_finite()
        )
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.len().over("date").alias("pre_size_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("pre_size_rank"),
        )
        .with_columns(
            (pl.col("pre_size_rank") / pl.col("pre_size_count")).alias(
                "market_cap_percentile"
            )
        )
        .filter(pl.col("market_cap_percentile") > 0.30)
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.len().over("date").alias("investable_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("investable_size_rank"),
        )
        .with_columns(
            (
                ((pl.col("investable_size_rank") - 1) * 5 / pl.col("investable_count"))
                .floor()
                .clip(0, 4)
                .cast(pl.UInt8)
            ).alias("size_bin")
        )
    )
    return (
        base.sort(["date", "size_bin", "mean_turnover_20d", "symbol"])
        .with_columns(
            pl.len().over(["date", "size_bin"]).alias("size_bin_count"),
            pl.col("mean_turnover_20d")
            .rank(method="ordinal")
            .over(["date", "size_bin"])
            .alias("turnover_rank"),
        )
        .with_columns(
            pl.when(pl.col("size_bin_count") > 1)
            .then(
                (pl.col("turnover_rank") - 1) / (pl.col("size_bin_count") - 1)
            )
            .otherwise(0.5)
            .alias("turnover_percentile")
        )
    )


def build_candidates(ranked: pl.DataFrame, direction: str) -> pl.DataFrame:
    if direction == LOW_TURNOVER:
        selected = ranked.filter(pl.col("turnover_percentile") <= 0.10)
        descending = [False, False, False, False]
    elif direction == HIGH_TURNOVER:
        selected = ranked.filter(pl.col("turnover_percentile") >= 0.90)
        descending = [False, True, True, False]
    else:
        raise ValueError(f"unknown direction: {direction}")
    return (
        selected.sort(
            ["date", "turnover_percentile", "mean_turnover_20d", "symbol"],
            descending=descending,
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "market_cap",
            "market_cap_percentile",
            "size_bin",
            "mean_turnover_20d",
            "turnover_percentile",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def benchmark_universe(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.filter(
            (pl.col("date") >= DEVELOPMENT_START)
            & (pl.col("market_cap") > 0)
            & pl.col("daily_return").is_finite()
        )
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.len().over("date").alias("day_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("day_size_rank"),
        )
        .filter(pl.col("day_size_rank") / pl.col("day_count") > 0.30)
    )


def evaluate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    control_annualized = float(control["metrics"].get("annualized") or -math.inf)
    complete_round_trips = int(candidate["account"].get("trade_count") or 0) // 2
    checks = {
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": float(metrics.get("max_drawdown") or -math.inf)
        >= -0.30,
        "at_least_5_positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "mean_cash_ratio_at_most_25pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.25,
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
        "at_least_100_round_trips": complete_round_trips >= 100,
        "beats_high_turnover_by_5pp": annualized - control_annualized >= 0.05,
    }
    passed = all(checks.values())
    return {
        "verdict": "FREEZE_CAPACITY_AND_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": annualized - benchmark_annualized,
        "annualized_minus_control": annualized - control_annualized,
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
        baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    )
    raw_source = raw_all.filter(pl.col("date") >= DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = attach_turnover_features(
        baseline.prepare_panel(
            baseline.attach_point_in_time_data(raw_all, data_dir)
        )
    )
    benchmark = shared.benchmark_metrics(benchmark_universe(panel))
    weekly, _action_dates = weekly_signal_panel(panel)
    ranked = rank_investable(weekly)
    del panel, weekly
    gc.collect()
    candidates = {
        direction: build_candidates(ranked, direction)
        for direction in (LOW_TURNOVER, HIGH_TURNOVER)
    }
    del ranked
    gc.collect()
    results = {}
    for direction, frame in candidates.items():
        results[direction] = fixed.simulate(
            frame,
            fixed.prepare_quotes(frame, raw_source, data_dir),
            all_dates,
            initial_cash=shared.INITIAL_CASH,
            target_positions=TARGET_POSITIONS,
            holding_trading_days=HOLD_TRADING_DAYS,
            maximum_exit_delay=MAX_EXIT_DELAY,
            period_start=DEVELOPMENT_START,
            period_end=DEVELOPMENT_END,
        )
    decision = evaluate(results[LOW_TURNOVER], results[HIGH_TURNOVER], benchmark)
    payload = {
        "schema_version": "p0-main-board-neglected-liquidity-premium-v2",
        "contract_frozen": "2026-09-03",
        "hypothesis_id": "ah-ai-3778d3454fb706a63bf9",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "market_cap_bottom_excluded": 0.30,
            "size_bins": 5,
            "turnover_tail": 0.10,
            "turnover_lookback_trading_days": 20,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "target_positions": TARGET_POSITIONS,
            "initial_cash_cny": shared.INITIAL_CASH,
        },
        "data": {
            direction: {
                "signal_rows": frame.height,
                "signal_symbols": frame.get_column("symbol").n_unique(),
                "rebalance_days": frame.get_column("entry_date").n_unique(),
                "minimum_market_cap_percentile": frame.get_column(
                    "market_cap_percentile"
                ).min(),
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
            "/app/data/research/p0_main_board_neglected_liquidity_premium_v2.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
