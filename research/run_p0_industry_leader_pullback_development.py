"""Run the frozen main-board industry-leader pullback development account."""

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
import run_p0_industry_momentum_development as industry  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_neglected_liquidity_premium as liquidity  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
TARGET_INDUSTRIES = 3
STOCKS_PER_INDUSTRY = 5
TARGET_POSITIONS = 10
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
MIN_INDUSTRY_MEMBERS = 10
MIN_INDUSTRY_MOMENTUM = 0.03
MIN_INDUSTRY_BREADTH = 0.60
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MARKET_CAP_PERCENTILE = 0.30
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MIN_MOMENTUM_60D = 0.05
MAX_MOMENTUM_60D = 0.50
PULLBACK_MIN = -0.08
PULLBACK_MAX = -0.01
CHASE_MIN = 0.01
CHASE_MAX = 0.08

PULLBACK = "industry_leader_pullback"
CHASE = "industry_leader_chase_control"


def attach_features(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        industry.attach_stock_features(panel)
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("_global_index").shift(5).over("symbol").alias("_index_5d"),
            pl.col("_global_index").shift(60).over("symbol").alias("_index_60d"),
            pl.col("close").shift(5).over("symbol").alias("_close_5d"),
            pl.col("close").shift(60).over("symbol").alias("_close_60d"),
            pl.col("close")
            .rolling_mean(60, min_samples=60)
            .over("symbol")
            .alias("ma60"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_index_5d") + 5)
            .then(pl.col("close") / pl.col("_close_5d") - 1.0)
            .otherwise(None)
            .alias("return_5d"),
            pl.when(pl.col("_global_index") == pl.col("_index_60d") + 60)
            .then(pl.col("close") / pl.col("_close_60d") - 1.0)
            .otherwise(None)
            .alias("return_60d"),
        )
    )


def rank_main_board_nonmicro(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.filter((pl.col("market_cap") > 0) & pl.col("mean_amount_20d").is_finite())
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.len().over("date").alias("day_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("size_rank"),
        )
        .with_columns(
            (pl.col("size_rank") / pl.col("day_count")).alias(
                "market_cap_percentile"
            )
        )
    )


def weekly_observations(panel: pl.DataFrame) -> pl.DataFrame:
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
    return panel.join(
        calendar.select("signal_date", "entry_date"),
        left_on="date",
        right_on="signal_date",
        how="inner",
    )


def select_industries(signal: pl.DataFrame) -> pl.DataFrame:
    return (
        signal.select(
            "date",
            "l1_code",
            "industry_member_count",
            "industry_momentum_20d",
            "industry_breadth_5d",
        )
        .unique(subset=["date", "l1_code"])
        .filter(
            (pl.col("industry_member_count") >= MIN_INDUSTRY_MEMBERS)
            & (pl.col("industry_momentum_20d") >= MIN_INDUSTRY_MOMENTUM)
            & (pl.col("industry_breadth_5d") >= MIN_INDUSTRY_BREADTH)
        )
        .sort(
            ["date", "industry_momentum_20d", "l1_code"],
            descending=[False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("industry_rank")
        )
        .filter(pl.col("industry_rank") <= TARGET_INDUSTRIES)
    )


def build_candidates(signal: pl.DataFrame, direction: str) -> pl.DataFrame:
    selected_industries = select_industries(signal)
    scoped = signal.join(
        selected_industries.select("date", "l1_code", "industry_rank"),
        on=["date", "l1_code"],
        how="inner",
    ).filter(
        (pl.col("market_cap_percentile") > MIN_MARKET_CAP_PERCENTILE)
        & (pl.col("market_cap") >= MIN_MARKET_CAP)
        & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
        & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        & pl.col("return_60d").is_between(
            MIN_MOMENTUM_60D, MAX_MOMENTUM_60D, closed="both"
        )
        & (pl.col("close") > pl.col("ma60"))
    )
    if direction == PULLBACK:
        selected = scoped.filter(
            pl.col("return_5d").is_between(PULLBACK_MIN, PULLBACK_MAX, closed="both")
        )
        descending = False
    elif direction == CHASE:
        selected = scoped.filter(
            pl.col("return_5d").is_between(CHASE_MIN, CHASE_MAX, closed="both")
        )
        descending = True
    else:
        raise ValueError(f"unknown direction: {direction}")
    return (
        selected.sort(
            ["date", "industry_rank", "return_5d", "market_cap", "symbol"],
            descending=[False, False, descending, False, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over(["date", "l1_code"])
            .alias("stock_rank")
        )
        .filter(pl.col("stock_rank") <= STOCKS_PER_INDUSTRY)
        .with_columns(
            (
                (pl.col("industry_rank") - 1) * STOCKS_PER_INDUSTRY
                + pl.col("stock_rank")
            ).alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "l1_code",
            "l1_name",
            "industry_rank",
            "stock_rank",
            "industry_momentum_20d",
            "industry_breadth_5d",
            "return_5d",
            "return_60d",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate(
    candidate: dict[str, Any], control: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    control_annualized = float(control["metrics"].get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    checks = {
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": float(metrics.get("max_drawdown") or -math.inf)
        >= -0.30,
        "at_least_5_positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "mean_cash_ratio_at_most_40pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.40,
        "at_least_300_round_trips": int(candidate["account"].get("trade_count") or 0)
        // 2
        >= 300,
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
        "cash_reconciled": candidate["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
        "beats_chase_control_by_5pp": annualized - control_annualized >= 0.05,
    }
    return {
        "passed": all(checks.values()),
        "verdict": "FREEZE_FOR_VALIDATION" if all(checks.values()) else "TERMINATE_INDUSTRY_FAMILY",
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
    trading_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = rank_main_board_nonmicro(
        attach_features(
            baseline.prepare_panel(
                baseline.attach_point_in_time_data(raw_all, data_dir)
            )
        )
    )
    benchmark = industry.benchmark_metrics(
        liquidity.benchmark_universe(panel).filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )
    membership, context = industry.load_industry_data(data_dir)
    signal = weekly_observations(industry.attach_industry(panel, membership, context))
    del panel, membership, context
    gc.collect()
    candidates = {
        PULLBACK: build_candidates(signal, PULLBACK),
        CHASE: build_candidates(signal, CHASE),
    }
    del signal
    gc.collect()
    results = {}
    for name, frame in candidates.items():
        results[name] = fixed.simulate(
            frame,
            fixed.prepare_quotes(frame, raw_source, data_dir),
            trading_dates,
            initial_cash=industry.INITIAL_CASH,
            target_positions=TARGET_POSITIONS,
            holding_trading_days=HOLD_TRADING_DAYS,
            maximum_exit_delay=MAX_EXIT_DELAY,
            period_start=DEVELOPMENT_START,
            period_end=DEVELOPMENT_END,
        )
    decision = evaluate(results[PULLBACK], results[CHASE], benchmark)
    payload = {
        "schema_version": "p0-industry-leader-pullback-development-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "target_industries": TARGET_INDUSTRIES,
            "stocks_per_industry": STOCKS_PER_INDUSTRY,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "initial_cash_cny": industry.INITIAL_CASH,
        },
        "data": {
            name: {
                "rows": frame.height,
                "symbols": frame.get_column("symbol").n_unique(),
                "entry_days": frame.get_column("entry_date").n_unique(),
                "industries": frame.get_column("l1_code").n_unique(),
            }
            for name, frame in candidates.items()
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
            "/app/data/research/p0_industry_leader_pullback_development_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
