"""Run the frozen main-board actual-corporate-buying development account."""

from __future__ import annotations

import argparse
import bisect
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
import run_p0_holder_increase_development as holder  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_neglected_liquidity_premium as liquidity  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_repurchase_drift_development as repurchase  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MIN_MARKET_CAP_PERCENTILE = 0.30
MAX_SNAPSHOT_AGE_DAYS = 7
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
TARGET_POSITIONS = 10

HOLDER_STRONG = "holder_strong"
HOLDER_WEAK = "holder_weak_control"
REPURCHASE_STRONG = "repurchase_strong"
REPURCHASE_WEAK = "repurchase_weak_control"
POOLED_STRONG = "pooled_strong"
POOLED_WEAK = "pooled_weak_control"

HOLDER_STRONG_FLOAT_FRACTION = 0.005
HOLDER_WEAK_FLOAT_FRACTION = 0.0005
REPURCHASE_STRONG_MCAP_FRACTION = 0.0015
REPURCHASE_STRONG_TURNOVER_DAYS = 0.15
REPURCHASE_WEAK_MCAP_FRACTION = 0.0003
REPURCHASE_WEAK_TURNOVER_DAYS = 0.035


def build_investable_snapshots(panel: pl.DataFrame) -> pl.DataFrame:
    ranked = (
        panel.filter((pl.col("market_cap") > 0) & pl.col("mean_turnover_20d").is_finite())
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
    return (
        ranked.filter(
            (pl.col("market_cap_percentile") > MIN_MARKET_CAP_PERCENTILE)
            & (pl.col("market_cap") >= MIN_MARKET_CAP)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .select(
            "symbol",
            pl.col("date").alias("snapshot_date"),
            "market_cap",
            "market_cap_percentile",
            "mean_amount_20d",
            pl.col("amount").alias("signal_amount"),
        )
        .sort(["symbol", "snapshot_date"])
    )


def attach_snapshots(events: pl.DataFrame, snapshots: pl.DataFrame) -> pl.DataFrame:
    return (
        events.sort(["symbol", "ann_date"])
        .join_asof(
            snapshots,
            left_on="ann_date",
            right_on="snapshot_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("snapshot_date").is_not_null()
            & (
                (pl.col("ann_date") - pl.col("snapshot_date")).dt.total_days()
                <= MAX_SNAPSHOT_AGE_DAYS
            )
        )
    )


def classify_holder_events(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.filter(pl.col("category").is_in(holder.POSITIVE_CATEGORIES))
        .with_columns(
            (pl.col("increase_float_ratio_pct") / 100.0).alias(
                "capital_intensity"
            )
        )
        .with_columns(
            pl.when(pl.col("capital_intensity") >= HOLDER_STRONG_FLOAT_FRACTION)
            .then(pl.lit(HOLDER_STRONG))
            .when(pl.col("capital_intensity") <= HOLDER_WEAK_FLOAT_FRACTION)
            .then(pl.lit(HOLDER_WEAK))
            .otherwise(None)
            .alias("signal_group")
        )
        .filter(pl.col("signal_group").is_not_null())
        .with_columns(pl.lit("holder_increase").alias("family"))
    )


def classify_repurchase_events(events: pl.DataFrame) -> pl.DataFrame:
    work = (
        events.filter(pl.col("category") == "completion")
        .filter(pl.col("repurchase_amount_cny").is_not_null())
        .with_columns(
            (pl.col("repurchase_amount_cny") / pl.col("market_cap")).alias(
                "mcap_fraction"
            ),
            (pl.col("repurchase_amount_cny") / pl.col("mean_amount_20d")).alias(
                "turnover_days"
            ),
        )
    )
    return (
        work.with_columns(
            pl.when(
                (pl.col("mcap_fraction") >= REPURCHASE_STRONG_MCAP_FRACTION)
                & (pl.col("turnover_days") >= REPURCHASE_STRONG_TURNOVER_DAYS)
            )
            .then(pl.lit(REPURCHASE_STRONG))
            .when(
                (pl.col("mcap_fraction") <= REPURCHASE_WEAK_MCAP_FRACTION)
                & (pl.col("turnover_days") <= REPURCHASE_WEAK_TURNOVER_DAYS)
            )
            .then(pl.lit(REPURCHASE_WEAK))
            .otherwise(None)
            .alias("signal_group"),
            pl.min_horizontal(
                pl.col("mcap_fraction") / REPURCHASE_STRONG_MCAP_FRACTION,
                pl.col("turnover_days") / REPURCHASE_STRONG_TURNOVER_DAYS,
            ).alias("capital_intensity"),
            pl.lit("repurchase_completion").alias("family"),
        ).filter(pl.col("signal_group").is_not_null())
    )


def map_entry_dates(events: pl.DataFrame, trading_dates: list[date]) -> pl.DataFrame:
    mapped = []
    for ann_date in events.get_column("ann_date").to_list():
        index = bisect.bisect_right(trading_dates, ann_date)
        mapped.append(trading_dates[index] if index < len(trading_dates) else None)
    return events.with_columns(pl.Series("entry_date", mapped, dtype=pl.Date)).drop_nulls(
        "entry_date"
    )


def build_candidates(
    events: pl.DataFrame, trading_dates: list[date], groups: tuple[str, ...]
) -> pl.DataFrame:
    scoped = map_entry_dates(
        events.filter(pl.col("signal_group").is_in(groups)), trading_dates
    )
    descending = groups[0] in (HOLDER_STRONG, REPURCHASE_STRONG)
    return (
        scoped.sort(
            ["entry_date", "capital_intensity", "symbol"],
            descending=[False, descending, False],
        )
        .unique(subset=["entry_date", "symbol"], keep="first", maintain_order=True)
        .with_columns(pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            pl.col("ann_date").alias("date"),
            "entry_date",
            "symbol",
            "signal_amount",
            "cap_rank",
            "family",
            "capital_intensity",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _evaluate(
    strong: dict[str, Any],
    weak: dict[str, Any],
    holder_strong: dict[str, Any],
    repurchase_strong: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    metrics = strong["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    weak_annualized = float(weak["metrics"].get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    arm_checks = {
        name: {
            "at_least_100_round_trips": int(result["account"].get("trade_count") or 0)
            // 2
            >= 100,
            "buy_execution_at_least_90pct": result["execution"]["buy"][
                "execution_rate"
            ]
            >= 0.90,
            "sell_execution_at_least_90pct": result["execution"]["sell"][
                "execution_rate"
            ]
            >= 0.90,
            "no_unresolved_positions": result["integrity"][
                "ending_unresolved_positions"
            ]
            == 0,
            "at_least_5_positive_years": int(result["metrics"].get("positive_years") or 0)
            >= 5,
        }
        for name, result in (
            (HOLDER_STRONG, holder_strong),
            (REPURCHASE_STRONG, repurchase_strong),
        )
    }
    checks = {
        "both_strong_arms_pass_integrity": all(
            all(values.values()) for values in arm_checks.values()
        ),
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": float(metrics.get("max_drawdown") or -math.inf)
        >= -0.30,
        "mean_cash_ratio_at_most_60pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.60,
        "beats_weak_control_by_5pp": annualized - weak_annualized >= 0.05,
        "cash_reconciled": strong["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
        "no_unresolved_positions": strong["integrity"]["ending_unresolved_positions"]
        == 0,
    }
    passed = all(checks.values())
    return {
        "verdict": "FREEZE_CAPACITY_AND_VALIDATION" if passed else "TERMINATE_FIXED_DEFINITION",
        "passed": passed,
        "annualized_excess": annualized - benchmark_annualized,
        "annualized_minus_weak_control": annualized - weak_annualized,
        "arm_checks": arm_checks,
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
    raw = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    )
    raw_source = raw.filter(pl.col("date") >= DEVELOPMENT_START)
    trading_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = liquidity.attach_turnover_features(
        baseline.prepare_panel(baseline.attach_point_in_time_data(raw, data_dir))
    )
    benchmark = shared.benchmark_metrics(
        liquidity.benchmark_universe(panel).filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )
    snapshots = build_investable_snapshots(panel)
    del panel
    gc.collect()

    holder_events = classify_holder_events(
        attach_snapshots(
            holder.aggregate_events(holder.load_holder_trades(data_dir)), snapshots
        )
    )
    repurchase_events = classify_repurchase_events(
        attach_snapshots(
            repurchase.categorize_events(repurchase.load_repurchase_events(data_dir)),
            snapshots,
        )
    )
    events = pl.concat(
        [holder_events, repurchase_events], how="diagonal_relaxed"
    ).sort(["ann_date", "symbol", "family"])
    del snapshots
    gc.collect()

    candidate_frames = {
        HOLDER_STRONG: build_candidates(events, trading_dates, (HOLDER_STRONG,)),
        HOLDER_WEAK: build_candidates(events, trading_dates, (HOLDER_WEAK,)),
        REPURCHASE_STRONG: build_candidates(events, trading_dates, (REPURCHASE_STRONG,)),
        REPURCHASE_WEAK: build_candidates(events, trading_dates, (REPURCHASE_WEAK,)),
        POOLED_STRONG: build_candidates(
            events, trading_dates, (HOLDER_STRONG, REPURCHASE_STRONG)
        ),
        POOLED_WEAK: build_candidates(
            events, trading_dates, (HOLDER_WEAK, REPURCHASE_WEAK)
        ),
    }
    results = {}
    for name, frame in candidate_frames.items():
        results[name] = fixed.simulate(
            frame,
            fixed.prepare_quotes(frame, raw_source, data_dir),
            trading_dates,
            initial_cash=shared.INITIAL_CASH,
            target_positions=TARGET_POSITIONS,
            holding_trading_days=HOLD_TRADING_DAYS,
            maximum_exit_delay=MAX_EXIT_DELAY,
            period_start=DEVELOPMENT_START,
            period_end=DEVELOPMENT_END,
        )
    decision = _evaluate(
        results[POOLED_STRONG],
        results[POOLED_WEAK],
        results[HOLDER_STRONG],
        results[REPURCHASE_STRONG],
        benchmark,
    )
    payload = {
        "schema_version": "p0-actual-corporate-buying-development-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "target_positions": TARGET_POSITIONS,
            "initial_cash_cny": shared.INITIAL_CASH,
            "holder_strong_float_fraction": HOLDER_STRONG_FLOAT_FRACTION,
            "holder_weak_float_fraction": HOLDER_WEAK_FLOAT_FRACTION,
            "repurchase_strong_mcap_fraction": REPURCHASE_STRONG_MCAP_FRACTION,
            "repurchase_strong_turnover_days": REPURCHASE_STRONG_TURNOVER_DAYS,
            "repurchase_weak_mcap_fraction": REPURCHASE_WEAK_MCAP_FRACTION,
            "repurchase_weak_turnover_days": REPURCHASE_WEAK_TURNOVER_DAYS,
        },
        "data": {
            "event_rows": events.height,
            "event_symbols": events.get_column("symbol").n_unique(),
            "groups": {
                row["signal_group"]: row["len"]
                for row in events.group_by("signal_group").len().to_dicts()
            },
            "candidates": {
                name: {
                    "rows": frame.height,
                    "symbols": frame.get_column("symbol").n_unique(),
                    "entry_days": frame.get_column("entry_date").n_unique(),
                }
                for name, frame in candidate_frames.items()
            },
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
            "/app/data/research/p0_actual_corporate_buying_development_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
