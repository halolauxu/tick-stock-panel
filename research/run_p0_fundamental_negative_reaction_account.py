"""Run the frozen fundamental-acceleration negative-reaction account screen."""

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
import run_p0_actual_corporate_buying_development as investable  # noqa: E402
import run_p0_confirmed_fundamental_acceleration_discovery as reaction  # noqa: E402
import run_p0_fundamental_acceleration_drift_discovery as fundamental  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_neglected_liquidity_premium as liquidity  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 20
TARGET_POSITIONS = 10

NEGATIVE_REACTION = "negative_reaction_candidate"
POSITIVE_REACTION = "mild_positive_reaction_control"


def build_reaction_events(data_dir: Path, raw: pl.DataFrame) -> pl.DataFrame:
    comparisons = fundamental.build_report_comparisons(
        fundamental.load_metrics(data_dir),
        start=DEVELOPMENT_START,
        end=DEVELOPMENT_END,
    )
    mother = fundamental.classify_events(comparisons).filter(
        pl.col("category") == fundamental.CANDIDATE
    )
    classified = reaction.classify_reactions(
        reaction.attach_first_reaction(mother, raw)
    )
    return classified.filter(
        pl.col("category").is_in(
            [reaction.NEGATIVE_CONTROL, reaction.CANDIDATE]
        )
    )


def attach_exact_investable_snapshot(
    events: pl.DataFrame, snapshots: pl.DataFrame
) -> pl.DataFrame:
    return events.join(
        snapshots.rename({"snapshot_date": "ann_date"}),
        on=["symbol", "ann_date"],
        how="inner",
    )


def build_candidates(
    events: pl.DataFrame,
    trading_dates: list[date],
    category: str,
) -> pl.DataFrame:
    if category == NEGATIVE_REACTION:
        source_category = reaction.NEGATIVE_CONTROL
        descending = False
    elif category == POSITIVE_REACTION:
        source_category = reaction.CANDIDATE
        descending = True
    else:
        raise ValueError(f"unknown category: {category}")
    scoped = investable.map_entry_dates(
        events.filter(pl.col("category") == source_category), trading_dates
    )
    return (
        scoped.sort(
            ["entry_date", "reaction_return", "symbol"],
            descending=[False, descending, False],
        )
        .unique(subset=["entry_date", "symbol"], keep="first", maintain_order=True)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            pl.col("ann_date").alias("date"),
            "entry_date",
            "symbol",
            "signal_amount",
            "cap_rank",
            "reaction_return",
            "report_announce_date",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
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
        "mean_cash_ratio_at_most_60pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.60,
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
        "beats_positive_control_by_5pp": annualized - control_annualized >= 0.05,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": "FREEZE_FOR_VALIDATION" if passed else "TERMINATE_FIXED_DEFINITION",
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
    panel = liquidity.attach_turnover_features(
        baseline.prepare_panel(
            baseline.attach_point_in_time_data(raw_all, data_dir)
        )
    )
    benchmark = shared.benchmark_metrics(
        liquidity.benchmark_universe(panel).filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )
    snapshots = investable.build_investable_snapshots(panel)
    del panel
    gc.collect()

    events = attach_exact_investable_snapshot(
        build_reaction_events(data_dir, raw_all), snapshots
    )
    del snapshots
    gc.collect()
    candidates = {
        NEGATIVE_REACTION: build_candidates(
            events, trading_dates, NEGATIVE_REACTION
        ),
        POSITIVE_REACTION: build_candidates(
            events, trading_dates, POSITIVE_REACTION
        ),
    }
    results = {
        name: fixed.simulate(
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
        for name, frame in candidates.items()
    }
    decision = evaluate(
        results[NEGATIVE_REACTION], results[POSITIVE_REACTION], benchmark
    )
    payload = {
        "schema_version": "p0-fundamental-negative-reaction-account-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "reaction_candidate_range": [-0.05, 0.0],
            "reaction_control_range": [0.0, 0.05],
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "target_positions": TARGET_POSITIONS,
            "initial_cash_cny": shared.INITIAL_CASH,
        },
        "data": {
            name: {
                "rows": frame.height,
                "symbols": frame.get_column("symbol").n_unique(),
                "entry_days": frame.get_column("entry_date").n_unique(),
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
            "/app/data/research/p0_fundamental_negative_reaction_account_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()

