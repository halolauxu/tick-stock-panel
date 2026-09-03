"""Run the frozen V2 main-board micro-cap short-horizon ablations.

The first formal run is M1 development only: carry the frozen weekly micro-cap
selection through each trading day and force a sell attempt on holding session
10.  Validation and known-stress dates are deliberately not loaded by this
entrypoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_short_horizon_baseline_audit as lifecycle_audit  # noqa: E402

SCHEMA_VERSION = "p0-main-board-short-horizon-v2-m1-development-v1"
INITIAL_CASH = 200_000.0
MAX_HOLDING_SESSIONS = 10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_delist_dates(data_dir: Path, symbols: list[str]) -> dict[str, date]:
    master = data_dir / "research" / "historical_stock_universe_all_a.parquet"
    if not master.is_file():
        raise ValueError("all-A PIT security master is required")
    rows = (
        pl.read_parquet(master)
        .with_columns(pl.col("delist_date").cast(pl.Date, strict=False))
        .filter(
            pl.col("symbol").is_in(symbols)
            & pl.col("delist_date").is_not_null()
        )
        .select("symbol", "delist_date")
        .unique(subset=["symbol"], keep="last")
        .to_dicts()
    )
    return {row["symbol"]: row["delist_date"] for row in rows}


def expand_weekly_targets(
    weekly_candidates: pl.DataFrame,
    action_dates: list[date],
) -> pl.DataFrame:
    """Carry each weekly point-in-time selection until the next rebalance."""

    if weekly_candidates.is_empty():
        return weekly_candidates.with_columns(
            pl.col("entry_date").alias("source_entry_date")
        )
    grouped = account._partition_rows(weekly_candidates, "entry_date")
    current: pl.DataFrame | None = None
    carried: list[pl.DataFrame] = []
    for action_date in sorted(set(action_dates)):
        replacement = grouped.get(action_date)
        if replacement is not None:
            current = replacement
        if current is None:
            continue
        carried.append(
            current.with_columns(
                pl.col("entry_date").alias("source_entry_date"),
                pl.lit(action_date).cast(pl.Date).alias("entry_date"),
            )
        )
    if not carried:
        return weekly_candidates.with_columns(
            pl.col("entry_date").alias("source_entry_date")
        ).head(0)
    return pl.concat(carried).sort(["entry_date", "cap_rank", "symbol"])


def build_lifecycle_report(
    simulation: dict[str, Any],
    trading_dates: list[date],
) -> dict[str, Any]:
    reconstructed = lifecycle_audit.reconstruct_lifecycles(
        simulation["orders"],
        simulation["settlements"],
        trading_dates,
        default_family="main_board_microcap",
    )
    cycles = reconstructed["cycles"]
    normal_closed = [
        row for row in cycles if row["closed"] and row["exit_type"] == "SELL"
    ]
    max_hold_exits = {
        (str(row["symbol"]), str(row["date"]))
        for row in simulation["orders"]
        if row.get("side") == "SELL"
        and row.get("status") == "FILLED"
        and row.get("exit_trigger") == "max_holding_sessions"
    }
    over_10 = [
        row for row in normal_closed if int(row["holding_sessions"]) > 10
    ]
    unexpected_over_10 = [
        row
        for row in over_10
        if (str(row["symbol"]), str(row["exit_date"])) not in max_hold_exits
    ]
    summary = lifecycle_audit.summarize_lifecycles(cycles)
    summary.update(
        {
            "normal_closed_cycles": len(normal_closed),
            "normal_under_2_cycles": sum(
                int(row["holding_sessions"]) < 2 for row in normal_closed
            ),
            "delayed_max_hold_exit_cycles": len(over_10) - len(unexpected_over_10),
            "unexpected_over_10_cycles": len(unexpected_over_10),
            "reconstruction_issues": reconstructed["issues"],
        }
    )
    return summary


def evaluate_development_gate(
    candidate: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    yearly = {
        int(row["year"]): row.get("account_return")
        for row in metrics.get("yearly", [])
    }
    lifecycle = candidate["lifecycle"]
    checks = {
        "annualized_within_5pp_of_baseline": (
            metrics.get("account_annualized") is not None
            and metrics["account_annualized"]
            >= baseline_metrics["account_annualized"] - 0.05
        ),
        "drawdown_improves_at_least_5pp": (
            metrics.get("account_max_drawdown") is not None
            and metrics["account_max_drawdown"]
            >= baseline_metrics["account_max_drawdown"] + 0.05
        ),
        "2017_and_2018_not_both_negative": (
            yearly.get(2017) is not None
            and yearly.get(2018) is not None
            and not (yearly[2017] < 0 and yearly[2018] < 0)
        ),
        "normal_holds_not_under_2_sessions": (
            lifecycle["normal_under_2_cycles"] == 0
        ),
        "no_unexpected_holds_over_10_sessions": (
            lifecycle["unexpected_over_10_cycles"] == 0
            and lifecycle["open_over_10_cycles"] == 0
        ),
        "buy_execution_at_least_80pct": (
            candidate["execution"]["buy"]["execution_rate"] >= 0.80
        ),
        "sell_execution_at_least_80pct": (
            candidate["execution"]["sell"]["execution_rate"] >= 0.80
        ),
        "no_unresolved_ending_marks": (
            candidate["integrity"]["ending_unresolved_positions"] == 0
        ),
        "cash_reconciled": (
            candidate["integrity"]["max_cash_reconciliation_error"] <= 0.01
        ),
        "lifecycle_reconstruction_clean": not lifecycle[
            "reconstruction_issues"
        ],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "verdict": "PASS_TO_M2" if not failures else "REJECT_M1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def run_m1_development(
    data_dir: Path,
    baseline_result: Path,
    output: Path,
) -> dict[str, Any]:
    source = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=baseline.DEVELOPMENT_END)
    )
    if source.is_empty():
        raise ValueError("no main-board development data")
    all_dates = source.get_column("date").unique().sort().to_list()
    source_rows = source.height
    source_symbols = source.get_column("symbol").n_unique()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()

    weekly_candidates = account.build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    candidate_symbols = weekly_candidates.get_column("symbol").unique().to_list()
    del panel, observations
    gc.collect()

    daily_candidates = expand_weekly_targets(weekly_candidates, all_dates)
    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=baseline.DEVELOPMENT_END)
    ).filter(pl.col("symbol").is_in(candidate_symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(daily_candidates, quotes)
    delist_dates = load_delist_dates(data_dir, candidate_symbols)

    simulation = account.simulate_account(
        daily_candidates,
        execution_grid,
        initial_cash=INITIAL_CASH,
        action_dates=all_dates,
        delist_dates=delist_dates,
        settle_only_after_delist_date=True,
        max_holding_sessions=MAX_HOLDING_SESSIONS,
    )
    daily, stale = account.build_daily_equity(
        simulation,
        quotes,
        all_dates,
        initial_cash=INITIAL_CASH,
    )
    metric = next(
        row
        for row in account.account_period_metrics(daily, weekly_market)
        if row["period"] == "development"
    )
    result = {
        "period": "development",
        "first_date": all_dates[0],
        "last_date": all_dates[-1],
        "metrics": metric,
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "lifecycle": build_lifecycle_report(simulation, all_dates),
        "account": account.account_summary(simulation, daily),
        "daily_equity": daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts(),
        "orders": simulation["orders"],
        "settlements": simulation["settlements"],
        "worst_weeks": account.worst_weeks(daily),
        "drawdown_episode": main_board.drawdown_episode(
            daily.select("date", "equity").to_dicts()
        ),
    }

    baseline_payload = json.loads(baseline_result.read_text(encoding="utf-8"))
    baseline_development = baseline_payload["accounts"]["200000"]["periods"][
        "development"
    ]["metrics"]
    decision = evaluate_development_gate(result, baseline_development)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "stage": "R1-02_M1_DEVELOPMENT_ONLY",
            "board_scope": "sh_sz_main_board_only",
            "initial_cash": INITIAL_CASH,
            "selection": "weekly_point_in_time_microcap_carried_daily",
            "max_holding_sessions": MAX_HOLDING_SESSIONS,
            "cooldown_sessions": 0,
            "entry_gate": None,
            "execution": "next_trade_day_open_sells_before_buys",
            "validation_accessed": False,
            "known_stress_accessed": False,
        },
        "sources": {
            "baseline_result": {
                "path": str(baseline_result),
                "sha256": _sha256(baseline_result),
            }
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "source_rows": source_rows,
            "source_symbols": source_symbols,
            "candidate_symbols": len(candidate_symbols),
            "weekly_signal_rows": weekly_candidates.height,
            "daily_carried_signal_rows": daily_candidates.height,
            "candidate_delist_dates": len(delist_dates),
        },
        "baseline_development_metrics": baseline_development,
        "result": result,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=main_board._json_default,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "data": payload["data"],
                "metrics": result["metrics"],
                "execution": result["execution"],
                "integrity": result["integrity"],
                "lifecycle": result["lifecycle"],
                "decision": decision,
                "output": str(output),
                "sha256": _sha256(output),
            },
            ensure_ascii=False,
            indent=2,
            default=main_board._json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("m1-development",), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_m1_development(args.data_dir, args.baseline_result, args.output)


if __name__ == "__main__":
    main()
