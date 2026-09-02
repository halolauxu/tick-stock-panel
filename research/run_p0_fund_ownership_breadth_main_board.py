"""Run the frozen main-board fund-ownership-breadth development account."""

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

import run_p0_fund_ownership_breadth_development as breadth  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = CAPITALS[0]


def evaluate(
    tiers: list[dict[str, Any]], benchmark: dict[str, Any]
) -> dict[str, Any]:
    primary = next(row for row in tiers if row["capital"] == PRIMARY_CAPITAL)
    metrics = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else -math.inf
    )
    checks = {
        "annualized_at_least_15pct": (annualized or -math.inf) >= 0.15,
        "excess_at_least_5pp": excess >= 0.05,
        "max_drawdown_within_25pct": (
            metrics.get("max_drawdown") or -math.inf
        )
        >= -0.25,
        "at_least_two_positive_signal_years": metrics["positive_years"] >= 2,
        "mean_cash_ratio_at_most_25pct": (
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.25,
        "buy_execution_at_least_90pct": (
            execution["buy"]["execution_rate"] >= 0.90
        ),
        "sell_execution_at_least_90pct": (
            execution["sell"]["execution_rate"] >= 0.90
        ),
        "no_unresolved_positions": (
            integrity["ending_unresolved_positions"] == 0
        ),
        "cash_reconciled": (
            integrity["max_cash_reconciliation_error"] <= 0.01
        ),
    }
    capacity = {
        str(int(row["capital"])): {
            "buy_execution_at_least_90pct": (
                row["execution"]["buy"]["execution_rate"] >= 0.90
            ),
            "sell_execution_at_least_90pct": (
                row["execution"]["sell"]["execution_rate"] >= 0.90
            ),
            "no_unresolved_positions": (
                row["integrity"]["ending_unresolved_positions"] == 0
            ),
            "cash_reconciled": (
                row["integrity"]["max_cash_reconciliation_error"] <= 0.01
            ),
        }
        for row in tiers
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION_DATA" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": excess if math.isfinite(excess) else None,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "capacity_checks": capacity,
        "validation_read": False,
        "known_stress_read": False,
    }


def _summary(result: dict[str, Any], capital: float) -> dict[str, Any]:
    return {"capital": capital, **result}


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    changes = breadth.load_development_changes(data_dir)
    raw_all = baseline.load_daily(data_dir, end=breadth.PRICE_END).filter(
        pl.col("date") >= breadth.CONTEXT_START
    )
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    all_panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    calendar = raw_all.get_column("date").unique().sort().to_list()

    legacy_targets, _ = breadth.build_quarterly_targets(
        changes, all_panel, calendar
    )
    panel = main_board.filter_main_board(all_panel)
    quarterly_targets, rebalance_dates = breadth.build_quarterly_targets(
        changes, panel, calendar
    )
    liquidation_start = breadth._next_trading_day(
        calendar, breadth.FINAL_EXIT_AVAILABLE_AFTER
    )
    liquidation_index = calendar.index(liquidation_start)
    exit_end_index = min(
        len(calendar) - 1,
        liquidation_index + breadth.MAX_EXIT_TRADING_DAYS - 1,
    )
    first_action = min(rebalance_dates)
    action_dates = calendar[calendar.index(first_action) : exit_end_index + 1]
    candidates = breadth.expand_daily_targets(
        quarterly_targets,
        rebalance_dates,
        action_dates,
        liquidation_start,
    )
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date").is_in(action_dates))
    )
    del panel, all_panel
    gc.collect()
    raw_source = main_board.filter_main_board(
        raw_all.filter(pl.col("date").is_in(action_dates))
    )
    tiers = [
        _summary(
            breadth.simulate(
                candidates,
                raw_source,
                action_dates,
                action_dates,
                data_dir,
                initial_cash=capital,
            ),
            capital,
        )
        for capital in CAPITALS
    ]
    decision = evaluate(tiers, benchmark)
    legacy_unsupported = legacy_targets.filter(
        ~pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN)
    )
    payload = {
        "schema_version": "p0-fund-ownership-breadth-main-board-v1",
        "contract_frozen": "2026-09-02",
        "period": {
            "signal_start": breadth.DEVELOPMENT_START,
            "signal_end": breadth.DEVELOPMENT_END,
            "account_start": first_action,
            "forced_exit_start": liquidation_start,
            "account_end": action_dates[-1],
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "capital_ladder": list(CAPITALS),
            "primary_capital": PRIMARY_CAPITAL,
            "selection": "unchanged_frozen_fund_ownership_breadth",
        },
        "data": {
            "eligible_metadata_events": changes.height,
            "legacy_all_a_target_rows": legacy_targets.height,
            "legacy_unsupported_target_rows": legacy_unsupported.height,
            "legacy_unsupported_target_symbols": legacy_unsupported.get_column(
                "symbol"
            ).n_unique(),
            "main_board_target_rows": quarterly_targets.height,
            "main_board_target_symbols": quarterly_targets.get_column(
                "symbol"
            ).n_unique(),
            "rebalance_dates": len(rebalance_dates),
            "daily_candidate_rows": candidates.height,
        },
        "benchmark": benchmark,
        "capital_tiers": tiers,
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
            "p0_fund_ownership_breadth_main_board_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
