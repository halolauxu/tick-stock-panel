"""Screen the frozen academic factors on the main-board relaxed mandate."""

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

import run_p0_academic_factor_development_screen as academic  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402


def evaluate(result: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    execution = result["execution"]
    integrity = result["integrity"]
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
        "at_least_five_positive_years": metrics["positive_years"] >= 5,
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
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": excess if math.isfinite(excess) else None,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=academic.DEVELOPMENT_END)
    )
    raw_source = raw_all.filter(
        pl.col("date") >= academic.DEVELOPMENT_START
    )
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = academic.attach_price_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= academic.DEVELOPMENT_START)
    )
    monthly, action_dates = academic.monthly_signal_panel(panel)
    monthly = academic.attach_annual_factors(
        monthly, academic.load_annual_factors(data_dir)
    )
    del panel
    gc.collect()
    results: dict[str, Any] = {}
    promoted: list[str] = []
    for factor_id in academic.FACTOR_IDS:
        candidates = academic.build_candidates(monthly, factor_id)
        result = academic.simulate_factor(
            candidates, raw_source, all_dates, action_dates, data_dir
        )
        decision = evaluate(result, benchmark)
        results[factor_id] = {
            "data": {
                "signal_rows": candidates.height,
                "signal_symbols": candidates.get_column("symbol").n_unique(),
                "rebalance_days": candidates.get_column("entry_date").n_unique(),
            },
            "strategy": result,
            "decision": decision,
        }
        if decision["passed"]:
            promoted.append(factor_id)
    payload = {
        "schema_version": "p0-academic-factor-main-board-relaxed-v1",
        "contract_frozen": "2026-09-02",
        "period": {
            "start": academic.DEVELOPMENT_START,
            "end": academic.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "initial_cash": shared.INITIAL_CASH,
            "factor_definitions": "unchanged_academic_factor_screen",
        },
        "benchmark": benchmark,
        "factor_ids": academic.FACTOR_IDS,
        "results": results,
        "promoted_to_validation_contract": promoted,
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
            "/app/data/research/p0_academic_factor_main_board_relaxed_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
