"""Run main-board low-volatility trend behind fundamental breadth state."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_academic_factor_development_screen as academic  # noqa: E402
import run_p0_fundamental_acceleration_drift_discovery as fundamental  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

FUNDAMENTAL_WARMUP_START = date(2011, 1, 1)
BREADTH_WINDOW_DAYS = 120
MIN_REPORTS_PER_WINDOW = 50
MIN_ACTIVE_REBALANCE_RATIO = 0.35


def prior_year(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


def build_fundamental_breadth(
    comparisons: pl.DataFrame, signal_dates: list[date]
) -> pl.DataFrame:
    labeled = (
        comparisons
        if "is_acceleration" in comparisons.columns
        else comparisons.with_columns(
            fundamental.candidate_expression().alias("is_acceleration")
        )
    )
    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        previous_date = prior_year(signal_date)

        def measure(end: date) -> tuple[int, float | None]:
            scoped = labeled.filter(
                pl.col("announce_date").is_between(
                    end - timedelta(days=BREADTH_WINDOW_DAYS - 1),
                    end,
                    closed="both",
                )
            )
            count = scoped.height
            breadth = (
                float(scoped.get_column("is_acceleration").sum()) / count
                if count
                else None
            )
            return count, breadth

        current_count, current_breadth = measure(signal_date)
        prior_count, prior_breadth = measure(previous_date)
        active = bool(
            current_count >= MIN_REPORTS_PER_WINDOW
            and prior_count >= MIN_REPORTS_PER_WINDOW
            and current_breadth is not None
            and prior_breadth is not None
            and current_breadth > prior_breadth
        )
        rows.append(
            {
                "date": signal_date,
                "current_report_count": current_count,
                "prior_year_report_count": prior_count,
                "fundamental_acceleration_breadth": current_breadth,
                "prior_year_acceleration_breadth": prior_breadth,
                "fundamental_breadth_active": active,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("date")


def evaluate(
    result: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
    active_ratio: float,
) -> dict[str, Any]:
    metrics = result["metrics"]
    control_metrics = control["metrics"]
    execution = result["execution"]
    integrity = result["integrity"]
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    annualized_excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else -math.inf
    )
    control_annualized = control_metrics.get("annualized")
    control_drawdown = control_metrics.get("max_drawdown")
    drawdown = metrics.get("max_drawdown")
    checks = {
        "annualized_at_least_15pct": (annualized or -math.inf) >= 0.15,
        "excess_at_least_5pp": annualized_excess >= 0.05,
        "max_drawdown_within_25pct": (drawdown or -math.inf) >= -0.25,
        "at_least_five_positive_years": metrics["positive_years"] >= 5,
        "active_rebalance_ratio_at_least_35pct": active_ratio >= 0.35,
        "buy_execution_at_least_90pct": execution["buy"]["execution_rate"] >= 0.90,
        "sell_execution_at_least_90pct": execution["sell"]["execution_rate"] >= 0.90,
        "no_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "cash_reconciled": integrity["max_cash_reconciliation_error"] <= 0.01,
        "drawdown_improves_control_by_5pp": (
            drawdown is not None
            and control_drawdown is not None
            and drawdown - control_drawdown >= 0.05
        ),
        "annualized_loss_vs_control_at_most_2pp": (
            annualized is not None
            and control_annualized is not None
            and annualized - control_annualized >= -0.02
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": annualized_excess
        if math.isfinite(annualized_excess)
        else None,
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
    raw_source = raw_all.filter(pl.col("date") >= academic.DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = academic.attach_price_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= academic.DEVELOPMENT_START)
    )
    monthly, action_dates = academic.monthly_signal_panel(panel)
    comparisons = fundamental.build_report_comparisons(
        fundamental.load_metrics(data_dir),
        start=FUNDAMENTAL_WARMUP_START,
        end=academic.DEVELOPMENT_END,
    )
    signal_dates = monthly.get_column("date").unique().sort().to_list()
    breadth = build_fundamental_breadth(comparisons, signal_dates)
    active_monthly = monthly.join(breadth, on="date", how="left").filter(
        pl.col("fundamental_breadth_active").fill_null(False)
    )
    candidates = academic.build_candidates(active_monthly, "low_volatility_trend")
    control_candidates = academic.build_candidates(monthly, "low_volatility_trend")
    del panel, monthly, active_monthly
    gc.collect()
    result = academic.simulate_factor(
        candidates, raw_source, all_dates, action_dates, data_dir
    )
    control = academic.simulate_factor(
        control_candidates, raw_source, all_dates, action_dates, data_dir
    )
    active_rebalances = candidates.get_column("entry_date").n_unique()
    active_ratio = active_rebalances / len(action_dates)
    decision = evaluate(result, control, benchmark, active_ratio)
    payload = {
        "schema_version": "p0-low-volatility-fundamental-breadth-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": academic.DEVELOPMENT_START,
            "end": academic.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "selection": "unchanged_low_volatility_trend",
            "breadth_window_calendar_days": BREADTH_WINDOW_DAYS,
            "state": "current_acceleration_breadth_gt_same_date_prior_year",
            "minimum_reports_per_window": MIN_REPORTS_PER_WINDOW,
        },
        "data": {
            "planned_rebalances": len(action_dates),
            "active_rebalances": active_rebalances,
            "active_rebalance_ratio": active_ratio,
            "signal_rows": candidates.height,
            "signal_symbols": candidates.get_column("symbol").n_unique(),
            "breadth": breadth.to_dicts(),
        },
        "benchmark": benchmark,
        "control": control,
        "strategy": result,
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
            "/app/data/research/p0_low_volatility_fundamental_breadth_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
