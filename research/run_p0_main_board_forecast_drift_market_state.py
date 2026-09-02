"""Gate the frozen main-board forecast-drift event with causal market states."""

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

import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_forecast_drift_account as parent  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

WARMUP_START = date(2013, 1, 1)
START = parent.DEVELOPMENT_START
END = parent.DEVELOPMENT_END
WINDOW = 120
VARIANT_IDS = (
    "trend_120_positive",
    "breadth_120_majority",
    "trend_and_breadth",
)


def build_market_states(panel: pl.DataFrame) -> pl.DataFrame:
    dates = panel.select("date").unique().sort("date").with_row_index("_state_index")
    stock_state = (
        panel.join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("_state_index").shift(WINDOW - 1).over("symbol").alias("_prior_index"),
            pl.col("close").rolling_mean(WINDOW, min_samples=WINDOW).over("symbol").alias("ma120"),
        )
        .with_columns(
            (pl.col("_state_index") == pl.col("_prior_index") + WINDOW - 1).alias("continuous_120")
        )
    )
    breadth = (
        stock_state.filter(pl.col("continuous_120") & pl.col("ma120").is_not_null())
        .group_by("date")
        .agg(
            (pl.col("close") > pl.col("ma120")).mean().alias("breadth_120"),
            pl.len().alias("breadth_symbols"),
        )
    )
    market = (
        panel.filter(pl.col("daily_return").is_finite())
        .group_by("date")
        .agg(pl.col("daily_return").mean().alias("market_daily_return"))
        .sort("date")
        .with_columns(
            (pl.col("market_daily_return") + 1.0)
            .log()
            .rolling_sum(WINDOW, min_samples=WINDOW)
            .exp()
            .sub(1.0)
            .alias("market_return_120d")
        )
    )
    return (
        dates.join(market, on="date", how="left")
        .join(breadth, on="date", how="left")
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("entry_date"))
        .drop_nulls(subset=["entry_date", "market_return_120d", "breadth_120"])
        .filter(pl.col("entry_date").is_between(START, END, closed="both"))
        .with_columns(
            (pl.col("market_return_120d") > 0).alias("trend_120_positive"),
            (pl.col("breadth_120") >= 0.50).alias("breadth_120_majority"),
        )
        .with_columns(
            (pl.col("trend_120_positive") & pl.col("breadth_120_majority")).alias(
                "trend_and_breadth"
            )
        )
        .select(
            "entry_date",
            pl.col("date").alias("state_date"),
            "market_return_120d",
            "breadth_120",
            "breadth_symbols",
            *VARIANT_IDS,
        )
    )


def filter_events_by_entry_state(
    events: pl.DataFrame,
    states: pl.DataFrame,
    all_dates: list[date],
    variant: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if variant not in VARIANT_IDS:
        raise ValueError(f"unknown state variant: {variant}")
    calendar = pl.DataFrame({"entry_date": all_dates}).sort("entry_date")
    mapped = (
        events.with_columns((pl.col("ann_date") + pl.duration(days=1)).alias("available_after"))
        .sort("available_after")
        .join_asof(
            calendar,
            left_on="available_after",
            right_on="entry_date",
            strategy="forward",
        )
        .drop_nulls("entry_date")
        .join(states, on="entry_date", how="left")
    )
    eligible = mapped.filter(pl.col(variant).fill_null(False)).select(events.columns)
    return eligible, {
        "input_events": events.height,
        "mapped_events": mapped.height,
        "eligible_events": eligible.height,
        "eligible_event_ratio": eligible.height / max(mapped.height, 1),
        "state_days": states.height,
        "active_state_days": states.filter(pl.col(variant)).height,
    }


def evaluate_variant(
    result: dict[str, Any],
    benchmark: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]
    control_metrics = control["metrics"]
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else None
    )
    checks = {
        "annualized_at_least_20pct": (annualized or -math.inf) >= 0.20,
        "annualized_excess_at_least_10pp": (excess or -math.inf) >= 0.10,
        "max_drawdown_no_worse_than_30pct": (metrics.get("max_drawdown") or -math.inf) >= -0.30,
        "at_least_5_positive_years": metrics["positive_years"] >= 5,
        "annualized_improves_control_by_2pp": (
            (annualized or -math.inf) >= (control_metrics.get("annualized") or -math.inf) + 0.02
        ),
        "drawdown_improves_control_by_5pp": (
            (metrics.get("max_drawdown") or -math.inf)
            >= (control_metrics.get("max_drawdown") or -math.inf) + 0.05
        ),
        "mean_cash_ratio_at_most_75pct": (metrics.get("mean_cash_ratio") or math.inf) <= 0.75,
        "no_unresolved_positions": (result["integrity"]["ending_unresolved_positions"] == 0),
        "cash_reconciled": (result["integrity"]["max_cash_reconciliation_error"] <= 0.01),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "checks": checks,
        "annualized_excess": excess,
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": result["metrics"],
        "execution": result["execution"],
        "integrity": result["integrity"],
        "account": result["account"],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=END).filter(pl.col("date") >= WARMUP_START)
    raw_source = main_board.filter_main_board(raw_all)
    dev_source = raw_source.filter(pl.col("date") >= START)
    all_dates = dev_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit, raw_all
    gc.collect()
    states = build_market_states(panel)
    dev_panel = panel.filter(pl.col("date") >= START)
    benchmark = shared.benchmark_metrics(dev_panel)
    events = parent.load_events(data_dir)
    control_candidates, control_audit = parent.build_candidates(events, dev_panel, all_dates)
    control = parent.simulate(
        control_candidates,
        dev_source,
        all_dates,
        data_dir,
        parent.PRIMARY_CAPITAL,
    )
    variants: dict[str, Any] = {}
    promoted: list[str] = []
    for variant in VARIANT_IDS:
        filtered_events, state_audit = filter_events_by_entry_state(
            events, states, all_dates, variant
        )
        candidates, candidate_audit = parent.build_candidates(filtered_events, dev_panel, all_dates)
        result = parent.simulate(
            candidates,
            dev_source,
            all_dates,
            data_dir,
            parent.PRIMARY_CAPITAL,
        )
        decision = evaluate_variant(result, benchmark, control)
        promoted.extend([variant] if decision["passed"] else [])
        variants[variant] = {
            "state_audit": state_audit,
            "candidate_audit": candidate_audit,
            "account_cny_200k": _summary(result),
            "decision": decision,
        }
    capacity: dict[str, Any] = {}
    for variant in promoted:
        filtered_events, _ = filter_events_by_entry_state(events, states, all_dates, variant)
        candidates, _ = parent.build_candidates(filtered_events, dev_panel, all_dates)
        capacity[variant] = {
            str(int(capital)): _summary(
                parent.simulate(
                    candidates,
                    dev_source,
                    all_dates,
                    data_dir,
                    capital,
                )
            )
            for capital in parent.CAPITALS
            if capital != parent.PRIMARY_CAPITAL
        }
    verdict = "PROMOTE_TO_VALIDATION" if promoted else "TERMINATE_EVENT_FAMILY"
    payload = {
        "schema_version": "p0-main-board-forecast-drift-market-state-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": START,
            "end": END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "window_trading_days": WINDOW,
            "breadth_threshold": 0.50,
            "state_applied_once": "prior_close_before_planned_entry",
            "event_rule": "unchanged_parent_medium_positive_forecast",
        },
        "state_coverage": {
            "days": states.height,
            "first_entry_date": states.get_column("entry_date").min(),
            "last_entry_date": states.get_column("entry_date").max(),
            "minimum_breadth_symbols": states.get_column("breadth_symbols").min(),
        },
        "benchmark": benchmark,
        "control": {
            "candidate_audit": control_audit,
            "account_cny_200k": _summary(control),
        },
        "variants": variants,
        "capacity_if_promoted": capacity,
        "decision": {
            "verdict": verdict,
            "promoted": promoted,
            "validation_read": False,
            "known_stress_read": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "state_coverage": payload["state_coverage"],
                "benchmark": benchmark,
                "control": payload["control"],
                "variants": variants,
                "capacity_if_promoted": capacity,
                "decision": payload["decision"],
                "output": str(output),
                "sha256": digest,
            },
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
        default=Path("/app/data/research/p0_main_board_forecast_drift_market_state.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
