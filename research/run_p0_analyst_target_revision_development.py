"""Run the frozen analyst target-price revision development event study."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH))

import audit_analyst_report_metadata as metadata  # noqa: E402
from research.run_p0_forecast_drift_development import (  # noqa: E402
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_large_order_flow_development import (  # noqa: E402
    MAX_EXIT_DELAY,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

CONTEXT_START = date(2016, 1, 1)
DEVELOPMENT_START = date(2017, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
CATEGORY = "analyst_target_up_breadth_2"
MIN_TRADABLE_EVENTS = 80
MIN_SIGNAL_DAYS = 70


def build_events(reports: pl.DataFrame) -> pl.DataFrame:
    revisions = metadata.prepare_revisions(reports)
    signals = metadata.breadth_signals(
        revisions, "target_up_10pct", minimum_brokers=2
    )
    if not signals:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "ann_date": pl.Date,
                "broker_count": pl.Int64,
                "category": pl.String,
            }
        )
    return (
        pl.DataFrame(signals, infer_schema_length=None)
        .rename({"signal_date": "ann_date"})
        .with_columns(pl.lit(CATEGORY).alias("category"))
        .select("symbol", "ann_date", "broker_count", "category")
        .sort(["ann_date", "symbol"])
    )


def promotion_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["tradable_events"] >= MIN_TRADABLE_EVENTS
        and metrics["announcement_days"] >= MIN_SIGNAL_DAYS
        and metrics["tradable_rate"] >= 0.90
        and metrics["benchmark_coverage"] >= 0.99
        and metrics["entry_capacity_feasible_rate"] >= 0.95
        and metrics["unresolved_exits"] == 0
        and (metrics["mean_net_return"] or -math.inf) >= 0.04
        and (metrics["mean_excess_return"] or -math.inf) >= 0.03
        and (metrics["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and metrics["positive_excess_years"] >= 3
        and (metrics["max_year_positive_excess_share"] or math.inf) <= 0.50
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    reports = metadata.load_reports(data_dir)
    events = build_events(reports)
    panel = prepare_panel(load_panel(data_dir, CONTEXT_START, PANEL_END))
    trades = build_trades(
        events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    metrics = summarize_category(
        trades,
        CATEGORY,
        positive_categories=(CATEGORY,),
        min_tradable_events=MIN_TRADABLE_EVENTS,
        min_announcement_days=MIN_SIGNAL_DAYS,
    )
    metrics["promotion_passed"] = promotion_passed(metrics)
    passed = metrics["promotion_passed"]
    payload = {
        "schema_version": "p0-analyst-target-revision-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "same_broker_comparison_days": metadata.COMPARISON_DAYS,
            "target_revision_minimum": metadata.TARGET_REVISION_MINIMUM,
            "breadth_calendar_days": metadata.BREADTH_DAYS,
            "minimum_brokers": 2,
            "cooldown_calendar_days": metadata.COOLDOWN_DAYS,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation": DAILY_PARTICIPATION,
        },
        "data": {
            "reports": reports.height,
            "events": events.height,
            "event_symbols": events["symbol"].n_unique() if events.height else 0,
            "raw_signal_days": events["ann_date"].n_unique() if events.height else 0,
            "panel_rows": panel.height,
            "panel_symbols": panel["symbol"].n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "metrics": metrics,
        "decision": {
            "development_passed": passed,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_rule_before_independent_validation"
                if passed
                else "terminate_analyst_target_revision"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
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
            "/app/data/research/p0_analyst_target_revision_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
