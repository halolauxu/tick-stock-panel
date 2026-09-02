"""Run the frozen market-wide margin-deleveraging micro-cap risk study."""

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

import audit_p0_margin_deleveraging_risk_data as margin_audit  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_microcap_pledge_risk_overlay as pledge  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
INDIVIDUAL_DELEVERAGE_THRESHOLD = -0.05
METRIC_WINDOW_DAYS = 5
HISTORY_DAYS = 252
BREADTH_PERCENTILE = 0.95
BALANCE_PERCENTILE = 0.05
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = baseline.DEVELOPMENT_END
EXPECTED_AUDIT_SHA256 = "aa7950a9069c5342730994f08abf38a7ee10792bed6f75cabde5e7cefeeba1ae"


def build_daily_metrics(margin: pl.DataFrame) -> pl.DataFrame:
    comparable = margin_audit.comparable_margin(margin)
    return (
        comparable.group_by("trade_date")
        .agg(
            (pl.col("balance_change") <= INDIVIDUAL_DELEVERAGE_THRESHOLD)
            .mean()
            .alias("deleverage_breadth"),
            (pl.col("rzye").sum() / pl.col("previous_rzye").sum() - 1.0).alias(
                "aggregate_balance_change"
            ),
            pl.len().alias("comparable_symbols"),
        )
        .sort("trade_date")
        .with_columns(
            pl.col("deleverage_breadth")
            .rolling_mean(window_size=METRIC_WINDOW_DAYS, min_samples=METRIC_WINDOW_DAYS)
            .alias("deleverage_breadth_5d"),
            (
                pl.col("aggregate_balance_change")
                .add(1.0)
                .log()
                .rolling_sum(window_size=METRIC_WINDOW_DAYS, min_samples=METRIC_WINDOW_DAYS)
                .exp()
                - 1.0
            ).alias("aggregate_balance_change_5d"),
        )
        .drop_nulls(["deleverage_breadth_5d", "aggregate_balance_change_5d"])
    )


def build_risk_state(daily_metrics: pl.DataFrame) -> pl.DataFrame:
    breadth_history: list[float] = []
    balance_history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in daily_metrics.sort("trade_date").iter_rows(named=True):
        breadth_threshold = (
            pledge._nearest_rank_percentile(breadth_history[-HISTORY_DAYS:], BREADTH_PERCENTILE)
            if len(breadth_history) >= HISTORY_DAYS
            else None
        )
        balance_threshold = (
            pledge._nearest_rank_percentile(balance_history[-HISTORY_DAYS:], BALANCE_PERCENTILE)
            if len(balance_history) >= HISTORY_DAYS
            else None
        )
        risk_off = bool(
            breadth_threshold is not None
            and balance_threshold is not None
            and row["deleverage_breadth_5d"] >= breadth_threshold
            and row["aggregate_balance_change_5d"] <= balance_threshold
        )
        rows.append(
            {
                **row,
                "breadth_historical_threshold": breadth_threshold,
                "balance_historical_threshold": balance_threshold,
                "risk_off": risk_off,
            }
        )
        breadth_history.append(row["deleverage_breadth_5d"])
        balance_history.append(row["aggregate_balance_change_5d"])
    return pl.DataFrame(rows, infer_schema_length=None)


def attach_weekly_state(weekly_dates: pl.DataFrame, risk_state: pl.DataFrame) -> pl.DataFrame:
    return (
        weekly_dates.sort("date")
        .join_asof(
            risk_state.sort("trade_date"),
            left_on="date",
            right_on="trade_date",
            strategy="backward",
        )
        .with_columns(pl.col("risk_off").fill_null(False))
    )


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily_equity", "rebalance_snapshots", "orders"}
    }


def evaluate(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["accounts"][str(int(PRIMARY_CAPITAL))]["deleveraging_gate"]
    metrics = candidate["metrics"]
    yearly = {row["year"]: row["account_return"] for row in metrics["yearly"]}
    checks = {
        "annualized_at_least_35pct": metrics["account_annualized"] >= 0.35,
        "drawdown_within_30pct": metrics["account_max_drawdown"] >= -0.30,
        "all_7_years_positive": all(yearly.get(year, -1.0) > 0 for year in range(2014, 2021)),
        "buy_execution_at_least_80pct": candidate["execution"]["buy"]["execution_rate"] >= 0.80,
        "sell_execution_at_least_80pct": candidate["execution"]["sell"]["execution_rate"] >= 0.80,
        "no_unresolved_positions": candidate["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": candidate["integrity"]["max_cash_reconciliation_error"] <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = data_dir / "research" / "p0_main_board_microcap_margin_deleveraging_data.json"
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if audit_sha != EXPECTED_AUDIT_SHA256:
        raise ValueError(f"margin audit hash mismatch: {audit_sha}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "DATA_QUALIFIED":
        raise ValueError("margin deleveraging data audit did not qualify")

    margin = margin_audit.load_margin(data_dir)
    risk_state = build_risk_state(build_daily_metrics(margin))
    source = main_board.filter_main_board(baseline.load_daily(data_dir, end=DEVELOPMENT_END))
    all_dates = source["date"].unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    control_all = account.build_signal_candidates(panel)
    weekly_dates = (
        control_all.filter(
            pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
        )
        .select("date", "entry_date")
        .unique()
        .sort("date")
    )
    weekly_state = attach_weekly_state(weekly_dates, risk_state)
    risk_off_entries = weekly_state.filter(pl.col("risk_off")).select("entry_date")
    candidate_all = control_all.join(risk_off_entries, on="entry_date", how="anti")
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select("date", "period", "market_net")
    control = control_all.filter(
        pl.col("entry_date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
    )
    action_dates = control["entry_date"].unique().sort().to_list()
    symbols = control_all["symbol"].unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    ).filter(pl.col("symbol").is_in(symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(control_all, quotes)

    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        accounts[str(int(capital))] = {"initial_cash": capital}
        for name, candidates in {
            "control": control_all,
            "deleveraging_gate": candidate_all,
        }.items():
            result = pledge.run_arm_account(
                "development",
                candidates,
                action_dates,
                execution_grid,
                quotes,
                all_dates,
                weekly_market,
                initial_cash=capital,
            )
            accounts[str(int(capital))][name] = _summary(result)

    development = {
        "stage": "development",
        "period": {"start": DEVELOPMENT_START, "end": DEVELOPMENT_END},
        "trading_days": len(
            [day for day in all_dates if DEVELOPMENT_START <= day <= DEVELOPMENT_END]
        ),
        "funnel": {
            "margin_rows": margin.height,
            "daily_metric_rows": risk_state.height,
            "rebalance_weeks": weekly_state.height,
            "risk_off_weeks": weekly_state.filter(pl.col("risk_off")).height,
            "control_candidate_rows": control.height,
            "candidate_rows": candidate_all.filter(
                pl.col("entry_date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            ).height,
        },
        "weekly_state": weekly_state.to_dicts(),
        "accounts": accounts,
    }
    decision = evaluate(development)
    development["decision"] = decision
    payload = {
        "schema_version": "p0-main-board-microcap-margin-deleveraging-risk-v1",
        "contract_frozen": "2026-09-03",
        "data_audit_sha256": audit_sha,
        "contract": {
            "individual_deleveraging_threshold": INDIVIDUAL_DELEVERAGE_THRESHOLD,
            "metric_window_days": METRIC_WINDOW_DAYS,
            "history_days": HISTORY_DAYS,
            "breadth_percentile": BREADTH_PERCENTILE,
            "balance_percentile": BALANCE_PERCENTILE,
            "capital_ladder": list(CAPITALS),
        },
        "stages": {
            "development": development,
            "validation": {
                "status": (
                    "READY_WITH_EXISTING_2021_2023_DATA"
                    if decision["passed"]
                    else "NOT_READ_AFTER_DEVELOPMENT_FAILURE"
                )
            },
            "known_stress": {"status": "NOT_READ_BEFORE_VALIDATION"},
        },
        "decision": (
            "DEVELOPMENT_PASSED_READY_FOR_VALIDATION"
            if decision["passed"]
            else "TERMINATE_MARGIN_DELEVERAGING_RISK"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    primary = accounts[str(int(PRIMARY_CAPITAL))]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_audit_sha256": audit_sha,
                "development_funnel": development["funnel"],
                "development_gate": decision,
                "primary_account": {
                    name: {
                        key: result[key] for key in ("metrics", "execution", "integrity", "account")
                    }
                    for name, result in primary.items()
                    if name != "initial_cash"
                },
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_main_board_microcap_margin_deleveraging_risk_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
