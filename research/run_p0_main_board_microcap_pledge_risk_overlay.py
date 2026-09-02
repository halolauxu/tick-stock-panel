"""Run the frozen main-board micro-cap equity-pledge risk overlay."""

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

import audit_p0_pledge_risk_data as pledge_audit  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
MATERIAL_PLEDGE_RATIO = 5.0
STOCK_EXCLUSION_DAYS = 126
SYSTEM_LOOKBACK_DAYS = 20
SYSTEM_HISTORY_WEEKS = 52
SYSTEM_PERCENTILE = 0.90
SYSTEM_MINIMUM_EVENTS = 10
STAGES = {
    "development": (date(2014, 1, 1), baseline.DEVELOPMENT_END),
    "validation": (date(2021, 1, 1), baseline.VALIDATION_END),
    "known_stress": (date(2024, 1, 1), date(2026, 8, 28)),
}


def _next_trading_day(day: date, trading_dates: list[date]) -> date | None:
    index = bisect.bisect_right(trading_dates, day)
    return trading_dates[index] if index < len(trading_dates) else None


def material_events(data_dir: Path, end: date) -> pl.DataFrame:
    events = pledge_audit.load_events(data_dir).filter(pl.col("ann_date") <= end)
    universe = pl.read_parquet(
        data_dir / "research" / "historical_stock_universe.parquet"
    ).filter(pl.col("market") == "主板")
    return (
        events.join(
            universe.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("ann_date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("ann_date") <= pl.col("delist_date"))
            )
            & ~pl.col("is_release").is_in(["1", "Y", "是"])
            & (pl.col("pledge_ratio") >= MATERIAL_PLEDGE_RATIO)
        )
        .select("symbol", "ann_date", "pledge_ratio")
        .unique()
        .sort(["ann_date", "symbol"])
    )


def attach_available_dates(
    events: pl.DataFrame, trading_dates: list[date]
) -> pl.DataFrame:
    rows = []
    for row in events.iter_rows(named=True):
        available = _next_trading_day(row["ann_date"], trading_dates)
        if available is not None:
            rows.append({**row, "available_date": available})
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "ann_date": pl.Date,
            "pledge_ratio": pl.Float64,
            "available_date": pl.Date,
        }
    )


def exclusion_calendar(
    available_events: pl.DataFrame, trading_dates: list[date]
) -> pl.DataFrame:
    date_index = {day: index for index, day in enumerate(trading_dates)}
    rows: list[dict[str, Any]] = []
    for event in available_events.iter_rows(named=True):
        start = date_index[event["available_date"]]
        for risk_date in trading_dates[start : start + STOCK_EXCLUSION_DAYS]:
            rows.append({"symbol": event["symbol"], "date": risk_date})
    if not rows:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date})
    return pl.DataFrame(rows).unique().sort(["date", "symbol"])


def _nearest_rank_percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def systemic_risk_clock(
    weekly_dates: pl.DataFrame,
    available_events: pl.DataFrame,
    trading_dates: list[date],
) -> pl.DataFrame:
    date_index = {day: index for index, day in enumerate(trading_dates)}
    events_by_index: dict[int, set[str]] = {}
    for row in available_events.iter_rows(named=True):
        index = date_index[row["available_date"]]
        events_by_index.setdefault(index, set()).add(row["symbol"])

    history: list[int] = []
    rows: list[dict[str, Any]] = []
    for weekly in weekly_dates.sort("date").iter_rows(named=True):
        index = date_index[weekly["date"]]
        symbols: set[str] = set()
        for event_index in range(max(0, index - SYSTEM_LOOKBACK_DAYS + 1), index + 1):
            symbols.update(events_by_index.get(event_index, set()))
        count = len(symbols)
        threshold = (
            _nearest_rank_percentile(history[-SYSTEM_HISTORY_WEEKS:], SYSTEM_PERCENTILE)
            if len(history) >= SYSTEM_HISTORY_WEEKS
            else None
        )
        risk_off = threshold is not None and count >= max(
            SYSTEM_MINIMUM_EVENTS, threshold
        )
        rows.append(
            {
                **weekly,
                "material_pledge_symbols_20d": count,
                "historical_threshold": threshold,
                "risk_off": risk_off,
            }
        )
        history.append(count)
    return pl.DataFrame(rows)


def build_arms(
    control: pl.DataFrame,
    available_events: pl.DataFrame,
    trading_dates: list[date],
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    exclusions = exclusion_calendar(available_events, trading_dates)
    stock_only = control.join(exclusions, on=["symbol", "date"], how="anti")
    weekly_dates = control.select("date", "entry_date").unique().sort("date")
    risk_clock = systemic_risk_clock(weekly_dates, available_events, trading_dates)
    risk_off_entries = risk_clock.filter(pl.col("risk_off")).select("entry_date")
    systemic_only = control.join(risk_off_entries, on="entry_date", how="anti")
    combined = stock_only.join(risk_off_entries, on="entry_date", how="anti")
    return (
        {
            "control": control,
            "stock_exclusion": stock_only,
            "systemic_gate": systemic_only,
            "combined": combined,
        },
        exclusions,
        risk_clock,
    )


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily_equity", "rebalance_snapshots", "orders"}
    }


def run_arm_account(
    stage: str,
    candidates: pl.DataFrame,
    action_dates: list[date],
    execution_grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
    weekly_market: pl.DataFrame,
    *,
    initial_cash: float,
) -> dict[str, Any]:
    start, end = STAGES[stage]
    scoped_dates = [day for day in all_dates if start <= day <= end]
    scoped_candidates = candidates.filter(
        pl.col("entry_date").is_between(start, end, closed="both")
    )
    scoped_grid = execution_grid.filter(
        pl.col("entry_date").is_between(start, end, closed="both")
    )
    scoped_actions = [day for day in action_dates if start <= day <= end]
    simulation = account.simulate_account(
        scoped_candidates,
        scoped_grid,
        initial_cash=initial_cash,
        action_dates=scoped_actions,
    )
    daily, stale = account.build_daily_equity(
        simulation,
        quotes,
        scoped_dates,
        initial_cash=initial_cash,
    )
    metric = next(
        row
        for row in account.account_period_metrics(daily, weekly_market)
        if row["period"] == stage
    )
    return {
        "period": stage,
        "first_date": scoped_dates[0],
        "last_date": scoped_dates[-1],
        "metrics": metric,
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
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
        "rebalance_snapshots": simulation["snapshots"],
        "orders": simulation["orders"],
        "worst_weeks": account.worst_weeks(daily),
    }


def _common_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "buy_execution_at_least_80pct": candidate["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.80,
        "sell_execution_at_least_80pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.80,
        "no_unresolved_positions": candidate["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciled": candidate["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 0.01,
    }


def evaluate(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    primary = result["accounts"][str(int(PRIMARY_CAPITAL))]["combined"]
    metrics = primary["metrics"]
    yearly = {row["year"]: row["account_return"] for row in metrics["yearly"]}
    if stage == "development":
        checks = {
            "annualized_at_least_35pct": metrics["account_annualized"] >= 0.35,
            "drawdown_within_30pct": metrics["account_max_drawdown"] >= -0.30,
            "all_7_years_positive": all(yearly.get(year, -1.0) > 0 for year in range(2014, 2021)),
            **_common_checks(primary),
        }
    elif stage == "validation":
        checks = {
            "annualized_at_least_30pct": metrics["account_annualized"] >= 0.30,
            "drawdown_within_25pct": metrics["account_max_drawdown"] >= -0.25,
            "all_3_years_positive": all(yearly.get(year, -1.0) > 0 for year in range(2021, 2024)),
            **_common_checks(primary),
        }
    else:
        checks = {
            "2024_2026_all_positive": all(yearly.get(year, -1.0) > 0 for year in range(2024, 2027)),
            "2026_return_above_30pct": yearly.get(2026, -1.0) > 0.30,
            **_common_checks(primary),
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run_stage(
    data_dir: Path,
    events: pl.DataFrame,
    stage: str,
) -> dict[str, Any]:
    start, end = STAGES[stage]
    source = main_board.filter_main_board(baseline.load_daily(data_dir, end=end))
    all_dates = source["date"].unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    control_all = account.build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    available = attach_available_dates(events.filter(pl.col("ann_date") <= end), all_dates)
    arms_all, exclusions, risk_clock = build_arms(control_all, available, all_dates)
    control = arms_all["control"].filter(
        pl.col("entry_date").is_between(start, end, closed="both")
    )
    action_dates = control["entry_date"].unique().sort().to_list()
    symbols = control_all["symbol"].unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=end)
    ).filter(pl.col("symbol").is_in(symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(control_all, quotes)

    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        accounts[str(int(capital))] = {"initial_cash": capital}
        for name, candidates in arms_all.items():
            result = run_arm_account(
                stage,
                candidates,
                action_dates,
                execution_grid,
                quotes,
                all_dates,
                weekly_market,
                initial_cash=capital,
            )
            accounts[str(int(capital))][name] = _summary(result)
    scoped_clock = risk_clock.filter(
        pl.col("entry_date").is_between(start, end, closed="both")
    )
    return {
        "stage": stage,
        "period": {"start": start, "end": end},
        "trading_days": len([day for day in all_dates if start <= day <= end]),
        "funnel": {
            "material_events_available": available.height,
            "exclusion_symbol_dates": exclusions.height,
            "control_candidate_rows": control.height,
            "stock_exclusion_candidate_rows": arms_all["stock_exclusion"].filter(
                pl.col("entry_date").is_between(start, end, closed="both")
            ).height,
            "systemic_risk_off_weeks": scoped_clock.filter(pl.col("risk_off")).height,
            "rebalance_weeks": scoped_clock.height,
        },
        "risk_clock": scoped_clock.to_dicts(),
        "accounts": accounts,
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = (
        data_dir / "research" / "p0_main_board_microcap_pledge_risk_data.json"
    )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "DATA_QUALIFIED":
        raise ValueError("pledge risk data audit did not qualify")
    events = material_events(data_dir, STAGES["known_stress"][1])
    stages: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for stage in STAGES:
        if stage == "validation" and not decisions["development"]["passed"]:
            stages[stage] = {"status": "NOT_READ_AFTER_DEVELOPMENT_FAILURE"}
            continue
        if stage == "known_stress" and not decisions.get("validation", {}).get(
            "passed", False
        ):
            stages[stage] = {"status": "NOT_READ_AFTER_VALIDATION_FAILURE"}
            continue
        stage_result = run_stage(data_dir, events, stage)
        decision = evaluate(stage, stage_result)
        stage_result["decision"] = decision
        stages[stage] = stage_result
        decisions[stage] = decision
    passed = all(decisions.get(stage, {}).get("passed", False) for stage in STAGES)
    payload = {
        "schema_version": "p0-main-board-microcap-pledge-risk-overlay-v1",
        "contract_frozen": "2026-09-03",
        "data_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "material_pledge_ratio_pct": MATERIAL_PLEDGE_RATIO,
            "stock_exclusion_trading_days": STOCK_EXCLUSION_DAYS,
            "system_lookback_trading_days": SYSTEM_LOOKBACK_DAYS,
            "system_history_weeks": SYSTEM_HISTORY_WEEKS,
            "system_percentile": SYSTEM_PERCENTILE,
            "system_minimum_events": SYSTEM_MINIMUM_EVENTS,
            "primary_arm": "combined",
            "capital_ladder": list(CAPITALS),
        },
        "stages": stages,
        "decision": "FORWARD_ELIGIBLE" if passed else "TERMINATE_PLEDGE_RISK_OVERLAY",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    development = stages["development"]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_audit_sha256": payload["data_audit_sha256"],
                "development_funnel": development["funnel"],
                "development_gate": development["decision"],
                "primary_account": development["accounts"][
                    str(int(PRIMARY_CAPITAL))
                ],
                "stage_status": {
                    stage: result.get("status", "READ")
                    for stage, result in stages.items()
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
        default=Path(
            "/app/data/research/p0_main_board_microcap_pledge_risk_overlay_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
