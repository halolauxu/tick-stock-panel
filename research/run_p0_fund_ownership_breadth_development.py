"""Run the frozen public-fund ownership-breadth development account."""

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

import audit_fund_ownership_breadth_data as metadata  # noqa: E402
import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CONTEXT_START = date(2018, 1, 1)
DEVELOPMENT_START = date(2018, 3, 31)
DEVELOPMENT_END = date(2020, 9, 30)
FINAL_EXIT_AVAILABLE_AFTER = date(2021, 4, 30)
PRICE_END = date(2021, 6, 15)
INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 10
MIN_COVERAGE_SHARE_GROWTH = 1.0
MIN_FUND_COUNT_INCREASE = 10
MIN_AVERAGE_HOLDING_CNY = 1_000_000.0
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MAX_EXIT_TRADING_DAYS = 20


def _next_trading_day(calendar: list[date], after: date) -> date:
    for current in calendar:
        if current > after:
            return current
    raise ValueError(f"no trading day after {after}")


def load_development_changes(data_dir: Path) -> pl.DataFrame:
    events, _ = metadata.load_events(data_dir)
    quality = metadata.quarter_quality(events)
    return metadata.same_depth_changes(events, quality).filter(
        pl.col("period_end").is_between(DEVELOPMENT_START, DEVELOPMENT_END)
        & pl.col("period_end").dt.month().is_in([3, 9])
        & (pl.col("previous_coverage_share") > 0)
        & (pl.col("coverage_share_growth") >= MIN_COVERAGE_SHARE_GROWTH)
        & (pl.col("fund_count_increase") >= MIN_FUND_COUNT_INCREASE)
        & (
            pl.col("average_market_value_per_fund_cny")
            >= MIN_AVERAGE_HOLDING_CNY
        )
    )


def build_quarterly_targets(
    changes: pl.DataFrame,
    panel: pl.DataFrame,
    calendar: list[date],
) -> tuple[pl.DataFrame, list[date]]:
    available_dates = changes.get_column("available_after").unique().sort().to_list()
    rebalance_map = pl.DataFrame(
        {
            "available_after": available_dates,
            "rebalance_date": [
                _next_trading_day(calendar, current) for current in available_dates
            ],
        }
    )
    snapshots = (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d")
        )
        .select(
            "symbol",
            "date",
            "raw_close",
            "market_cap",
            "amount",
            "mean_amount_20d",
        )
        .sort(["symbol", "date"])
    )
    targets = (
        changes.sort(["symbol", "available_after"])
        .join_asof(
            snapshots,
            left_on="available_after",
            right_on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .join(rebalance_map, on="available_after", how="left")
        .filter(
            (pl.col("market_cap") >= MIN_MARKET_CAP)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .sort(
            [
                "available_after",
                "coverage_share_growth",
                "fund_count_increase",
                "symbol",
            ],
            descending=[False, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over("available_after")
            .alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            pl.col("available_after").alias("date"),
            "rebalance_date",
            "symbol",
            "coverage_share_growth",
            "fund_count_increase",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "mean_amount_20d",
            "cap_rank",
        )
        .sort(["rebalance_date", "cap_rank", "symbol"])
    )
    return targets, rebalance_map.get_column("rebalance_date").to_list()


def expand_daily_targets(
    quarterly_targets: pl.DataFrame,
    rebalance_dates: list[date],
    action_dates: list[date],
    liquidation_start: date,
) -> pl.DataFrame:
    schedule = pl.DataFrame(
        {"entry_date": [day for day in action_dates if day < liquidation_start]}
    ).sort("entry_date")
    rebalances = pl.DataFrame({"rebalance_date": rebalance_dates}).sort(
        "rebalance_date"
    )
    membership = schedule.join_asof(
        rebalances,
        left_on="entry_date",
        right_on="rebalance_date",
        strategy="backward",
    ).drop_nulls("rebalance_date")
    return (
        quarterly_targets.join(membership, on="rebalance_date", how="inner")
        .select(
            "date",
            "entry_date",
            "symbol",
            "coverage_share_growth",
            "fund_count_increase",
            "market_cap",
            "signal_amount",
            "mean_amount_20d",
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def simulate(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    action_dates: list[date],
    data_dir: Path,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in (2018, 2019, 2020):
        values = (
            account_daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
            "trading_days": account_daily.height,
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, account_daily),
        "worst_weeks": account.worst_weeks(account_daily),
    }


def evaluate_gate(result: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else -math.inf
    )
    checks = {
        "annualized_at_least_50pct": (annualized or -math.inf) >= 0.50,
        "annualized_excess_at_least_20pp": excess >= 0.20,
        "max_drawdown_no_worse_than_30pct": (
            metrics.get("max_drawdown") or -math.inf
        )
        >= -0.30,
        "at_least_two_positive_signal_years": metrics.get("positive_years", 0)
        >= 2,
        "mean_cash_ratio_at_most_25pct": (
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.25,
        "buy_execution_at_least_90pct": result["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_execution_at_least_90pct": result["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "ending_positions_resolved": result["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciled": result["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "annualized_excess": excess if math.isfinite(excess) else None,
        "validation_read": False,
        "known_stress_read": False,
        "counts_toward_50pct_goal": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    changes = load_development_changes(data_dir)
    raw_all = baseline.load_daily(data_dir, end=PRICE_END).filter(
        pl.col("date") >= CONTEXT_START
    )
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    calendar = raw_all.get_column("date").unique().sort().to_list()
    quarterly_targets, rebalance_dates = build_quarterly_targets(
        changes, panel, calendar
    )
    liquidation_start = _next_trading_day(calendar, FINAL_EXIT_AVAILABLE_AFTER)
    liquidation_index = calendar.index(liquidation_start)
    exit_end_index = min(
        len(calendar) - 1, liquidation_index + MAX_EXIT_TRADING_DAYS - 1
    )
    first_action = min(rebalance_dates)
    action_dates = calendar[calendar.index(first_action) : exit_end_index + 1]
    candidates = expand_daily_targets(
        quarterly_targets,
        rebalance_dates,
        action_dates,
        liquidation_start,
    )
    benchmark = shared.benchmark_metrics(panel.filter(pl.col("date").is_in(action_dates)))
    del panel
    gc.collect()
    raw_source = raw_all.filter(pl.col("date").is_in(action_dates))
    result = simulate(candidates, raw_source, action_dates, action_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-fund-ownership-breadth-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "signal_start": DEVELOPMENT_START,
            "signal_end": DEVELOPMENT_END,
            "account_start": first_action,
            "forced_exit_start": liquidation_start,
            "account_end": action_dates[-1],
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "quarters": "Q1 and Q3 only",
            "same_depth_lag_quarters": 2,
            "minimum_coverage_share_growth": MIN_COVERAGE_SHARE_GROWTH,
            "minimum_fund_count_increase": MIN_FUND_COUNT_INCREASE,
            "minimum_average_holding_cny": MIN_AVERAGE_HOLDING_CNY,
            "target_positions": TARGET_POSITIONS,
            "initial_cash": INITIAL_CASH,
            "maximum_exit_trading_days": MAX_EXIT_TRADING_DAYS,
        },
        "data": {
            "eligible_metadata_events": changes.height,
            "quarterly_target_rows": quarterly_targets.height,
            "target_symbols": quarterly_targets.get_column("symbol").n_unique(),
            "rebalance_dates": len(rebalance_dates),
            "daily_candidate_rows": candidates.height,
        },
        "benchmark": benchmark,
        "account": result,
        "decision": decision,
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
        default=Path("/app/data/research/p0_fund_ownership_breadth_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
