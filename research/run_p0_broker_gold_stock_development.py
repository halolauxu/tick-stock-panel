"""Run the frozen broker gold-stock consensus development account."""

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

import audit_broker_gold_stock_data as metadata  # noqa: E402
import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CONTEXT_START = date(2020, 5, 1)
DEVELOPMENT_SIGNAL_START = date(2020, 7, 1)
DEVELOPMENT_SIGNAL_END = date(2022, 12, 1)
FINAL_LIST_AVAILABLE_AFTER = date(2023, 1, 3)
PRICE_END = date(2023, 2, 15)
INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 10
MINIMUM_BROKERS = 3
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MAX_EXIT_TRADING_DAYS = 20


def load_development_consensus(data_dir: Path) -> pl.DataFrame:
    events, _ = metadata.load_events(data_dir)
    return metadata.build_consensus(events).filter(
        pl.col("recommendation_month").is_between(
            DEVELOPMENT_SIGNAL_START, DEVELOPMENT_SIGNAL_END, closed="both"
        )
    )


def _next_trading_day(calendar: list[date], after: date) -> date:
    for current in calendar:
        if current > after:
            return current
    raise ValueError(f"no trading day after {after}")


def build_monthly_targets(
    consensus: pl.DataFrame,
    panel: pl.DataFrame,
    calendar: list[date],
) -> tuple[pl.DataFrame, list[date]]:
    available_dates = (
        consensus.get_column("available_after").unique().sort().to_list()
    )
    rebalance_map = pl.DataFrame(
        {
            "available_after": available_dates,
            "rebalance_date": [
                _next_trading_day(calendar, current) for current in available_dates
            ],
        }
    )
    featured = panel.sort(["symbol", "date"]).with_columns(
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d")
    )
    snapshots = featured.select(
        "symbol",
        "date",
        "raw_close",
        "market_cap",
        "amount",
        "mean_amount_20d",
    ).sort(["symbol", "date"])
    targets = (
        consensus.filter(pl.col("broker_count") >= MINIMUM_BROKERS)
        .sort(["symbol", "available_after"])
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
            ["available_after", "broker_count", "symbol"],
            descending=[False, True, False],
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
            "broker_count",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "mean_amount_20d",
            "cap_rank",
        )
        .sort(["rebalance_date", "cap_rank", "symbol"])
    )
    return targets, rebalance_map.get_column("rebalance_date").to_list()


def expand_daily_targets(
    monthly_targets: pl.DataFrame,
    rebalance_dates: list[date],
    action_dates: list[date],
    liquidation_start: date,
) -> pl.DataFrame:
    active_actions = [current for current in action_dates if current < liquidation_start]
    schedule = pl.DataFrame({"entry_date": active_actions}).sort("entry_date")
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
        monthly_targets.join(membership, on="rebalance_date", how="inner")
        .select(
            "date",
            "entry_date",
            "symbol",
            "broker_count",
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
    for year in (2020, 2021, 2022):
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
        "orders": simulation["orders"],
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
    consensus = load_development_consensus(data_dir)
    raw_all = baseline.load_daily(data_dir, end=PRICE_END).filter(
        pl.col("date") >= CONTEXT_START
    )
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    calendar = raw_all.get_column("date").unique().sort().to_list()
    monthly_targets, rebalance_dates = build_monthly_targets(
        consensus, panel, calendar
    )
    liquidation_start = _next_trading_day(calendar, FINAL_LIST_AVAILABLE_AFTER)
    liquidation_index = calendar.index(liquidation_start)
    exit_end_index = min(
        len(calendar) - 1, liquidation_index + MAX_EXIT_TRADING_DAYS - 1
    )
    first_action = min(rebalance_dates)
    action_dates = calendar[calendar.index(first_action) : exit_end_index + 1]
    candidates = expand_daily_targets(
        monthly_targets,
        rebalance_dates,
        action_dates,
        liquidation_start,
    )
    all_dates = action_dates
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date").is_in(all_dates))
    )
    del panel
    gc.collect()
    raw_source = raw_all.filter(pl.col("date").is_in(all_dates))
    result = simulate(candidates, raw_source, all_dates, action_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-broker-gold-stock-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "signal_start": DEVELOPMENT_SIGNAL_START,
            "signal_end": DEVELOPMENT_SIGNAL_END,
            "account_start": first_action,
            "forced_exit_start": liquidation_start,
            "account_end": action_dates[-1],
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "minimum_distinct_brokers": MINIMUM_BROKERS,
            "availability": "calendar day 3 close; next trading open",
            "target_positions": TARGET_POSITIONS,
            "initial_cash": INITIAL_CASH,
            "minimum_market_cap": MIN_MARKET_CAP,
            "minimum_mean_amount_20d": MIN_MEAN_AMOUNT_20D,
            "maximum_exit_trading_days": MAX_EXIT_TRADING_DAYS,
        },
        "data": {
            "consensus_events": consensus.height,
            "monthly_target_rows": monthly_targets.height,
            "target_symbols": monthly_targets.get_column("symbol").n_unique(),
            "rebalance_months": len(rebalance_dates),
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
        default=Path("/app/data/research/p0_broker_gold_stock_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
