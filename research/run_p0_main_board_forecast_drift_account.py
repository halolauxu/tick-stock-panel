"""Run the frozen development account for main-board forecast drift."""
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

import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_forecast_drift_development as forecast  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
TARGET_POSITIONS = 10
SIGNAL_LIFETIME_TRADING_DAYS = 10


def load_events(data_dir: Path) -> pl.DataFrame:
    return (
        forecast.categorize_events(forecast.load_forecasts(data_dir))
        .filter(
            pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN)
            & pl.col("category").is_in(["growth_0_50", "growth_50_100"])
            & (pl.col("net_profit_min") > 0)
            & pl.col("p_change_min").is_between(0, 100, closed="left")
        )
        .sort(["ann_date", "symbol"])
    )


def build_candidates(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    all_dates: list[date],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    calendar = pl.DataFrame({"entry_date": all_dates}).with_row_index(
        "action_index"
    )
    last_entry_index = len(all_dates) - SIGNAL_LIFETIME_TRADING_DAYS - 1
    signal_quotes = (
        events.sort(["symbol", "ann_date"])
        .join_asof(
            panel.select(
                "symbol",
                "date",
                "raw_close",
                "amount",
                "market_cap",
            ).sort(["symbol", "date"]),
            left_on="ann_date",
            right_on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .rename({"date": "signal_quote_date"})
        .with_columns(
            (pl.col("ann_date") + pl.duration(days=1)).alias("available_after")
        )
        .sort("available_after")
        .join_asof(
            calendar.sort("entry_date"),
            left_on="available_after",
            right_on="entry_date",
            strategy="forward",
        )
        .drop_nulls("entry_date")
        .filter(
            (pl.col("action_index") <= last_entry_index)
            & (pl.col("amount") >= 50_000_000.0)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
    )
    expanded = (
        signal_quotes.with_columns(
            pl.int_ranges(
                pl.col("action_index"),
                pl.col("action_index") + SIGNAL_LIFETIME_TRADING_DAYS,
            ).alias("_active_indices")
        )
        .explode("_active_indices")
        .drop("entry_date", "action_index")
        .join(
            calendar.rename({"action_index": "_active_indices"}),
            on="_active_indices",
            how="inner",
        )
        .sort(
            [
                "entry_date",
                "symbol",
                "ann_date",
                "p_change_min",
                "p_change_max",
            ],
            descending=[False, False, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["entry_date", "symbol"], keep="first")
        .sort(
            ["entry_date", "p_change_min", "p_change_max", "symbol"],
            descending=[False, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            pl.col("ann_date").alias("date"),
            "entry_date",
            "symbol",
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )
    audit = {
        "eligible_unique_events": signal_quotes.height,
        "eligible_event_symbols": signal_quotes.get_column("symbol").n_unique(),
        "daily_candidate_rows": expanded.height,
        "active_account_days": expanded.get_column("entry_date").n_unique(),
        "candidate_symbols": expanded.get_column("symbol").n_unique(),
        "last_accepted_entry_index": last_entry_index,
    }
    return expanded, audit


def simulate(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    data_dir: Path,
    initial_cash: float,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=TARGET_POSITIONS,
        action_dates=all_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            account_daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        value = baseline._compound(values)
        positive_years += int(value is not None and value > 0)
        yearly.append({"year": year, "account_return": value})
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
    }


def evaluate(primary: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
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
        "max_drawdown_no_worse_than_30pct": (
            metrics.get("max_drawdown") or -math.inf
        )
        >= -0.30,
        "at_least_5_positive_years": metrics["positive_years"] >= 5,
        "mean_cash_ratio_at_most_50pct": (
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.50,
        "buy_execution_at_least_90pct": execution["buy"]["execution_rate"]
        >= 0.90,
        "sell_execution_at_least_90pct": execution["sell"]["execution_rate"]
        >= 0.90,
        "no_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "cash_reconciled": integrity["max_cash_reconciliation_error"] <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "annualized_excess": excess,
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START
    )
    raw_source = main_board.filter_main_board(raw_all)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    candidates, signal_audit = build_candidates(load_events(data_dir), panel, all_dates)
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()
    tiers = {
        str(int(capital)): simulate(
            candidates, raw_source, all_dates, data_dir, capital
        )
        for capital in CAPITALS
    }
    decision = evaluate(tiers[str(int(PRIMARY_CAPITAL))], benchmark)
    payload = {
        "schema_version": "p0-main-board-forecast-drift-account-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "capital_tiers_cny": list(CAPITALS),
            "primary_capital_cny": PRIMARY_CAPITAL,
            "target_positions": TARGET_POSITIONS,
            "signal_lifetime_trading_days": SIGNAL_LIFETIME_TRADING_DAYS,
            "ranking": "p_change_min_desc_p_change_max_desc_symbol_asc",
        },
        "data": signal_audit,
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
            {
                "data": signal_audit,
                "benchmark": benchmark,
                "capital_tiers": tiers,
                "decision": decision,
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
        default=Path(
            "/app/data/research/p0_main_board_forecast_drift_account.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
