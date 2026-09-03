"""Run the frozen development account for idiosyncratic forecast surprise."""

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
import run_p0_industry_confirmed_forecast_drift_discovery as industry  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_forecast_drift_account as base  # noqa: E402
import run_p0_main_board_forecast_drift_regime_discovery as regime  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0


def load_events(data_dir: Path) -> pl.DataFrame:
    mapped, _audit = industry.attach_industry(
        industry.forecast.categorize_events(
            industry.forecast.load_forecasts(data_dir)
        ),
        industry.load_point_in_time_membership(data_dir),
    )
    return industry.classify_events(mapped).filter(
        pl.col("category") == industry.NEGATIVE_CONTROL
    )


def simulate_tier(
    candidates: pl.DataFrame,
    grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
    initial_cash: float,
) -> dict[str, Any]:
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=base.TARGET_POSITIONS,
        action_dates=all_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly: list[dict[str, Any]] = []
    positive_years = 0
    for year in range(base.DEVELOPMENT_START.year, base.DEVELOPMENT_END.year + 1):
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
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "daily_attempt_execution": account.execution_summary(simulation["orders"]),
        "intent_execution": regime.intent_execution_summary(
            simulation["orders"], all_dates
        ),
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
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else None
    )
    complete_round_trips = primary["account"]["trade_count"] // 2
    checks = {
        "annualized_at_least_20pct": (annualized or -math.inf) >= 0.20,
        "annualized_excess_at_least_10pp": (excess or -math.inf) >= 0.10,
        "max_drawdown_no_worse_than_30pct": (
            metrics.get("max_drawdown") or -math.inf
        )
        >= -0.30,
        "at_least_5_positive_years": metrics["positive_years"] >= 5,
        "mean_cash_ratio_at_most_70pct": (
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.70,
        "at_least_100_complete_round_trips": complete_round_trips >= 100,
        "buy_intent_execution_at_least_90pct": primary["intent_execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_intent_execution_at_least_90pct": primary["intent_execution"][
            "sell"
        ]["execution_rate"]
        >= 0.90,
        "no_unresolved_positions": primary["integrity"]["ending_unresolved_positions"]
        == 0,
        "cash_reconciled": primary["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "FREEZE_VALIDATION_CONTRACT" if passed else "TERMINATE",
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
    raw_all = baseline.load_daily(data_dir, end=base.DEVELOPMENT_END).filter(
        pl.col("date") >= base.DEVELOPMENT_START
    )
    raw_source = main_board.filter_main_board(raw_all)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = baseline.prepare_panel(
        baseline.attach_point_in_time_data(raw_source, data_dir)
    )
    candidates, signal_audit = base.build_candidates(
        load_events(data_dir), panel, all_dates
    )
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()

    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    tiers = {
        str(int(capital)): simulate_tier(
            candidates, grid, quotes, all_dates, capital
        )
        for capital in CAPITALS
    }
    decision = evaluate(tiers[str(int(PRIMARY_CAPITAL))], benchmark)
    payload = {
        "schema_version": "p0-idiosyncratic-forecast-surprise-account-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": base.DEVELOPMENT_START,
            "end": base.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "source_group": industry.NEGATIVE_CONTROL,
            "capital_tiers_cny": list(CAPITALS),
            "primary_capital_cny": PRIMARY_CAPITAL,
            "target_positions": base.TARGET_POSITIONS,
            "signal_lifetime_trading_days": base.SIGNAL_LIFETIME_TRADING_DAYS,
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
            "p0_idiosyncratic_forecast_surprise_account.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()

