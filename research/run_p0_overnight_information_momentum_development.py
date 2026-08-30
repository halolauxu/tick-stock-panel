"""Run the frozen development-only overnight-information momentum account."""

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
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 2)
DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 10
LOOKBACK_DAYS = 20
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0


def attach_overnight_features(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("_global_index").shift(1).over("symbol").alias("_prev_index"),
            pl.col("close").shift(1).over("symbol").alias("_prev_close"),
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_prev_index") + 1)
            .then(pl.col("open") / pl.col("_prev_close") - 1.0)
            .otherwise(None)
            .alias("overnight_return")
        )
        .with_columns(
            pl.col("overnight_return")
            .rolling_mean(window_size=LOOKBACK_DAYS, min_samples=LOOKBACK_DAYS)
            .over("symbol")
            .alias("overnight_momentum_20d")
        )
    )


def monthly_signal_panel(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[date]]:
    schedule = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
        )
        .group_by("month", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .filter(pl.col("entry_date") >= pl.lit(DEVELOPMENT_START))
    )
    monthly = panel.join(
        schedule.select("signal_date", "entry_date"),
        left_on="date",
        right_on="signal_date",
        how="inner",
    )
    return monthly, schedule.get_column("entry_date").to_list()


def build_candidates(monthly: pl.DataFrame) -> pl.DataFrame:
    return (
        monthly.filter(
            (pl.col("market_cap") >= MIN_MARKET_CAP)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & pl.col("overnight_momentum_20d").is_finite()
        )
        .sort(
            ["date", "overnight_momentum_20d", "mean_amount_20d", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            pl.col("overnight_momentum_20d").alias("factor_value"),
            "market_cap",
            pl.col("amount").alias("signal_amount"),
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
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
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
        "annualized": (annualized or -math.inf) >= 0.50,
        "annualized_excess": excess >= 0.20,
        "max_drawdown": (metrics.get("max_drawdown") or -math.inf) >= -0.30,
        "positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "buy_execution": result["execution"]["buy"]["execution_rate"] >= 0.90,
        "sell_execution": result["execution"]["sell"]["execution_rate"] >= 0.90,
        "ending_unresolved_positions": result["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciliation": result["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 0.01,
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "annualized_excess": excess if math.isfinite(excess) else None,
        "counts_toward_50pct_goal": False,
        "next_step": (
            "freeze_independent_validation"
            if all(checks.values())
            else "terminate_overnight_information_momentum"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    raw_source = raw_all.filter(pl.col("date") >= DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = attach_overnight_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= DEVELOPMENT_START)
    )
    monthly, action_dates = monthly_signal_panel(panel)
    candidates = build_candidates(monthly)
    del panel, monthly
    gc.collect()
    result = simulate(candidates, raw_source, all_dates, action_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-overnight-information-momentum-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "lookback_observations": LOOKBACK_DAYS,
            "direction": "high_past_overnight_return",
            "rebalance": "monthly signal close, next trading open",
            "initial_cash": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "minimum_market_cap": MIN_MARKET_CAP,
            "minimum_mean_amount_20d": MIN_MEAN_AMOUNT_20D,
        },
        "data": {
            "candidate_rows": candidates.height,
            "candidate_symbols": candidates.get_column("symbol").n_unique(),
            "rebalance_days": len(action_dates),
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
        default=Path(
            "/app/data/research/p0_overnight_information_momentum_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
