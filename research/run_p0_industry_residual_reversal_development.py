"""Run the frozen development account for industry-neutral short reversal."""

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
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 10
MAX_PER_INDUSTRY = 2
LOOKBACK_DAYS = 5
MIN_INDUSTRY_MEMBERS = 20
MIN_STOCK_RETURN = -0.30
MAX_STOCK_RETURN = -0.05
MAX_RESIDUAL_RETURN = -0.03
MIN_MEAN_AMOUNT_20D = 50_000_000.0


def attach_residual_features(panel: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    work = (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("_global_index").shift(LOOKBACK_DAYS).over("symbol").alias("_index_5d"),
            pl.col("close").shift(LOOKBACK_DAYS).over("symbol").alias("_close_5d"),
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_index_5d") + LOOKBACK_DAYS)
            .then(pl.col("close") / pl.col("_close_5d") - 1.0)
            .otherwise(None)
            .alias("stock_return_5d")
        )
        .sort(["symbol", "date"])
        .join_asof(
            membership,
            left_on="date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("l1_code").is_not_null()
            & (pl.col("out_date").is_null() | (pl.col("date") <= pl.col("out_date")))
        )
    )
    industry = (
        work.drop_nulls("stock_return_5d")
        .group_by("date", "l1_code")
        .agg(
            pl.len().alias("industry_members"),
            pl.col("stock_return_5d").median().alias("industry_return_5d"),
        )
    )
    return work.join(industry, on=["date", "l1_code"], how="left").with_columns(
        (pl.col("stock_return_5d") - pl.col("industry_return_5d")).alias("residual_return_5d")
    )


def build_candidates(panel: pl.DataFrame) -> pl.DataFrame:
    weekly = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%G-%V").alias("week"),
        )
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )
    eligible = (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            & (pl.col("industry_members") >= MIN_INDUSTRY_MEMBERS)
            & pl.col("stock_return_5d").is_between(
                MIN_STOCK_RETURN, MAX_STOCK_RETURN, closed="both"
            )
            & (pl.col("residual_return_5d") <= MAX_RESIDUAL_RETURN)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .sort(
            ["date", "l1_code", "residual_return_5d", "amount", "symbol"],
            descending=[False, False, False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["date", "l1_code"]).alias("industry_rank")
        )
        .filter(pl.col("industry_rank") <= MAX_PER_INDUSTRY)
        .sort(
            ["date", "residual_return_5d", "amount", "market_cap", "symbol"],
            descending=[False, False, True, False, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("portfolio_rank"))
        .filter(pl.col("portfolio_rank") <= TARGET_POSITIONS)
    )
    return eligible.select(
        "date",
        "entry_date",
        "symbol",
        "l1_code",
        "l1_name",
        "stock_return_5d",
        "industry_return_5d",
        "residual_return_5d",
        "industry_rank",
        "market_cap",
        pl.col("amount").alias("signal_amount"),
        pl.col("portfolio_rank").alias("cap_rank"),
    ).sort(["entry_date", "cap_rank", "symbol"])


def _annualized(returns: list[float]) -> float | None:
    total = baseline._compound(returns)
    if not returns or total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(returns)) - 1.0


def benchmark_metrics(panel: pl.DataFrame) -> dict[str, Any]:
    daily = (
        panel.filter(
            pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            & pl.col("daily_return").is_finite()
        )
        .group_by("date")
        .agg(pl.col("daily_return").mean().alias("return"))
        .sort("date")
    )
    returns = daily.get_column("return").to_list()
    return {
        "trading_days": daily.height,
        "annualized": _annualized(returns),
        "total_return": baseline._compound(returns),
        "max_drawdown": baseline._max_drawdown(returns),
    }


def simulate(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    data_dir: Path,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir)
    )
    grid = account.build_execution_grid(candidates, quotes)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
            "trading_days": daily.height,
            "annualized": _annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        },
        "account": account.account_summary(simulation, daily),
        "signal_rows": candidates.height,
        "rebalance_days": candidates.get_column("entry_date").n_unique(),
        "worst_weeks": account.worst_weeks(daily),
    }


def evaluate_gate(result: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    annualized = metrics.get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else None
    )
    checks = {
        "annualized_at_least_50pct": (annualized or -math.inf) >= 0.50,
        "annualized_excess_at_least_20pp": (excess or -math.inf) >= 0.20,
        "max_drawdown_no_worse_than_30pct": (metrics.get("max_drawdown") or -math.inf) >= -0.30,
        "at_least_five_positive_years": metrics.get("positive_years", 0) >= 5,
        "mean_cash_ratio_at_most_25pct": (metrics.get("mean_cash_ratio") or math.inf) <= 0.25,
        "buy_execution_at_least_90pct": result["execution"]["buy"]["execution_rate"] >= 0.90,
        "sell_execution_at_least_90pct": result["execution"]["sell"]["execution_rate"] >= 0.90,
        "ending_positions_resolved": result["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": result["integrity"]["max_cash_reconciliation_error"] <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
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
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    if raw_source.is_empty():
        raise ValueError("no development daily data")
    all_dates = (
        raw_source.filter(pl.col("date") >= DEVELOPMENT_START)
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    membership, _ = shared.load_industry_data(data_dir)
    panel = attach_residual_features(panel, membership)
    candidates = build_candidates(panel)
    benchmark = benchmark_metrics(panel)
    del panel, membership
    gc.collect()
    result = simulate(candidates, raw_source, all_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-industry-residual-reversal-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "initial_cash_cny": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "max_positions_per_industry": MAX_PER_INDUSTRY,
            "lookback_trading_days": LOOKBACK_DAYS,
            "minimum_industry_members": MIN_INDUSTRY_MEMBERS,
            "stock_return_range": [MIN_STOCK_RETURN, MAX_STOCK_RETURN],
            "maximum_residual_return": MAX_RESIDUAL_RETURN,
            "minimum_mean_amount_20d_cny": MIN_MEAN_AMOUNT_20D,
            "execution": "weekly next trading day open, sells before buys",
            "benchmark": "PIT eligible all-A equal-weight daily return",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "signal_rows": candidates.height,
            "rebalance_days": candidates.get_column("entry_date").n_unique(),
            "signal_symbols": candidates.get_column("symbol").n_unique(),
        },
        "benchmark": benchmark,
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
            {
                "data": payload["data"],
                "benchmark": benchmark,
                "strategy": {
                    "metrics": result["metrics"],
                    "execution": result["execution"],
                    "integrity": result["integrity"],
                    "account": result["account"],
                    "worst_weeks": result["worst_weeks"],
                },
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
        default=Path("/app/data/research/p0_industry_residual_reversal_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
