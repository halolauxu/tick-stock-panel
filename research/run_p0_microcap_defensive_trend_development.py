"""Run the frozen micro-cap defensive-trend development account."""

from __future__ import annotations

import argparse
import gc
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

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 10
ACCOUNT_START = date(2014, 1, 2)
DEVELOPMENT_END = date(2020, 12, 31)


def load_daily(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        return pl.DataFrame()
    return (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "close",
            "volume",
            "amount",
            "raw_close",
            "momentum_60d",
            "annual_vol_20d",
        )
        .filter(
            (pl.col("date") >= pl.lit(baseline.START))
            & (pl.col("date") <= pl.lit(DEVELOPMENT_END))
            & pl.col("symbol").str.contains(baseline.SYMBOL_PATTERN)
        )
        .collect(engine="streaming")
    )


def defensive_filter(ranked: pl.DataFrame) -> pl.DataFrame:
    micro = (
        ranked.filter(pl.col("cap_decile") == 0)
        .with_columns(
            pl.col("annual_vol_20d").median().over("date").alias("micro_vol_median")
        )
        .filter(
            (pl.col("momentum_60d") > 0)
            & pl.col("annual_vol_20d").is_not_null()
            & (pl.col("annual_vol_20d") <= pl.col("micro_vol_median"))
        )
    )
    return micro.sort(["entry_date", "cap_rank", "symbol"])


def build_signal_candidates(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[date]]:
    dates = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("entry_date"))
    )
    weekly = (
        dates.with_columns(pl.col("date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .filter(pl.col("entry_date") >= pl.lit(ACCOUNT_START))
    )
    ranked = (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
            & pl.col("momentum_60d").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile")
        )
    )
    selected = defensive_filter(ranked).select(
        "date",
        "entry_date",
        "symbol",
        "market_cap",
        pl.col("amount").alias("signal_amount"),
        "cap_rank",
        "momentum_60d",
        "annual_vol_20d",
        "micro_vol_median",
    )
    return selected, weekly.get_column("entry_date").to_list()


def _account_metrics(
    daily: pl.DataFrame,
    weekly_market: pl.DataFrame,
) -> dict[str, Any]:
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    total = baseline._compound(returns)
    annualized = (
        (1.0 + total) ** (252.0 / len(returns)) - 1.0
        if returns and total is not None and total > -1.0
        else None
    )
    market_annualized = baseline._annualized(
        weekly_market.get_column("market_net").drop_nulls().to_list()
    )
    equity = daily.get_column("equity")
    drawdown = (equity / equity.cum_max() - 1.0).min()
    yearly = []
    positive_years = 0
    for year in range(2014, 2021):
        year_return = baseline._compound(
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        positive_years += int(year_return is not None and year_return > 0)
        yearly.append({"year": year, "account_return": year_return})
    return {
        "account_total_return": total,
        "account_annualized": annualized,
        "market_annualized": market_annualized,
        "annualized_excess": (
            annualized - market_annualized
            if annualized is not None and market_annualized is not None
            else None
        ),
        "max_drawdown": drawdown,
        "positive_years": positive_years,
        "yearly": yearly,
    }


def evaluate_gate(
    metrics: dict[str, Any],
    execution: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "annualized": (metrics.get("account_annualized") or -99.0) >= 0.50,
        "annualized_excess": (metrics.get("annualized_excess") or -99.0) >= 0.20,
        "max_drawdown": (metrics.get("max_drawdown") or -99.0) >= -0.30,
        "positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "buy_execution": execution["buy"]["execution_rate"] >= 0.90,
        "sell_execution": execution["sell"]["execution_rate"] >= 0.90,
        "ending_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "cash_reconciliation": integrity["max_cash_reconciliation_error"] <= 0.01,
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "next_step": (
            "freeze_independent_validation"
            if all(checks.values())
            else "terminate_microcap_defensive_trend"
        ),
        "counts_toward_50pct_goal": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source = load_daily(data_dir)
    if source.is_empty():
        raise ValueError("no enriched daily data")
    account_dates = [
        day
        for day in source.get_column("date").unique().sort().to_list()
        if day >= ACCOUNT_START
    ]
    factors = source.select("symbol", "date", "momentum_60d", "annual_vol_20d")
    base = source.drop("momentum_60d", "annual_vol_20d")
    pit = baseline.attach_point_in_time_data(base, data_dir)
    del source, base
    gc.collect()
    panel = baseline.prepare_panel(pit).join(factors, on=["symbol", "date"], how="left")
    del pit, factors
    gc.collect()

    candidates, action_dates = build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = (
        baseline.weekly_portfolios(observations)
        .filter(
            (pl.col("date") >= pl.lit(ACCOUNT_START))
            & (pl.col("date") <= pl.lit(DEVELOPMENT_END))
        )
        .select("date", "market_net")
    )
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del panel, observations
    gc.collect()
    if not candidate_symbols:
        raise ValueError("defensive trend filter produced no development candidates")

    quote_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("symbol").is_in(candidate_symbols)
    )
    quote_source = account.attach_quote_names(quote_source, data_dir)
    quotes = account.prepare_quote_panel(quote_source)
    del quote_source
    gc.collect()

    dummy_symbol = candidate_symbols[0]
    grid_keys = pl.concat(
        [
            candidates.select("symbol", "entry_date"),
            pl.DataFrame(
                {"symbol": [dummy_symbol] * len(action_dates), "entry_date": action_dates}
            ),
        ],
        how="vertical_relaxed",
    )
    execution_grid = account.build_execution_grid(grid_keys, quotes)
    simulation = account.simulate_account(
        candidates,
        execution_grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
    )
    daily, stale = account.build_daily_equity(
        simulation,
        quotes,
        account_dates,
        initial_cash=INITIAL_CASH,
    )
    execution = account.execution_summary(simulation["orders"])
    integrity = {
        **stale,
        "max_cash_reconciliation_error": simulation[
            "max_cash_reconciliation_error"
        ],
    }
    metrics = _account_metrics(daily, weekly_market)
    decision = evaluate_gate(metrics, execution, integrity)
    payload = {
        "schema_version": "p0-microcap-defensive-trend-development-v1",
        "period": {
            "start": ACCOUNT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "contract": {
            "initial_cash": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "cap_decile": 0,
            "momentum_60d_minimum": 0.0,
            "annual_vol_20d_maximum": "same-day microcap median",
            "execution": "weekly signal, next trading open, sells before buys",
        },
        "data": {
            "signal_rows": candidates.height,
            "symbols": len(candidate_symbols),
            "rebalance_days": len(action_dates),
        },
        "metrics": metrics,
        "execution": execution,
        "integrity": integrity,
        "account": account.account_summary(simulation, daily),
        "worst_weeks": account.worst_weeks(daily),
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_microcap_defensive_trend_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
