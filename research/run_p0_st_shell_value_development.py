"""Run the frozen development-only ordinary-ST shell-value account study."""
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
sys.path.insert(0, str(RESEARCH))

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_END = baseline.DEVELOPMENT_END
CAPITAL_TIERS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
CANDIDATE_QUEUE = 30


def attach_st_point_in_time_data(
    panel: pl.DataFrame, data_dir: Path
) -> pl.DataFrame:
    research = data_dir / "research"
    universe = (
        pl.read_parquet(research / "historical_stock_universe_all_a.parquet")
        .with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "list_date", "delist_date")
    )
    names = (
        pl.read_parquet(research / "historical_stock_names_all_a.parquet")
        .with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "name", "start_date", "end_date")
        .sort(["symbol", "start_date"])
    )
    shares = baseline.load_share_history(data_dir)
    return (
        panel.with_columns(pl.col("date").cast(pl.Date))
        .join(universe, on="symbol", how="left")
        .filter(
            pl.col("list_date").is_not_null()
            & (pl.col("date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("date") <= pl.col("delist_date"))
            )
            & (
                (pl.col("date") - pl.col("list_date")).dt.total_days()
                >= baseline.MIN_LISTING_DAYS
            )
        )
        .sort(["symbol", "date"])
        .join_asof(
            names,
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("name").is_not_null()
            & (
                pl.col("end_date").is_null()
                | (pl.col("date") <= pl.col("end_date"))
            )
            & pl.col("name").str.to_uppercase().str.contains("ST")
            & ~pl.col("name")
            .str.to_uppercase()
            .str.contains(r"(?:\*ST|退)")
        )
        .join_asof(
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            (pl.col("total_shares") > 0)
            & (pl.col("float_shares") > 0)
            & (pl.col("float_shares") <= pl.col("total_shares"))
        )
        .drop("delist_date", "start_date", "end_date", "available_date")
    )


def build_signal_candidates(panel: pl.DataFrame) -> pl.DataFrame:
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
    return (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= CANDIDATE_QUEUE)
        .select(
            "date",
            "entry_date",
            "symbol",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _run_tier(
    *,
    capital: float,
    candidates: pl.DataFrame,
    execution_grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
    weekly_market: pl.DataFrame,
    include_audit: bool,
) -> dict[str, Any]:
    simulation = account.simulate_account(
        candidates,
        execution_grid,
        initial_cash=capital,
        target_positions=TARGET_POSITIONS,
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=capital
    )
    metric = next(
        row
        for row in account.account_period_metrics(daily, weekly_market)
        if row["period"] == "development"
    )
    payload = {
        "capital": capital,
        "metrics": metric,
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
        "worst_weeks": account.worst_weeks(daily),
    }
    if include_audit:
        payload["daily_equity"] = daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts()
        payload["orders"] = simulation["orders"]
        payload["rebalance_snapshots"] = simulation["snapshots"]
    return payload


def evaluate_development(main: dict[str, Any]) -> dict[str, Any]:
    metrics = main["metrics"]
    full_years = [
        row
        for row in metrics["yearly"]
        if 2014 <= int(row["year"]) <= 2020
    ]
    checks = {
        "annualized_at_least_50pct": (
            metrics.get("account_annualized") or -math.inf
        )
        >= 0.50,
        "annualized_excess_at_least_20pp": (
            metrics.get("annualized_excess") or -math.inf
        )
        >= 0.20,
        "max_drawdown_not_worse_than_30pct": (
            metrics.get("account_max_drawdown") or -math.inf
        )
        >= -0.30,
        "at_least_5_of_7_positive_full_years": sum(
            (row.get("account_return") or 0.0) > 0 for row in full_years
        )
        >= 5,
        "buy_execution_at_least_90pct": main["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_execution_at_least_90pct": main["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "no_unresolved_valuation": main["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_error_at_most_one_cent": main["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_INDEPENDENT_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "validation_read": False,
        "known_stress_read": False,
        "counts_toward_50pct_goal": False,
        "next_step": (
            "freeze_development_result_then_run_2021_2023_validation"
            if passed
            else "terminate_st_shell_value_and_move_to_next_mechanism"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    if source.is_empty():
        raise ValueError("development daily data are required")
    all_dates = source.get_column("date").unique().sort().to_list()

    market_pit = baseline.attach_point_in_time_data(source, data_dir)
    market_panel = baseline.prepare_panel(market_pit)
    market_observations = baseline.build_weekly_observations(market_panel)
    weekly_market = baseline.weekly_portfolios(market_observations).select(
        "date", "period", "market_net"
    )
    del market_pit, market_panel, market_observations
    gc.collect()

    st_pit = attach_st_point_in_time_data(source, data_dir)
    st_rows = st_pit.height
    st_symbols = st_pit.get_column("symbol").n_unique()
    signal_panel = baseline.prepare_panel(st_pit)
    del st_pit, source
    gc.collect()
    candidates = build_signal_candidates(signal_panel)
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del signal_panel
    gc.collect()

    source_quotes = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("symbol").is_in(candidate_symbols)
    )
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes).with_columns(
        pl.lit(False).alias("is_excluded_name")
    )
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(candidates, quotes)

    tiers: dict[str, Any] = {}
    for capital in CAPITAL_TIERS:
        tiers[str(int(capital))] = _run_tier(
            capital=capital,
            candidates=candidates,
            execution_grid=execution_grid,
            quotes=quotes,
            all_dates=all_dates,
            weekly_market=weekly_market,
            include_audit=capital == CAPITAL_TIERS[0],
        )
    decision = evaluate_development(tiers["200000"])
    payload = {
        "schema_version": "p0-st-shell-value-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": all_dates[0],
            "end": all_dates[-1],
            "role": "development_only",
        },
        "assumptions": {
            "initial_capital_tiers_cny": CAPITAL_TIERS,
            "target_positions": TARGET_POSITIONS,
            "candidate_queue": CANDIDATE_QUEUE,
            "signal": "weekly_smallest_pit_market_cap_ordinary_st",
            "excluded_names": ["*ST", "退"],
            "execution": "next_trade_day_open_sells_before_buys",
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "lot_size": account.LOT_SIZE,
            "commission_rate": baseline.COMMISSION_PCT,
            "minimum_commission": account.MIN_COMMISSION,
            "slippage_rate_each_side": baseline.SLIPPAGE_PCT,
        },
        "data": {
            "trading_days": len(all_dates),
            "st_point_in_time_rows": st_rows,
            "st_symbols": st_symbols,
            "candidate_symbols": len(candidate_symbols),
            "signal_rows": candidates.height,
            "rebalance_days": candidates.get_column("entry_date").n_unique(),
        },
        "tiers": tiers,
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
            {
                "data": payload["data"],
                "tiers": {
                    key: {
                        "metrics": value["metrics"],
                        "execution": value["execution"],
                        "integrity": value["integrity"],
                        "account": value["account"],
                    }
                    for key, value in tiers.items()
                },
                "decision": decision,
                "output": str(output),
                "sha256": sha256,
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
        default=Path("/app/data/research/p0_st_shell_value_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
