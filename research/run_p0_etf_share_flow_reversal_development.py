"""Run the frozen development-only ETF share-flow reversal account study."""
from __future__ import annotations

import argparse
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

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
CAPITAL_LEVELS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 5
MIN_LISTING_DAYS = 180
FLOW_LOOKBACK_DAYS = 5
FLOW_CUTOFF = -0.05
FLOW_DECILE = 0
LIQUIDITY_DAYS = 20
MIN_MEAN_AMOUNT = 20_000_000.0
ETF_LIMIT_PCT = 0.10
BENCHMARK_SYMBOL = "510300.SH"


def prepare_panel(
    daily: pl.DataFrame,
    adjustments: pl.DataFrame,
    master: pl.DataFrame,
    shares: pl.DataFrame,
) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    panel = (
        daily.join(
            adjustments.rename({"trade_date": "date"}),
            on=["symbol", "date"],
            how="inner",
        )
        .join(
            master.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
        .join(shares, on=["symbol", "date"], how="left")
        .join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("open").alias("raw_open"),
            pl.col("close").alias("raw_close"),
            (pl.col("open") * pl.col("adj_factor")).alias("open"),
            (pl.col("close") * pl.col("adj_factor")).alias("close"),
            (pl.col("shares_10k") / pl.col("adj_factor")).alias(
                "split_adjusted_shares"
            ),
            (pl.col("date") - pl.col("list_date"))
            .dt.total_days()
            .alias("listing_days"),
        )
        .with_columns(
            pl.col("amount")
            .rolling_mean(
                window_size=LIQUIDITY_DAYS,
                min_samples=LIQUIDITY_DAYS,
            )
            .over("symbol")
            .alias("_mean_amount_20d"),
            pl.col("_global_index")
            .shift(LIQUIDITY_DAYS - 1)
            .over("symbol")
            .alias("_index_19d"),
            pl.col("split_adjusted_shares")
            .shift(1)
            .over("symbol")
            .alias("_known_shares"),
            pl.col("_global_index")
            .shift(1)
            .over("symbol")
            .alias("_known_index"),
            pl.col("split_adjusted_shares")
            .shift(FLOW_LOOKBACK_DAYS + 1)
            .over("symbol")
            .alias("_prior_shares"),
            pl.col("_global_index")
            .shift(FLOW_LOOKBACK_DAYS + 1)
            .over("symbol")
            .alias("_prior_index"),
            pl.col("raw_close").shift(1).over("symbol").alias("_prev_raw_close"),
            pl.col("adj_factor").shift(1).over("symbol").alias("_prev_adj_factor"),
            pl.col("_global_index").shift(1).over("symbol").alias("_prev_index"),
        )
        .with_columns(
            pl.when(
                pl.col("_global_index")
                == pl.col("_index_19d") + LIQUIDITY_DAYS - 1
            )
            .then(pl.col("_mean_amount_20d"))
            .otherwise(None)
            .alias("mean_amount_20d"),
            pl.when(
                (pl.col("_global_index") == pl.col("_known_index") + 1)
                & (
                    pl.col("_global_index")
                    == pl.col("_prior_index") + FLOW_LOOKBACK_DAYS + 1
                )
            )
            .then(pl.col("_known_shares") / pl.col("_prior_shares") - 1.0)
            .otherwise(None)
            .alias("share_flow_5d"),
            pl.when(pl.col("_global_index") == pl.col("_prev_index") + 1)
            .then(
                pl.col("_prev_raw_close")
                * pl.col("_prev_adj_factor")
                / pl.col("adj_factor")
            )
            .otherwise(None)
            .alias("reference_close"),
        )
        .with_columns(
            (
                (
                    pl.col("reference_close")
                    * (1.0 + ETF_LIMIT_PCT)
                    * 1000
                    + 0.5
                ).floor()
                / 1000
            ).alias("limit_up_price"),
            (
                (
                    pl.col("reference_close")
                    * (1.0 - ETF_LIMIT_PCT)
                    * 1000
                    + 0.5
                ).floor()
                / 1000
            ).alias("limit_down_price"),
            pl.lit(False).alias("is_excluded_name"),
        )
    )
    return panel


def weekly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
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
        .filter(
            pl.col("signal_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )


def build_candidates(
    panel: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    eligible = (
        panel.join(
            schedule, left_on="date", right_on="signal_date", how="inner"
        )
        .filter(
            (pl.col("listing_days") >= MIN_LISTING_DAYS)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT)
            & pl.col("share_flow_5d").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("share_flow_5d")
            .rank(method="ordinal")
            .over("date")
            .alias("flow_rank"),
        )
        .with_columns(
            (
                ((pl.col("flow_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("flow_decile")
        )
        .filter(
            (pl.col("flow_decile") == FLOW_DECILE)
            & (pl.col("share_flow_5d") <= FLOW_CUTOFF)
        )
        .sort(
            ["date", "share_flow_5d", "mean_amount_20d", "symbol"],
            descending=[False, False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
    )
    return eligible.select(
        "date",
        "entry_date",
        "symbol",
        "share_flow_5d",
        pl.col("amount").alias("signal_amount"),
        "mean_amount_20d",
        "cap_rank",
    ).sort(["entry_date", "cap_rank", "symbol"])


def build_execution_grid(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    action_dates: list[date],
) -> pl.DataFrame:
    symbols = candidates.select("symbol").unique().sort("symbol")
    grid = symbols.join(pl.DataFrame({"entry_date": action_dates}), how="cross")
    history = quotes.rename(
        {
            "date": "quote_date",
            "amount": "entry_amount",
            "volume": "entry_volume",
        }
    ).sort(["symbol", "quote_date"])
    return (
        grid.sort(["symbol", "entry_date"])
        .join_asof(
            history,
            left_on="entry_date",
            right_on="quote_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("quote_date") == pl.col("entry_date")).alias("exact_quote")
        )
        .sort(["entry_date", "symbol"])
    )


def annualized(returns: list[float]) -> float | None:
    total = baseline._compound(returns)
    if not returns or total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(returns)) - 1.0


def benchmark_metrics(panel: pl.DataFrame) -> dict[str, Any]:
    scoped = panel.filter(
        (pl.col("symbol") == BENCHMARK_SYMBOL)
        & pl.col("date").is_between(
            DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
        )
    ).sort("date")
    returns = (
        scoped.select(pl.col("close") / pl.col("close").shift(1) - 1.0)
        .to_series()
        .drop_nulls()
        .to_list()
    )
    return {
        "symbol": BENCHMARK_SYMBOL,
        "trading_days": len(returns),
        "annualized": annualized(returns),
        "total_return": baseline._compound(returns),
        "max_drawdown": baseline._max_drawdown(returns),
    }


def simulate(
    candidates: pl.DataFrame,
    panel: pl.DataFrame,
    all_dates: list[date],
    action_dates: list[date],
    initial_cash: float,
) -> dict[str, Any]:
    symbols = candidates["symbol"].unique().to_list()
    quotes = panel.filter(pl.col("symbol").is_in(symbols)).select(
        "symbol",
        "date",
        "open",
        "raw_open",
        "close",
        "raw_close",
        "volume",
        "amount",
        "limit_up_price",
        "limit_down_price",
        "is_excluded_name",
    )
    grid = build_execution_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
        stamp_tax_rate=0.0,
    )
    daily, integrity = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = daily["daily_return"].drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            daily.filter(pl.col("date").dt.year() == year)["daily_return"]
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
            "trading_days": daily.height,
            "annualized": annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": daily["cash_ratio"].mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **integrity,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
    }


def evaluate_gate(
    accounts: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
    active_rebalances: int,
) -> dict[str, Any]:
    checks = {"at_least_100_active_rebalances": active_rebalances >= 100}
    benchmark_annualized = benchmark.get("annualized")
    for name, result in accounts.items():
        metrics = result["metrics"]
        strategy_annualized = metrics.get("annualized")
        excess = (
            strategy_annualized - benchmark_annualized
            if strategy_annualized is not None
            and benchmark_annualized is not None
            else -math.inf
        )
        checks.update(
            {
                f"{name}_annualized_at_least_50pct": (
                    strategy_annualized or -math.inf
                )
                >= 0.50,
                f"{name}_excess_at_least_20pp": excess >= 0.20,
                f"{name}_drawdown_no_worse_than_30pct": (
                    metrics.get("max_drawdown") or -math.inf
                )
                >= -0.30,
                f"{name}_at_least_five_positive_years": metrics.get(
                    "positive_years", 0
                )
                >= 5,
                f"{name}_buy_execution_at_least_90pct": result["execution"][
                    "buy"
                ]["execution_rate"]
                >= 0.90,
                f"{name}_sell_execution_at_least_90pct": result["execution"][
                    "sell"
                ]["execution_rate"]
                >= 0.90,
                f"{name}_ending_positions_zero": result["account"][
                    "ending_positions"
                ]
                == 0,
                f"{name}_cash_reconciled": result["integrity"][
                    "max_cash_reconciliation_error"
                ]
                <= 0.01,
            }
        )
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
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
    price_root = data_dir / "research" / "etf_cross_asset_v2"
    flow_root = data_dir / "research" / "etf_share_flow"
    daily = pl.read_parquet(price_root / "daily_raw.parquet")
    adjustments = pl.read_parquet(price_root / "adjustments.parquet")
    master = pl.read_parquet(flow_root / "master.parquet")
    shares = pl.read_parquet(flow_root / "share_history.parquet")
    panel = prepare_panel(daily, adjustments, master, shares)
    schedule = weekly_schedule(panel)
    candidates = build_candidates(panel, schedule)
    all_dates = (
        panel.filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )["date"]
        .unique()
        .sort()
        .to_list()
    )
    action_dates = schedule["entry_date"].to_list()
    if all_dates[-1] not in action_dates:
        action_dates.append(all_dates[-1])
    action_dates = sorted(set(action_dates))
    accounts = {
        f"cny_{int(cash / 1000)}k": simulate(
            candidates, panel, all_dates, action_dates, cash
        )
        for cash in CAPITAL_LEVELS
    }
    benchmark = benchmark_metrics(panel)
    active_rebalances = candidates["entry_date"].n_unique()
    decision = evaluate_gate(accounts, benchmark, active_rebalances)
    payload = {
        "schema_version": "p0-etf-share-flow-reversal-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "capital_levels_cny": CAPITAL_LEVELS,
            "target_positions": TARGET_POSITIONS,
            "flow_lookback_days": FLOW_LOOKBACK_DAYS,
            "flow_cutoff": FLOW_CUTOFF,
            "flow_decile": FLOW_DECILE,
            "share_information_lag_sessions": 1,
            "split_adjustment": "shares_10k / adj_factor",
            "minimum_listing_days": MIN_LISTING_DAYS,
            "liquidity_days": LIQUIDITY_DAYS,
            "minimum_mean_amount_cny": MIN_MEAN_AMOUNT,
            "commission_pct": baseline.COMMISSION_PCT,
            "minimum_commission_cny": account.MIN_COMMISSION,
            "slippage_pct": baseline.SLIPPAGE_PCT,
            "stamp_tax_pct": 0.0,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "execution": "weekly signal close, next trade day open",
        },
        "data": {
            "master_symbols": master.height,
            "share_symbols": shares["symbol"].n_unique(),
            "signal_rows": candidates.height,
            "signal_symbols": candidates["symbol"].n_unique(),
            "scheduled_rebalances": schedule.height,
            "active_rebalances": active_rebalances,
        },
        "benchmark": benchmark,
        "accounts": accounts,
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
                "accounts": accounts,
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
            "/app/data/research/"
            "p0_etf_share_flow_reversal_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
