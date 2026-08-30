"""Run the frozen development-only ETF absolute/relative momentum study."""
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
BASE_CASH = 200_000.0
CAPACITY_CASH = 1_000_000.0
TARGET_POSITIONS = 1
MIN_LISTING_DAYS = 180
MOMENTUM_DAYS = 120
LIQUIDITY_DAYS = 20
MIN_MEAN_AMOUNT = 50_000_000.0
ETF_LIMIT_PCT = 0.10
BENCHMARK_SYMBOL = "510300.SH"


def prepare_panel(
    daily: pl.DataFrame,
    adjustments: pl.DataFrame,
    master: pl.DataFrame,
) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    return (
        daily.join(
            adjustments.rename({"trade_date": "date"}),
            on=["symbol", "date"],
            how="inner",
        )
        .join(master.select("symbol", "list_date"), on="symbol", how="inner")
        .join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("open").alias("raw_open"),
            pl.col("close").alias("raw_close"),
            (pl.col("open") * pl.col("adj_factor")).alias("open"),
            (pl.col("close") * pl.col("adj_factor")).alias("close"),
            (pl.col("date") - pl.col("list_date"))
            .dt.total_days()
            .alias("listing_days"),
            pl.col("_global_index")
            .shift(MOMENTUM_DAYS)
            .over("symbol")
            .alias("_index_120d"),
            pl.col("close")
            .shift(MOMENTUM_DAYS)
            .over("symbol")
            .alias("_raw_close_120d"),
            pl.col("adj_factor")
            .shift(MOMENTUM_DAYS)
            .over("symbol")
            .alias("_adj_factor_120d"),
            (pl.col("close") * pl.col("adj_factor"))
            .rolling_mean(window_size=MOMENTUM_DAYS, min_samples=MOMENTUM_DAYS)
            .over("symbol")
            .alias("ma120"),
            pl.col("amount")
            .rolling_mean(window_size=LIQUIDITY_DAYS, min_samples=LIQUIDITY_DAYS)
            .over("symbol")
            .alias("mean_amount_20d"),
            pl.col("close").shift(1).over("symbol").alias("prev_raw_close"),
            pl.col("_global_index").shift(1).over("symbol").alias("_prev_index"),
        )
        .with_columns(
            pl.when(
                pl.col("_global_index")
                == pl.col("_index_120d") + MOMENTUM_DAYS
            )
            .then(
                pl.col("close")
                / (pl.col("_raw_close_120d") * pl.col("_adj_factor_120d"))
                - 1.0
            )
            .otherwise(None)
            .alias("momentum_120d"),
            pl.when(pl.col("_global_index") == pl.col("_prev_index") + 1)
            .then(pl.col("prev_raw_close"))
            .otherwise(None)
            .alias("reference_close"),
        )
        .with_columns(
            (
                (pl.col("reference_close") * (1.0 + ETF_LIMIT_PCT) * 1000 + 0.5)
                .floor()
                / 1000
            ).alias("limit_up_price"),
            (
                (pl.col("reference_close") * (1.0 - ETF_LIMIT_PCT) * 1000 + 0.5)
                .floor()
                / 1000
            ).alias("limit_down_price"),
            pl.lit(False).alias("is_excluded_name"),
        )
    )


def monthly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
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
        .filter(
            pl.col("signal_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )


def build_candidates(
    panel: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    return (
        panel.join(
            schedule, left_on="date", right_on="signal_date", how="inner"
        )
        .filter(
            (pl.col("listing_days") >= MIN_LISTING_DAYS)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT)
            & (pl.col("momentum_120d") > 0)
            & (pl.col("close") > pl.col("ma120"))
        )
        .sort(
            ["date", "momentum_120d", "mean_amount_20d", "symbol"],
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
            "momentum_120d",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


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
        scoped.select((pl.col("close") / pl.col("close").shift(1) - 1.0))
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
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    quotes = panel.filter(pl.col("symbol").is_in(candidate_symbols)).select(
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
            "annualized": annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
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
        "orders": simulation["orders"],
        "daily_equity": daily.select(
            "date", "equity", "cash", "position_value", "position_count"
        ).to_dicts(),
    }


def evaluate_gate(
    accounts: dict[str, dict[str, Any]], benchmark: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
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
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "etf_cross_asset_v2"
    daily = pl.read_parquet(root / "daily_raw.parquet")
    adjustments = pl.read_parquet(root / "adjustments.parquet")
    master = pl.read_parquet(root / "master.parquet")
    panel = prepare_panel(daily, adjustments, master)
    schedule = monthly_schedule(panel)
    candidates = build_candidates(panel, schedule)
    action_dates = schedule.get_column("entry_date").to_list()
    all_dates = (
        panel.filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    accounts = {
        "cny_200k": simulate(
            candidates, panel, all_dates, action_dates, BASE_CASH
        ),
        "cny_1m": simulate(
            candidates, panel, all_dates, action_dates, CAPACITY_CASH
        ),
    }
    benchmark = benchmark_metrics(panel)
    decision = evaluate_gate(accounts, benchmark)
    payload = {
        "schema_version": "p0-etf-dual-momentum-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "base_cash_cny": BASE_CASH,
            "capacity_cash_cny": CAPACITY_CASH,
            "target_positions": TARGET_POSITIONS,
            "momentum_days": MOMENTUM_DAYS,
            "minimum_listing_days": MIN_LISTING_DAYS,
            "liquidity_days": LIQUIDITY_DAYS,
            "minimum_mean_amount_cny": MIN_MEAN_AMOUNT,
            "commission_pct": baseline.COMMISSION_PCT,
            "slippage_pct": baseline.SLIPPAGE_PCT,
            "stamp_tax_pct": 0.0,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "execution": "monthly signal close, next trade day open",
        },
        "data": {
            "master_symbols": master.height,
            "daily_symbols": daily["symbol"].n_unique(),
            "signal_rows": candidates.height,
            "signal_symbols": candidates["symbol"].n_unique(),
            "scheduled_rebalances": len(action_dates),
            "active_rebalances": candidates["entry_date"].n_unique(),
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
                "accounts": {
                    name: {
                        "metrics": result["metrics"],
                        "execution": result["execution"],
                        "integrity": result["integrity"],
                        "account": result["account"],
                    }
                    for name, result in accounts.items()
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
        default=Path(
            "/app/data/research/p0_etf_dual_momentum_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
