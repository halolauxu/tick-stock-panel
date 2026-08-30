"""Run the frozen development-only convertible-bond double-low study."""
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

DEVELOPMENT_START = date(2017, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
BASE_CASH = 200_000.0
CAPACITY_CASH = 1_000_000.0
TARGET_POSITIONS = 10
LOT_SIZE = 10
MIN_LISTING_DAYS = 30
MIN_MATURITY_DAYS = 180
LIQUIDITY_DAYS = 20
MIN_MEAN_AMOUNT = 10_000_000.0
MIN_PRICE = 80.0
MAX_PRICE = 130.0
MIN_CONVERSION_PREMIUM = -10.0
MAX_CONVERSION_PREMIUM = 50.0
MIN_BOND_VALUE = 80.0


def prepare_panel(daily: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    return (
        daily.join(
            master.select(
                "symbol", "list_date", "delist_date", "maturity_date"
            ),
            on="symbol",
            how="inner",
        )
        .join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("date") - pl.col("list_date"))
            .dt.total_days()
            .alias("listing_days"),
            (pl.col("maturity_date") - pl.col("date"))
            .dt.total_days()
            .alias("maturity_days"),
            pl.col("amount")
            .rolling_mean(window_size=LIQUIDITY_DAYS, min_samples=LIQUIDITY_DAYS)
            .over("symbol")
            .alias("mean_amount_20d"),
            pl.col("_global_index").shift(1).over("symbol").alias("_prev_index"),
            pl.col("close").shift(1).over("symbol").alias("_prev_close"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_prev_index") + 1)
            .then(pl.col("close") / pl.col("_prev_close") - 1.0)
            .otherwise(None)
            .alias("daily_return"),
            (pl.col("close") + pl.col("cb_over_rate")).alias(
                "double_low_score"
            ),
        )
    )


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
    return (
        panel.join(
            schedule, left_on="date", right_on="signal_date", how="inner"
        )
        .filter(
            (pl.col("listing_days") >= MIN_LISTING_DAYS)
            & (pl.col("maturity_days") >= MIN_MATURITY_DAYS)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT)
            & pl.col("close").is_between(MIN_PRICE, MAX_PRICE, closed="both")
            & pl.col("cb_over_rate").is_between(
                MIN_CONVERSION_PREMIUM,
                MAX_CONVERSION_PREMIUM,
                closed="both",
            )
            & (pl.col("cb_value") > 0)
            & (pl.col("bond_value") >= MIN_BOND_VALUE)
            & (pl.col("volume") > 0)
            & (pl.col("amount") > 0)
        )
        .sort(
            ["date", "double_low_score", "mean_amount_20d", "symbol"],
            descending=[False, False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "double_low_score",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def prepare_quotes(panel: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    return panel.filter(pl.col("symbol").is_in(symbols)).select(
        "symbol",
        "date",
        "open",
        pl.col("open").alias("raw_open"),
        "close",
        pl.col("close").alias("raw_close"),
        "volume",
        "amount",
        pl.lit(None, dtype=pl.Float64).alias("limit_up_price"),
        pl.lit(None, dtype=pl.Float64).alias("limit_down_price"),
        pl.lit(False).alias("is_excluded_name"),
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
    daily = (
        panel.filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & pl.col("daily_return").is_finite()
            & (pl.col("volume") > 0)
            & (pl.col("amount") > 0)
        )
        .group_by("date")
        .agg(pl.col("daily_return").mean().alias("return"))
        .sort("date")
    )
    returns = daily["return"].to_list()
    return {
        "name": "active_convertible_bond_equal_weight",
        "trading_days": daily.height,
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
    quotes = prepare_quotes(panel, symbols)
    grid = build_execution_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
        stamp_tax_rate=0.0,
        lot_size=LOT_SIZE,
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
                f"{name}_at_least_three_positive_years": metrics.get(
                    "positive_years", 0
                )
                >= 3,
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
    root = data_dir / "research" / "convertible_bond"
    master = pl.read_parquet(root / "master.parquet")
    daily = pl.read_parquet(root / "daily.parquet")
    panel = prepare_panel(daily, master)
    schedule = weekly_schedule(panel)
    candidates = build_candidates(panel, schedule)
    action_dates = schedule["entry_date"].to_list()
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
        "schema_version": "p0-convertible-bond-double-low-development-v1",
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
            "lot_size": LOT_SIZE,
            "minimum_listing_days": MIN_LISTING_DAYS,
            "minimum_maturity_days": MIN_MATURITY_DAYS,
            "liquidity_days": LIQUIDITY_DAYS,
            "minimum_mean_amount_cny": MIN_MEAN_AMOUNT,
            "price_range": [MIN_PRICE, MAX_PRICE],
            "conversion_premium_range": [
                MIN_CONVERSION_PREMIUM,
                MAX_CONVERSION_PREMIUM,
            ],
            "minimum_bond_value": MIN_BOND_VALUE,
            "commission_pct": baseline.COMMISSION_PCT,
            "slippage_pct": baseline.SLIPPAGE_PCT,
            "stamp_tax_pct": 0.0,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "execution": "weekly signal close, next trade day open",
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
            "/app/data/research/p0_convertible_bond_double_low_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
