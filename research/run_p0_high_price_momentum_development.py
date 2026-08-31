"""Run the frozen development-only high-nominal-price momentum study."""

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

DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
MIN_NOMINAL_PRICE = 3.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
FORMATION_MONTH_ENDS = 12
FULL_YEARS = tuple(range(2015, 2021))


def attach_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.sort(["symbol", "date"]).with_columns(
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d")
    )


def build_monthly_signal_panel(
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, list[date]]:
    calendar = (
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
        .with_row_index("month_index")
    )
    monthly = (
        panel.join(
            calendar.select("signal_date", "entry_date", "month_index"),
            left_on="date",
            right_on="signal_date",
            how="inner",
        )
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("month_index")
            .shift(FORMATION_MONTH_ENDS - 1)
            .over("symbol")
            .alias("formation_month_index"),
            pl.col("close")
            .shift(FORMATION_MONTH_ENDS - 1)
            .over("symbol")
            .alias("formation_close"),
        )
        .with_columns(
            pl.when(
                pl.col("month_index")
                == pl.col("formation_month_index") + FORMATION_MONTH_ENDS - 1
            )
            .then(pl.col("close") / pl.col("formation_close") - 1.0)
            .otherwise(None)
            .alias("momentum_12m")
        )
    )
    return monthly, calendar.get_column("entry_date").to_list()


def rank_monthly_universe(monthly: pl.DataFrame) -> pl.DataFrame:
    broad = (
        monthly.filter(pl.col("market_cap") > 0)
        .with_columns(
            pl.len().over("date").alias("broad_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("market_cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("market_cap_rank") - 1) * 10 / pl.col("broad_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("market_cap_decile")
        )
    )
    liquid = broad.filter(
        (pl.col("market_cap_decile") > 0)
        & (pl.col("raw_close") >= MIN_NOMINAL_PRICE)
        & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
        & pl.col("momentum_12m").is_finite()
    )
    return liquid.with_columns(
        pl.len().over("date").alias("eligible_count"),
        pl.col("raw_close").rank(method="ordinal").over("date").alias("price_rank"),
        pl.col("momentum_12m")
        .rank(method="ordinal")
        .over("date")
        .alias("momentum_rank"),
    ).with_columns(
        (
            ((pl.col("price_rank") - 1) * 10 / pl.col("eligible_count"))
            .floor()
            .clip(0, 9)
            .cast(pl.UInt8)
        ).alias("price_decile"),
        (
            ((pl.col("momentum_rank") - 1) * 5 / pl.col("eligible_count"))
            .floor()
            .clip(0, 4)
            .cast(pl.UInt8)
        ).alias("momentum_quintile"),
    )


def build_candidates(ranked: pl.DataFrame, *, require_high_price: bool) -> pl.DataFrame:
    selected = ranked.filter(pl.col("momentum_quintile") == 4)
    if require_high_price:
        selected = selected.filter(pl.col("price_decile") == 9)
    return (
        selected.sort(
            ["date", "momentum_12m", "raw_close", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"))
        .select(
            "date",
            "entry_date",
            "symbol",
            "momentum_12m",
            "raw_close",
            "market_cap",
            "market_cap_decile",
            "price_decile",
            "momentum_quintile",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _yearly_returns(daily_equity: pl.DataFrame) -> tuple[list[dict[str, Any]], int]:
    rows = []
    positive = 0
    for year in FULL_YEARS:
        returns = (
            daily_equity.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        value = baseline._compound(returns)
        positive += int(value is not None and value > 0)
        rows.append({"year": year, "account_return": value})
    return rows, positive


def _profit_concentration(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_by_symbol: dict[str, float] = {}
    for trade in trades:
        symbol = str(trade["symbol"])
        pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0.0) + float(
            trade.get("cash_delta") or 0.0
        )
    positive = {symbol: pnl for symbol, pnl in pnl_by_symbol.items() if pnl > 0}
    total_positive = sum(positive.values())
    largest_symbol = max(positive, key=positive.get) if positive else None
    largest_share = (
        positive[largest_symbol] / total_positive
        if largest_symbol is not None and total_positive > 0
        else None
    )
    return {
        "total_positive_symbol_pnl": total_positive,
        "largest_positive_symbol": largest_symbol,
        "largest_positive_symbol_pnl": (
            positive.get(largest_symbol) if largest_symbol is not None else None
        ),
        "largest_positive_symbol_share": largest_share,
    }


def prepare_variant_execution(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    data_dir: Path,
    action_dates: list[date],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, action_dates)
    return quotes, grid


def simulate_variant(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    grid: pl.DataFrame,
    all_dates: list[date],
    action_dates: list[date],
    initial_cash: float,
) -> dict[str, Any]:
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
    )
    daily_equity, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = daily_equity.get_column("daily_return").drop_nulls().to_list()
    yearly, positive_years = _yearly_returns(daily_equity)
    execution = account.execution_summary(simulation["orders"])
    ending_open_positions = len(simulation["ending_positions"])
    return {
        "metrics": {
            "trading_days": daily_equity.height,
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_full_years": positive_years,
            "yearly": yearly,
        },
        "execution": execution,
        "integrity": {
            **stale,
            "ending_open_positions": ending_open_positions,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily_equity),
        "completed_trades": execution["sell"]["filled"],
        "profit_concentration": _profit_concentration(simulation["trades"]),
    }


def benchmark_metrics(panel: pl.DataFrame, start: date) -> dict[str, Any]:
    daily_returns = (
        panel.filter(
            (pl.col("date") >= pl.lit(start)) & pl.col("daily_return").is_finite()
        )
        .group_by("date")
        .agg(pl.col("daily_return").mean().alias("return"))
        .sort("date")
        .get_column("return")
        .to_list()
    )
    return {
        "annualized": shared._annualized(daily_returns),
        "total_return": baseline._compound(daily_returns),
        "max_drawdown": baseline._max_drawdown(daily_returns),
    }


def evaluate_gate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    candidate_annualized = metrics.get("annualized")
    control_annualized = control["metrics"].get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    concentration = candidate["profit_concentration"].get(
        "largest_positive_symbol_share"
    )
    checks = {
        "annualized_at_least_50pct": (
            candidate_annualized is not None and candidate_annualized >= 0.50
        ),
        "market_excess_at_least_20pp": (
            candidate_annualized is not None
            and benchmark_annualized is not None
            and candidate_annualized - benchmark_annualized >= 0.20
        ),
        "plain_momentum_increment_at_least_10pp": (
            candidate_annualized is not None
            and control_annualized is not None
            and candidate_annualized - control_annualized >= 0.10
        ),
        "max_drawdown_not_worse_than_30pct": (
            metrics.get("max_drawdown") is not None and metrics["max_drawdown"] >= -0.30
        ),
        "at_least_five_positive_full_years": (
            metrics.get("positive_full_years", 0) >= 5
        ),
        "buy_execution_at_least_90pct": (
            candidate["execution"]["buy"]["execution_rate"] >= 0.90
        ),
        "sell_execution_at_least_90pct": (
            candidate["execution"]["sell"]["execution_rate"] >= 0.90
        ),
        "no_ending_open_positions": (
            candidate["integrity"]["ending_open_positions"] == 0
        ),
        "cash_reconciles": (
            candidate["integrity"]["max_cash_reconciliation_error"] <= 0.01
        ),
        "at_least_300_completed_trades": candidate["completed_trades"] >= 300,
        "largest_positive_symbol_share_at_most_25pct": (
            concentration is not None and concentration <= 0.25
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "counts_toward_50pct_goal": False,
        "next_step": (
            "freeze_independent_validation"
            if passed
            else "terminate_high_price_momentum"
        ),
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
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = attach_daily_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()

    monthly, monthly_entries = build_monthly_signal_panel(panel)
    ranked = rank_monthly_universe(monthly)
    high_price_candidates = build_candidates(ranked, require_high_price=True)
    plain_candidates = build_candidates(ranked, require_high_price=False)
    del monthly, ranked
    gc.collect()
    if high_price_candidates.is_empty():
        raise ValueError("no high-price momentum candidates")

    first_action = max(
        high_price_candidates.get_column("entry_date").min(),
        plain_candidates.get_column("entry_date").min(),
    )
    all_dates = [
        value
        for value in panel.get_column("date").unique().sort().to_list()
        if first_action <= value <= DEVELOPMENT_END
    ]
    action_dates = sorted(
        {value for value in monthly_entries if first_action <= value <= DEVELOPMENT_END}
        | {DEVELOPMENT_END}
    )
    benchmark = benchmark_metrics(panel, first_action)
    del panel
    gc.collect()

    high_price_quotes, high_price_grid = prepare_variant_execution(
        high_price_candidates, raw_source, data_dir, action_dates
    )
    plain_quotes, plain_grid = prepare_variant_execution(
        plain_candidates, raw_source, data_dir, action_dates
    )
    del raw_source
    gc.collect()

    results: dict[str, Any] = {}
    for initial_cash in INITIAL_CAPITALS:
        key = str(int(initial_cash))
        high_price = simulate_variant(
            high_price_candidates,
            high_price_quotes,
            high_price_grid,
            all_dates,
            action_dates,
            initial_cash,
        )
        plain = simulate_variant(
            plain_candidates,
            plain_quotes,
            plain_grid,
            all_dates,
            action_dates,
            initial_cash,
        )
        results[key] = {
            "high_price_momentum": high_price,
            "plain_momentum_control": plain,
        }

    decision = evaluate_gate(
        results["200000"]["high_price_momentum"],
        results["200000"]["plain_momentum_control"],
        benchmark,
    )
    payload = {
        "schema_version": "p0-high-price-momentum-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": first_action,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "formation_month_ends": FORMATION_MONTH_ENDS,
            "minimum_nominal_price": MIN_NOMINAL_PRICE,
            "minimum_mean_amount_20d": MIN_MEAN_AMOUNT_20D,
            "excluded_market_cap_decile": 0,
            "selected_price_decile": 9,
            "selected_momentum_quintile": 4,
            "target_positions": TARGET_POSITIONS,
            "execution": "month-end signal, next trading open, final-date liquidation attempt",
        },
        "data": {
            "high_price_signal_rows": high_price_candidates.height,
            "high_price_symbols": high_price_candidates.get_column("symbol").n_unique(),
            "plain_signal_rows": plain_candidates.height,
            "plain_symbols": plain_candidates.get_column("symbol").n_unique(),
            "action_dates": len(action_dates),
        },
        "benchmark": benchmark,
        "capital_results": results,
        "decision": decision,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        capital: {
            variant: {
                "metrics": row[variant]["metrics"],
                "execution": row[variant]["execution"],
                "integrity": row[variant]["integrity"],
                "account": row[variant]["account"],
                "completed_trades": row[variant]["completed_trades"],
                "profit_concentration": row[variant]["profit_concentration"],
            }
            for variant in ("high_price_momentum", "plain_momentum_control")
        }
        for capital, row in results.items()
    }
    print(
        json.dumps(
            {
                "data": payload["data"],
                "benchmark": benchmark,
                "capital_results": summary,
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
        default=Path("/app/data/research/p0_high_price_momentum_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
