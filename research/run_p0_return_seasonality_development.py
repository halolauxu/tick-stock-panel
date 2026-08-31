"""Run the frozen development-only same-calendar-month seasonality study."""

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
import run_p0_high_price_momentum_development as high_price  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
FORMATION_YEARS = 5
MIN_OTHER_MONTH_OBSERVATIONS = 50
MIN_NOMINAL_PRICE = 3.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MIN_MARKET_CAP_DECILE = 3
FULL_YEARS = tuple(range(2014, 2021))


def attach_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.sort(["symbol", "date"]).with_columns(
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d")
    )


def monthly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%Y-%m").alias("month_key"),
        )
        .group_by("month_key", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .with_columns(
            pl.col("entry_date").dt.year().alias("target_year"),
            pl.col("entry_date").dt.month().alias("target_month"),
        )
        .sort("signal_date")
    )


def _monthly_returns(
    frame: pl.DataFrame, *, date_column: str, close_column: str
) -> pl.DataFrame:
    monthly = (
        frame.select(
            "symbol",
            pl.col(date_column).alias("month_end"),
            pl.col(close_column).cast(pl.Float64).alias("adjusted_close"),
        )
        .with_columns(
            pl.col("month_end").dt.year().alias("year"),
            pl.col("month_end").dt.month().alias("month"),
        )
        .with_columns((pl.col("year") * 12 + pl.col("month")).alias("month_index"))
    )
    return (
        monthly.sort(["symbol", "month_end"])
        .with_columns(
            pl.col("month_index").shift(1).over("symbol").alias("previous_index"),
            pl.col("adjusted_close").shift(1).over("symbol").alias("previous_close"),
        )
        .with_columns(
            pl.when(pl.col("month_index") == pl.col("previous_index") + 1)
            .then(pl.col("adjusted_close") / pl.col("previous_close") - 1.0)
            .otherwise(None)
            .alias("monthly_return")
        )
        .select("symbol", "year", "month", "monthly_return")
    )


def build_monthly_return_history(
    raw_source: pl.DataFrame, warmup: pl.DataFrame
) -> pl.DataFrame:
    current_month_ends = (
        raw_source.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_key"))
        .group_by("month_key", maintain_order=True)
        .agg(pl.col("date").max().alias("month_end"))
    )
    current = raw_source.join(
        current_month_ends, left_on="date", right_on="month_end", how="inner"
    )
    warmup_returns = _monthly_returns(
        warmup, date_column="month_end", close_column="adjusted_close"
    )
    current_returns = _monthly_returns(
        current, date_column="date", close_column="close"
    )
    return (
        pl.concat([warmup_returns, current_returns], how="vertical")
        .unique(subset=["symbol", "year", "month"], keep="last")
        .sort(["symbol", "year", "month"])
    )


def build_seasonality_scores(monthly_returns: pl.DataFrame) -> pl.DataFrame:
    valid = monthly_returns.filter(pl.col("monthly_return").is_finite())
    expanded = pl.concat(
        [
            valid.with_columns((pl.col("year") + offset).alias("target_year"))
            for offset in range(1, FORMATION_YEARS + 1)
        ]
    )
    totals = expanded.group_by("symbol", "target_year").agg(
        pl.col("monthly_return").sum().alias("five_year_sum"),
        pl.len().alias("five_year_count"),
    )
    same = expanded.group_by(
        "symbol", "target_year", pl.col("month").alias("target_month")
    ).agg(
        pl.col("monthly_return").sum().alias("same_month_sum"),
        pl.len().alias("same_month_count"),
    )
    return (
        same.join(totals, on=["symbol", "target_year"], how="inner")
        .with_columns(
            (pl.col("five_year_count") - pl.col("same_month_count")).alias(
                "other_month_count"
            )
        )
        .filter(
            (pl.col("same_month_count") == FORMATION_YEARS)
            & (pl.col("other_month_count") >= MIN_OTHER_MONTH_OBSERVATIONS)
        )
        .with_columns(
            (pl.col("same_month_sum") / pl.col("same_month_count")).alias(
                "same_month_score"
            ),
            (
                (pl.col("five_year_sum") - pl.col("same_month_sum"))
                / pl.col("other_month_count")
            ).alias("other_month_score"),
        )
        .select(
            "symbol",
            "target_year",
            "target_month",
            "same_month_score",
            "other_month_score",
            "same_month_count",
            "other_month_count",
        )
    )


def broad_market_cap_deciles(
    raw_source: pl.DataFrame, schedule: pl.DataFrame, data_dir: Path
) -> pl.DataFrame:
    universe = (
        pl.read_parquet(
            data_dir / "research" / "historical_stock_universe_all_a.parquet"
        )
        .with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "list_date", "delist_date")
    )
    shares = baseline.load_share_history(data_dir)
    broad = (
        raw_source.join(
            schedule.select("signal_date"),
            left_on="date",
            right_on="signal_date",
            how="inner",
        )
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
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(pl.col("total_shares") > 0)
        .with_columns(
            (pl.col("raw_close") * pl.col("total_shares")).alias("broad_market_cap")
        )
        .with_columns(
            pl.len().over("date").alias("broad_count"),
            pl.col("broad_market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("broad_cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("broad_cap_rank") - 1) * 10 / pl.col("broad_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("market_cap_decile")
        )
    )
    return broad.select("symbol", "date", "broad_market_cap", "market_cap_decile")


def rank_signal_universe(
    panel: pl.DataFrame,
    schedule: pl.DataFrame,
    scores: pl.DataFrame,
    cap_deciles: pl.DataFrame,
) -> pl.DataFrame:
    eligible = (
        panel.join(schedule, left_on="date", right_on="signal_date", how="inner")
        .join(
            scores,
            on=["symbol", "target_year", "target_month"],
            how="inner",
        )
        .join(cap_deciles, on=["symbol", "date"], how="inner")
        .filter(
            (pl.col("market_cap_decile") >= MIN_MARKET_CAP_DECILE)
            & (pl.col("raw_close") >= MIN_NOMINAL_PRICE)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("same_month_score").is_finite()
            & pl.col("other_month_score").is_finite()
        )
        .with_columns(
            pl.len().over("date").alias("eligible_count"),
            pl.col("same_month_score")
            .rank(method="ordinal")
            .over("date")
            .alias("same_rank"),
            pl.col("other_month_score")
            .rank(method="ordinal")
            .over("date")
            .alias("other_rank"),
        )
    )
    return eligible.with_columns(
        (
            ((pl.col("same_rank") - 1) * 10 / pl.col("eligible_count"))
            .floor()
            .clip(0, 9)
            .cast(pl.UInt8)
        ).alias("same_month_decile"),
        (
            ((pl.col("other_rank") - 1) * 10 / pl.col("eligible_count"))
            .floor()
            .clip(0, 9)
            .cast(pl.UInt8)
        ).alias("other_month_decile"),
    )


def build_candidates(ranked: pl.DataFrame, *, use_same_month: bool) -> pl.DataFrame:
    score = "same_month_score" if use_same_month else "other_month_score"
    decile = "same_month_decile" if use_same_month else "other_month_decile"
    return (
        ranked.filter(pl.col(decile) == 9)
        .sort(
            ["date", score, "amount", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"))
        .select(
            "date",
            "entry_date",
            "symbol",
            "target_year",
            "target_month",
            "same_month_score",
            "other_month_score",
            "market_cap_decile",
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
    return quotes, daily.build_action_grid(candidates, quotes, action_dates)


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
            "ending_open_positions": len(simulation["ending_positions"]),
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily_equity),
        "completed_trades": execution["sell"]["filled"],
        "profit_concentration": high_price._profit_concentration(simulation["trades"]),
    }


def benchmark_metrics(panel: pl.DataFrame, start: date) -> dict[str, Any]:
    returns = (
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
        "annualized": shared._annualized(returns),
        "total_return": baseline._compound(returns),
        "max_drawdown": baseline._max_drawdown(returns),
    }


def evaluate_gate(
    candidate: dict[str, Any], control: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    annualized = metrics.get("annualized")
    control_annualized = control["metrics"].get("annualized")
    benchmark_annualized = benchmark.get("annualized")
    concentration = candidate["profit_concentration"].get(
        "largest_positive_symbol_share"
    )
    checks = {
        "annualized_at_least_50pct": annualized is not None and annualized >= 0.50,
        "market_excess_at_least_20pp": (
            annualized is not None
            and benchmark_annualized is not None
            and annualized - benchmark_annualized >= 0.20
        ),
        "other_month_increment_at_least_10pp": (
            annualized is not None
            and control_annualized is not None
            and annualized - control_annualized >= 0.10
        ),
        "max_drawdown_not_worse_than_30pct": (
            metrics.get("max_drawdown") is not None and metrics["max_drawdown"] >= -0.30
        ),
        "at_least_five_positive_full_years": metrics.get("positive_full_years", 0) >= 5,
        "buy_execution_at_least_90pct": candidate["execution"]["buy"]["execution_rate"]
        >= 0.90,
        "sell_execution_at_least_90pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "no_ending_open_positions": candidate["integrity"]["ending_open_positions"]
        == 0,
        "cash_reconciles": candidate["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
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
            else "terminate_return_seasonality"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    warmup_path = (
        data_dir / "research" / "return_seasonality_warmup" / "monthly.parquet"
    )
    if not warmup_path.is_file():
        raise ValueError("qualified return-seasonality warmup data is required")
    warmup = pl.read_parquet(warmup_path)
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = attach_daily_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()

    schedule = monthly_schedule(panel)
    history = build_monthly_return_history(raw_source, warmup)
    scores = build_seasonality_scores(history)
    cap_deciles = broad_market_cap_deciles(raw_source, schedule, data_dir)
    ranked = rank_signal_universe(panel, schedule, scores, cap_deciles)
    same_candidates = build_candidates(ranked, use_same_month=True)
    other_candidates = build_candidates(ranked, use_same_month=False)
    del history, scores, cap_deciles, ranked, warmup
    gc.collect()
    if same_candidates.is_empty() or other_candidates.is_empty():
        raise ValueError("return seasonality candidates are empty")

    first_action = max(
        same_candidates.get_column("entry_date").min(),
        other_candidates.get_column("entry_date").min(),
    )
    all_dates = [
        value
        for value in panel.get_column("date").unique().sort().to_list()
        if first_action <= value <= DEVELOPMENT_END
    ]
    action_dates = sorted(
        {
            value
            for value in schedule.get_column("entry_date").to_list()
            if first_action <= value <= DEVELOPMENT_END
        }
        | {DEVELOPMENT_END}
    )
    benchmark = benchmark_metrics(panel, first_action)
    del panel, schedule
    gc.collect()

    same_quotes, same_grid = prepare_variant_execution(
        same_candidates, raw_source, data_dir, action_dates
    )
    other_quotes, other_grid = prepare_variant_execution(
        other_candidates, raw_source, data_dir, action_dates
    )
    del raw_source
    gc.collect()

    results: dict[str, Any] = {}
    for initial_cash in INITIAL_CAPITALS:
        key = str(int(initial_cash))
        results[key] = {
            "same_month_seasonality": simulate_variant(
                same_candidates,
                same_quotes,
                same_grid,
                all_dates,
                action_dates,
                initial_cash,
            ),
            "other_month_control": simulate_variant(
                other_candidates,
                other_quotes,
                other_grid,
                all_dates,
                action_dates,
                initial_cash,
            ),
        }

    decision = evaluate_gate(
        results["200000"]["same_month_seasonality"],
        results["200000"]["other_month_control"],
        benchmark,
    )
    payload = {
        "schema_version": "p0-return-seasonality-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": first_action,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "formation_years": FORMATION_YEARS,
            "minimum_other_month_observations": MIN_OTHER_MONTH_OBSERVATIONS,
            "minimum_market_cap_decile": MIN_MARKET_CAP_DECILE,
            "minimum_nominal_price": MIN_NOMINAL_PRICE,
            "minimum_mean_amount_20d": MIN_MEAN_AMOUNT_20D,
            "target_positions": TARGET_POSITIONS,
            "execution": "prior month-end signal, next open, monthly rebalance",
        },
        "data": {
            "same_month_signal_rows": same_candidates.height,
            "same_month_symbols": same_candidates.get_column("symbol").n_unique(),
            "other_month_signal_rows": other_candidates.height,
            "other_month_symbols": other_candidates.get_column("symbol").n_unique(),
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
    print(
        json.dumps(
            {
                "period": payload["period"],
                "data": payload["data"],
                "benchmark": benchmark,
                "capital_results": results,
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
        default=Path("/app/data/research/p0_return_seasonality_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
