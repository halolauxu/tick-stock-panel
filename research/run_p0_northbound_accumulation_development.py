"""Run the frozen development-only northbound accumulation account study."""

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

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

WARMUP_START = date(2017, 1, 1)
DEVELOPMENT_START = date(2017, 3, 17)
DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CASH_TIERS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
DISCLOSURE_LAG_TRADING_DAYS = 2
MIN_MEAN_AMOUNT_20D = 20_000_000.0
MIN_FULL_POSITIVE_YEARS = 2


def load_holdings(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "research" / "northbound_weekly_holdings.parquet"
    if not path.is_file():
        raise ValueError("qualified northbound weekly holdings are required")
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("holding_shares").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            & (pl.col("holding_shares") >= 0)
        )
        .select("date", "symbol", "holding_shares")
        .unique(subset=["date", "symbol"], keep="last")
        .sort(["date", "symbol"])
    )


def attach_signal_features(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.sort(["symbol", "date"]).with_columns(
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d")
    )


def build_candidates(
    holdings: pl.DataFrame,
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, list[date], dict[str, Any]]:
    snapshot_calendar = (
        holdings.select("date").unique().sort("date").with_row_index("snapshot_index")
    )
    trading_calendar = (
        panel.select("date", "_global_index")
        .unique()
        .sort("date")
        .select(
            pl.col("_global_index").alias("entry_index"),
            pl.col("date").alias("entry_date"),
        )
    )
    signal_fields = panel.select(
        "symbol",
        "date",
        "_global_index",
        "total_shares",
        "raw_close",
        "mean_amount_20d",
    )
    work = (
        holdings.join(snapshot_calendar, on="date", how="left")
        .join(signal_fields, on=["symbol", "date"], how="inner")
        .filter(
            (pl.col("total_shares") > 0)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
        )
        .with_columns(
            (pl.col("holding_shares") / pl.col("total_shares") * 100.0).alias(
                "holding_ratio_pct_recomputed"
            )
        )
        .sort(["symbol", "snapshot_index"])
        .with_columns(
            pl.col("holding_ratio_pct_recomputed")
            .shift(1)
            .over("symbol")
            .alias("previous_holding_ratio_pct"),
            pl.col("snapshot_index").shift(1).over("symbol").alias("previous_snapshot_index"),
        )
        .with_columns(
            (pl.col("holding_ratio_pct_recomputed") - pl.col("previous_holding_ratio_pct")).alias(
                "holding_ratio_delta_pct"
            ),
            (pl.col("_global_index") + DISCLOSURE_LAG_TRADING_DAYS).alias("entry_index"),
        )
        .join(trading_calendar, on="entry_index", how="inner")
    )
    eligible = work.filter(
        (pl.col("previous_snapshot_index") == pl.col("snapshot_index") - 1)
        & (pl.col("holding_ratio_delta_pct") > 0)
    )
    candidates = (
        eligible.sort(
            ["date", "holding_ratio_delta_pct", "mean_amount_20d", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "holding_shares",
            "holding_ratio_pct_recomputed",
            "previous_holding_ratio_pct",
            "holding_ratio_delta_pct",
            "mean_amount_20d",
            pl.col("mean_amount_20d").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )
    action_dates = (
        snapshot_calendar.join(
            panel.select("date", "_global_index").unique(),
            on="date",
            how="inner",
        )
        .with_columns((pl.col("_global_index") + DISCLOSURE_LAG_TRADING_DAYS).alias("entry_index"))
        .join(trading_calendar, on="entry_index", how="inner")
        .get_column("entry_date")
        .sort()
        .to_list()
    )
    audit = {
        "snapshot_count": snapshot_calendar.height,
        "joined_rows": work.height,
        "adjacent_positive_rows": eligible.height,
        "candidate_rows": candidates.height,
        "active_signal_weeks": candidates.get_column("date").n_unique(),
        "candidate_symbols": candidates.get_column("symbol").n_unique(),
    }
    return candidates, action_dates, audit


def build_action_grid(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    action_dates: list[date],
) -> pl.DataFrame:
    symbols = candidates.select("symbol").unique().sort("symbol")
    actions = pl.DataFrame({"entry_date": action_dates})
    grid = symbols.join(actions, how="cross").sort(["symbol", "entry_date"])
    quote_history = quotes.rename(
        {
            "date": "quote_date",
            "amount": "entry_amount",
            "volume": "entry_volume",
        }
    ).sort(["symbol", "quote_date"])
    return (
        grid.join_asof(
            quote_history,
            left_on="entry_date",
            right_on="quote_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns((pl.col("quote_date") == pl.col("entry_date")).alias("exact_quote"))
        .sort(["entry_date", "symbol"])
    )


def _annualized(returns: list[float]) -> float | None:
    total = baseline._compound(returns)
    if not returns or total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(returns)) - 1.0


def benchmark_metrics(panel: pl.DataFrame) -> dict[str, Any]:
    daily = (
        panel.filter((pl.col("date") >= DEVELOPMENT_START) & pl.col("daily_return").is_finite())
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


def simulate_tier(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    grid: pl.DataFrame,
    action_dates: list[date],
    all_dates: list[date],
    initial_cash: float,
) -> dict[str, Any]:
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_full_years = 0
    for year in range(2018, DEVELOPMENT_END.year + 1):
        values = (
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_full_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
            "trading_days": daily.height,
            "annualized": _annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_full_years": positive_full_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        },
        "account": account.account_summary(simulation, daily),
        "worst_weeks": account.worst_weeks(daily),
    }


def evaluate_gate(account_200k: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = account_200k["metrics"]
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
        "at_least_two_positive_full_years": metrics.get("positive_full_years", 0)
        >= MIN_FULL_POSITIVE_YEARS,
        "buy_execution_at_least_90pct": account_200k["execution"]["buy"]["execution_rate"] >= 0.90,
        "sell_execution_at_least_90pct": account_200k["execution"]["sell"]["execution_rate"]
        >= 0.90,
        "ending_positions_resolved": account_200k["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": account_200k["integrity"]["max_cash_reconciliation_error"] <= 0.01,
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
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= WARMUP_START
    )
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
    panel = attach_signal_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    holdings = load_holdings(data_dir)
    candidates, action_dates, signal_audit = build_candidates(holdings, panel)
    benchmark = benchmark_metrics(panel)
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del holdings, panel
    gc.collect()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(candidate_symbols)),
            data_dir,
        )
    )
    del raw_source
    gc.collect()
    grid = build_action_grid(candidates, quotes, action_dates)
    accounts = {
        str(int(initial_cash)): simulate_tier(
            candidates,
            quotes,
            grid,
            action_dates,
            all_dates,
            initial_cash,
        )
        for initial_cash in INITIAL_CASH_TIERS
    }
    decision = evaluate_gate(accounts["200000"], benchmark)
    payload = {
        "schema_version": "p0-northbound-accumulation-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "cash_tiers_cny": list(INITIAL_CASH_TIERS),
            "target_positions": TARGET_POSITIONS,
            "signal": "positive weekly change in PIT recomputed holding ratio",
            "disclosure_lag_trading_days": DISCLOSURE_LAG_TRADING_DAYS,
            "minimum_mean_amount_20d_cny": MIN_MEAN_AMOUNT_20D,
            "execution": "second A-share trading-day open after disclosure; weekly rotation",
            "benchmark": "PIT eligible all-A equal-weight daily return",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            **signal_audit,
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
                    key: {
                        "metrics": value["metrics"],
                        "execution": value["execution"],
                        "integrity": value["integrity"],
                        "account": value["account"],
                    }
                    for key, value in accounts.items()
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
        default=Path("/app/data/research/p0_northbound_accumulation_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
