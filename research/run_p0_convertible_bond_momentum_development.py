"""Run the frozen development-only convertible-bond momentum study."""

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

import run_p0_convertible_bond_double_low_development as cbbase  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2017, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
ACCOUNT_SIZES = {
    "cny_200k": 200_000.0,
    "cny_300k": 300_000.0,
    "cny_500k": 500_000.0,
    "cny_1m": 1_000_000.0,
}
TARGET_POSITIONS = 5
LOOKBACK_DAYS = 20
MIN_LISTING_DAYS = 30
MIN_MATURITY_DAYS = 365
MIN_MEAN_AMOUNT = 30_000_000.0
MIN_PRICE = 80.0
MAX_PRICE = 150.0
MIN_CONVERSION_PREMIUM = -10.0
MAX_CONVERSION_PREMIUM = 80.0
ORDINARY_CB_PREFIXES = ["110", "113", "123", "127", "128"]


def prepare_panel(daily: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    return (
        daily.join(
            master.select("symbol", "list_date", "maturity_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            pl.col("symbol").str.slice(0, 3).is_in(ORDINARY_CB_PREFIXES)
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
            .rolling_mean(window_size=LOOKBACK_DAYS, min_samples=LOOKBACK_DAYS)
            .over("symbol")
            .alias("mean_amount_20d"),
            pl.col("_global_index")
            .shift(1)
            .over("symbol")
            .alias("_prev_index"),
            pl.col("close").shift(1).over("symbol").alias("_prev_close"),
            pl.col("_global_index")
            .shift(LOOKBACK_DAYS)
            .over("symbol")
            .alias("_lookback_index"),
            pl.col("close")
            .shift(LOOKBACK_DAYS)
            .over("symbol")
            .alias("_lookback_close"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_prev_index") + 1)
            .then(pl.col("close") / pl.col("_prev_close") - 1.0)
            .otherwise(None)
            .alias("daily_return"),
            pl.when(
                pl.col("_global_index")
                == pl.col("_lookback_index") + LOOKBACK_DAYS
            )
            .then(pl.col("close") / pl.col("_lookback_close") - 1.0)
            .otherwise(None)
            .alias("momentum_20d"),
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
            & (pl.col("momentum_20d") > 0)
            & (pl.col("volume") > 0)
            & (pl.col("amount") > 0)
        )
        .sort(
            ["date", "momentum_20d", "mean_amount_20d", "symbol"],
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
            pl.col("momentum_20d").alias("factor_value"),
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate_gate(
    accounts: dict[str, dict[str, Any]], benchmark: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    benchmark_annualized = benchmark.get("annualized")
    for name, result in accounts.items():
        metrics = result["metrics"]
        annualized = metrics.get("annualized")
        excess = (
            annualized - benchmark_annualized
            if annualized is not None and benchmark_annualized is not None
            else -math.inf
        )
        checks.update(
            {
                f"{name}_annualized_at_least_50pct": (
                    annualized or -math.inf
                )
                >= 0.50,
                f"{name}_excess_at_least_20pp": excess >= 0.20,
                f"{name}_drawdown_no_worse_than_30pct": (
                    metrics.get("max_drawdown") or -math.inf
                )
                >= -0.30,
                f"{name}_at_least_three_positive_years": int(
                    metrics.get("positive_years") or 0
                )
                >= 3,
                f"{name}_buy_execution_at_least_90pct": result[
                    "execution"
                ]["buy"]["execution_rate"]
                >= 0.90,
                f"{name}_sell_execution_at_least_90pct": result[
                    "execution"
                ]["sell"]["execution_rate"]
                >= 0.90,
                f"{name}_no_unresolved_positions": result["integrity"][
                    "ending_unresolved_positions"
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
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "convertible_bond"
    master = pl.read_parquet(root / "master.parquet")
    daily = pl.read_parquet(root / "daily.parquet")
    panel = prepare_panel(daily, master)
    schedule = cbbase.weekly_schedule(panel)
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
        name: cbbase.simulate(
            candidates, panel, all_dates, action_dates, initial_cash
        )
        for name, initial_cash in ACCOUNT_SIZES.items()
    }
    benchmark = cbbase.benchmark_metrics(panel)
    decision = evaluate_gate(accounts, benchmark)
    payload = {
        "schema_version": "p0-convertible-bond-momentum-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "account_sizes_cny": ACCOUNT_SIZES,
            "target_positions": TARGET_POSITIONS,
            "lookback_days": LOOKBACK_DAYS,
            "minimum_listing_days": MIN_LISTING_DAYS,
            "minimum_maturity_days": MIN_MATURITY_DAYS,
            "minimum_mean_amount_cny": MIN_MEAN_AMOUNT,
            "ordinary_cb_prefixes": ORDINARY_CB_PREFIXES,
            "price_range": [MIN_PRICE, MAX_PRICE],
            "conversion_premium_range": [
                MIN_CONVERSION_PREMIUM,
                MAX_CONVERSION_PREMIUM,
            ],
            "commission_pct": baseline.COMMISSION_PCT,
            "slippage_pct": baseline.SLIPPAGE_PCT,
            "stamp_tax_pct": 0.0,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "execution": "weekly signal close, next trading open",
        },
        "data": {
            "ordinary_cb_symbols": panel["symbol"].n_unique(),
            "candidate_rows": candidates.height,
            "candidate_symbols": candidates["symbol"].n_unique(),
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
            "/app/data/research/p0_convertible_bond_momentum_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
