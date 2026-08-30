"""Run the frozen development-only market-up daily momentum study."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
MIN_DAILY_RETURN = 0.02
MAX_DAILY_RETURN = 0.09
MIN_MEAN_AMOUNT_20D = 100_000_000.0
TARGET_POSITIONS = 10


def attach_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    featured = (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d")
        )
    )
    context = featured.group_by("date").agg(
        pl.col("daily_return")
        .filter(pl.col("daily_return").is_finite())
        .mean()
        .alias("market_return"),
        pl.col("market_cap")
        .filter(pl.col("market_cap").is_finite())
        .median()
        .alias("median_market_cap"),
    )
    return featured.join(context, on="date", how="left")


def build_candidates(panel: pl.DataFrame) -> pl.DataFrame:
    calendar = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("entry_date"))
        .drop_nulls("entry_date")
    )
    return (
        panel.join(calendar, on="date", how="inner")
        .filter(
            (pl.col("market_return") > 0)
            & pl.col("daily_return").is_between(
                MIN_DAILY_RETURN, MAX_DAILY_RETURN, closed="both"
            )
            & (pl.col("market_cap") >= pl.col("median_market_cap"))
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .sort(
            ["date", "daily_return", "market_cap", "amount", "symbol"],
            descending=[False, True, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "daily_return",
            "market_return",
            "market_cap",
            "median_market_cap",
            "mean_amount_20d",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


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
        .with_columns(
            (pl.col("quote_date") == pl.col("entry_date")).alias(
                "exact_quote"
            )
        )
        .sort(["entry_date", "symbol"])
    )


def simulate(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    data_dir: Path,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    action_dates = all_dates[1:]
    grid = build_action_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=shared.INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=shared.INITIAL_CASH
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
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
        "action_days": len(action_dates),
        "active_signal_days": candidates.get_column("entry_date").n_unique(),
    }


def evaluate_gate(
    result: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    decision = shared.evaluate_gate(result, benchmark)
    decision["checks"].pop("mean_cash_ratio_at_most_25pct")
    decision["passed"] = all(decision["checks"].values())
    decision["verdict"] = (
        "PROMOTE_TO_VALIDATION" if decision["passed"] else "TERMINATE"
    )
    return decision


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START
    )
    if raw_source.is_empty():
        raise ValueError("no development daily data")
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = attach_daily_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    candidates = build_candidates(panel)
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()
    result = simulate(candidates, raw_source, all_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-daily-momentum-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "initial_cash_cny": shared.INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "daily_return_range": [MIN_DAILY_RETURN, MAX_DAILY_RETURN],
            "market_state": "PIT eligible all-A equal-weight daily return > 0",
            "minimum_market_cap": "daily PIT eligible median",
            "minimum_mean_amount_20d_cny": MIN_MEAN_AMOUNT_20D,
            "execution": "signal close; next open buy; following open sell/rotate",
            "benchmark": "PIT eligible all-A equal-weight daily return",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "signal_rows": candidates.height,
            "active_signal_days": candidates.get_column("entry_date").n_unique(),
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
                    "action_days": result["action_days"],
                    "active_signal_days": result["active_signal_days"],
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
            "/app/data/research/p0_daily_momentum_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
