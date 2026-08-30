"""Run the frozen development-only standardized earnings-surprise drift study."""
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

import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MAX_ANNOUNCEMENT_DELAY_DAYS = 180
SUE_HISTORY_QUARTERS = 8
MIN_SUE_HISTORY = 6
MIN_SUE = 2.0
MAX_SUE = 10.0
MIN_EPS_INNOVATION = 0.02
MAX_DEBT_RATIO = 80.0
MIN_SIGNAL_AMOUNT = 50_000_000.0
HOLDING_TRADING_DAYS = 60
TARGET_POSITIONS = 10


def load_metrics(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "financials" / "metrics" / "part.parquet"
    if not path.is_file():
        raise ValueError("financial metrics history is required")
    needed = (
        "symbol",
        "period_end",
        "announce_date",
        "eps_basic",
        "revenue_yoy",
        "roe",
        "debt_to_asset_ratio",
        "operating_cash_to_revenue",
    )
    frame = pl.read_parquet(path)
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f"financial metrics missing columns: {sorted(missing)}")
    return (
        frame.select(needed)
        .with_columns(
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("report_period_end"),
            pl.col("announce_date")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("report_announce_date"),
        )
        .drop("period_end", "announce_date")
        .filter(
            pl.col("report_period_end").is_not_null()
            & pl.col("report_announce_date").is_not_null()
            & pl.col("report_period_end").dt.month().is_in([3, 6, 9, 12])
            & (
                pl.col("report_announce_date")
                > pl.col("report_period_end")
            )
            & (
                (pl.col("report_announce_date") - pl.col("report_period_end"))
                .dt.total_days()
                <= MAX_ANNOUNCEMENT_DELAY_DAYS
            )
            & (pl.col("report_announce_date") <= DEVELOPMENT_END)
        )
        .sort(["symbol", "report_period_end"])
        .unique(subset=["symbol", "report_period_end"], keep="first")
    )


def compute_sue(metrics: pl.DataFrame) -> pl.DataFrame:
    ordered = (
        metrics.sort(["symbol", "report_period_end"])
        .with_columns(
            pl.col("report_period_end").dt.year().alias("report_year"),
            (pl.col("report_period_end").dt.month() // 3).alias(
                "report_quarter"
            ),
            pl.col("eps_basic").shift(1).over("symbol").alias("_prior_cum_eps"),
            pl.col("report_period_end")
            .shift(1)
            .over("symbol")
            .alias("_prior_period"),
            pl.col("report_announce_date")
            .shift(1)
            .over("symbol")
            .alias("_prior_announce"),
        )
        .with_columns(
            pl.when(pl.col("report_quarter") == 1)
            .then(pl.col("eps_basic"))
            .when(
                (pl.col("_prior_period").dt.year() == pl.col("report_year"))
                & (
                    (pl.col("_prior_period").dt.month() // 3)
                    == pl.col("report_quarter") - 1
                )
            )
            .then(pl.col("eps_basic") - pl.col("_prior_cum_eps"))
            .otherwise(None)
            .alias("quarter_eps")
        )
        .with_columns(
            pl.col("quarter_eps").shift(4).over("symbol").alias("_eps_lag4"),
            pl.col("report_period_end")
            .shift(4)
            .over("symbol")
            .alias("_period_lag4"),
        )
        .with_columns(
            pl.when(
                (pl.col("_period_lag4").dt.year() == pl.col("report_year") - 1)
                & (
                    (pl.col("_period_lag4").dt.month() // 3)
                    == pl.col("report_quarter")
                )
            )
            .then(pl.col("quarter_eps") - pl.col("_eps_lag4"))
            .otherwise(None)
            .alias("eps_innovation")
        )
        .with_columns(
            pl.col("eps_innovation")
            .shift(1)
            .rolling_std(
                window_size=SUE_HISTORY_QUARTERS,
                min_samples=MIN_SUE_HISTORY,
            )
            .over("symbol")
            .alias("historical_innovation_std")
        )
        .with_columns(
            (pl.col("eps_innovation") / pl.col("historical_innovation_std"))
            .alias("sue")
        )
    )
    return ordered.filter(
        pl.col("_prior_announce").is_null()
        | (pl.col("report_announce_date") > pl.col("_prior_announce"))
    )


def build_candidates(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    all_dates: list[date],
) -> pl.DataFrame:
    calendar = pl.DataFrame({"entry_date": all_dates}).with_row_index(
        "action_index"
    )
    event_quotes = (
        events.sort(["symbol", "report_announce_date"])
        .join_asof(
            panel.select(
                "symbol", "date", "raw_close", "amount", "market_cap"
            ).sort(["symbol", "date"]),
            left_on="report_announce_date",
            right_on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .rename({"date": "signal_quote_date"})
        .with_columns(
            (pl.col("report_announce_date") + pl.duration(days=1)).alias(
                "available_after"
            )
        )
        .sort("available_after")
        .join_asof(
            calendar.sort("entry_date"),
            left_on="available_after",
            right_on="entry_date",
            strategy="forward",
        )
        .drop_nulls("entry_date")
        .filter(
            pl.col("sue").is_between(MIN_SUE, MAX_SUE, closed="both")
            & (pl.col("quarter_eps") > 0)
            & (pl.col("eps_innovation") >= MIN_EPS_INNOVATION)
            & (pl.col("revenue_yoy") > 0)
            & (pl.col("roe") > 0)
            & (pl.col("operating_cash_to_revenue") > 0)
            & (pl.col("debt_to_asset_ratio") <= MAX_DEBT_RATIO)
            & (pl.col("amount") >= MIN_SIGNAL_AMOUNT)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
    )
    expanded = (
        event_quotes.with_columns(
            pl.int_ranges(
                pl.col("action_index"),
                pl.min_horizontal(
                    pl.col("action_index") + HOLDING_TRADING_DAYS,
                    pl.lit(len(all_dates)),
                ),
            ).alias("_active_indices")
        )
        .explode("_active_indices")
        .drop("entry_date", "action_index")
        .join(
            calendar.rename({"action_index": "_active_indices"}),
            on="_active_indices",
            how="inner",
        )
        .sort(
            [
                "entry_date",
                "symbol",
                "sue",
                "eps_innovation",
                "report_announce_date",
            ],
            descending=[False, False, True, True, True],
        )
        .unique(subset=["entry_date", "symbol"], keep="first")
        .sort(
            ["entry_date", "sue", "eps_innovation", "market_cap", "symbol"],
            descending=[False, True, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over("entry_date")
            .alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
    )
    return expanded.select(
        pl.col("report_announce_date").alias("date"),
        "entry_date",
        "symbol",
        "report_period_end",
        "quarter_eps",
        "eps_innovation",
        "historical_innovation_std",
        "sue",
        "revenue_yoy",
        "roe",
        "operating_cash_to_revenue",
        "debt_to_asset_ratio",
        "market_cap",
        pl.col("amount").alias("signal_amount"),
        "cap_rank",
    ).sort(["entry_date", "cap_rank", "symbol"])


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
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=shared.INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=all_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=shared.INITIAL_CASH
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            account_daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        value = baseline._compound(values)
        positive_years += int(value is not None and value > 0)
        yearly.append({"year": year, "account_return": value})
    return {
        "metrics": {
            "trading_days": account_daily.height,
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, account_daily),
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
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    events = compute_sue(load_metrics(data_dir)).filter(
        pl.col("report_announce_date").is_between(
            DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
        )
    )
    candidates = build_candidates(events, panel, all_dates)
    benchmark = shared.benchmark_metrics(panel)
    del panel, events
    gc.collect()
    result = simulate(candidates, raw_source, all_dates, data_dir)
    decision = evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-earnings-surprise-drift-development-v1",
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
            "sue_range": [MIN_SUE, MAX_SUE],
            "sue_history_quarters": SUE_HISTORY_QUARTERS,
            "minimum_sue_history": MIN_SUE_HISTORY,
            "minimum_eps_innovation": MIN_EPS_INNOVATION,
            "holding_trading_days": HOLDING_TRADING_DAYS,
            "minimum_signal_amount_cny": MIN_SIGNAL_AMOUNT,
            "execution": "first trading-day open after announcement",
            "benchmark": "PIT eligible all-A equal-weight daily return",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "signal_rows": candidates.height,
            "active_days": candidates.get_column("entry_date").n_unique(),
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
            "/app/data/research/p0_earnings_surprise_drift_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
