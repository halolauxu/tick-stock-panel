"""Screen nine frozen academic factor mechanisms on development data only."""
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
TARGET_POSITIONS = 10
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MEAN_AMOUNT_20D = 50_000_000.0
MAX_FINANCIAL_AGE_DAYS = 550
FACTOR_IDS = (
    "high_52week",
    "low_lottery",
    "low_volatility_trend",
    "cheap_quality",
    "conservative_investment",
    "low_accrual",
    "gross_profitability",
    "cashflow_yield",
    "earnings_yield",
)


def attach_price_features(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("_global_index")
            .shift(120)
            .over("symbol")
            .alias("_index_120d"),
            pl.col("close").shift(120).over("symbol").alias("_close_120d"),
            pl.col("close")
            .rolling_max(window_size=250, min_samples=250)
            .over("symbol")
            .alias("high_250d"),
            pl.col("close")
            .rolling_mean(window_size=120, min_samples=120)
            .over("symbol")
            .alias("ma120"),
            pl.col("daily_return")
            .rolling_std(window_size=20, min_samples=20)
            .over("symbol")
            .alias("volatility_20d"),
            pl.col("daily_return")
            .rolling_max(window_size=20, min_samples=20)
            .over("symbol")
            .alias("max_daily_return_20d"),
            pl.col("amount")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("mean_amount_20d"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_index_120d") + 120)
            .then(pl.col("close") / pl.col("_close_120d") - 1.0)
            .otherwise(None)
            .alias("momentum_120d"),
            (pl.col("close") / pl.col("high_250d")).alias(
                "high_52week_proximity"
            ),
        )
    )


def _load_statement(
    data_dir: Path,
    dataset: str,
    columns: tuple[str, ...],
    announce_alias: str,
) -> pl.DataFrame:
    path = data_dir / "financials" / dataset / "part.parquet"
    if not path.is_file():
        raise ValueError(f"financial statement missing: {dataset}")
    frame = pl.read_parquet(path)
    needed = {"symbol", "period_end", "announce_date", *columns}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset} missing columns: {sorted(missing)}")
    return (
        frame.select("symbol", "period_end", "announce_date", *columns)
        .with_columns(
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("report_period_end"),
            pl.col("announce_date")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias(announce_alias),
        )
        .drop("period_end", "announce_date")
        .filter(
            (pl.col("report_period_end").dt.month() == 12)
            & pl.col("report_period_end").is_not_null()
            & pl.col(announce_alias).is_not_null()
            & (pl.col(announce_alias) > pl.col("report_period_end"))
            & (pl.col(announce_alias) <= DEVELOPMENT_END)
        )
        .sort(["symbol", "report_period_end", announce_alias])
        .unique(subset=["symbol", "report_period_end"], keep="first")
    )


def compute_annual_factors(
    income: pl.DataFrame,
    cashflow: pl.DataFrame,
    balance: pl.DataFrame,
) -> pl.DataFrame:
    joined = (
        income.join(
            cashflow, on=["symbol", "report_period_end"], how="inner"
        )
        .join(balance, on=["symbol", "report_period_end"], how="inner")
        .with_columns(
            pl.max_horizontal(
                "income_announce_date",
                "cashflow_announce_date",
                "balance_announce_date",
            ).alias("financial_available_date")
        )
        .sort(["symbol", "report_period_end"])
        .with_columns(
            pl.col("total_assets")
            .shift(1)
            .over("symbol")
            .alias("prior_total_assets"),
            pl.col("report_period_end")
            .shift(1)
            .over("symbol")
            .alias("prior_report_period"),
        )
        .with_columns(
            pl.when(
                pl.col("prior_report_period").dt.year()
                == pl.col("report_period_end").dt.year() - 1
            )
            .then(pl.col("total_assets") / pl.col("prior_total_assets") - 1.0)
            .otherwise(None)
            .alias("asset_growth"),
            (
                (pl.col("revenue") - pl.col("operating_cost"))
                / pl.col("total_assets")
            ).alias("gross_profitability"),
            (
                (pl.col("net_income_attributable") - pl.col("net_operating_cash_flow"))
                / pl.col("total_assets")
            ).alias("accrual_ratio"),
            (
                pl.col("net_income_attributable") / pl.col("total_equity")
            ).alias("roe_proxy"),
            (pl.col("total_liabilities") / pl.col("total_assets")).alias(
                "debt_ratio"
            ),
        )
    )
    return joined.filter(
        (pl.col("total_assets") > 0)
        & (pl.col("total_equity") > 0)
        & (
            (pl.col("financial_available_date") - pl.col("report_period_end"))
            .dt.total_days()
            <= 365
        )
    ).sort(["symbol", "financial_available_date"])


def load_annual_factors(data_dir: Path) -> pl.DataFrame:
    income = _load_statement(
        data_dir,
        "income",
        ("revenue", "operating_cost", "net_income_attributable"),
        "income_announce_date",
    )
    cashflow = _load_statement(
        data_dir,
        "cash_flow",
        ("net_operating_cash_flow",),
        "cashflow_announce_date",
    )
    balance = _load_statement(
        data_dir,
        "balance_sheet",
        ("total_assets", "total_liabilities", "total_equity"),
        "balance_announce_date",
    )
    return compute_annual_factors(income, cashflow, balance)


def monthly_signal_panel(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[date]]:
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
        .filter(pl.col("signal_date") >= DEVELOPMENT_START)
    )
    monthly = panel.join(
        calendar.select("signal_date", "entry_date"),
        left_on="date",
        right_on="signal_date",
        how="inner",
    )
    return monthly, calendar.get_column("entry_date").to_list()


def attach_annual_factors(
    monthly: pl.DataFrame, annual: pl.DataFrame
) -> pl.DataFrame:
    return (
        monthly.sort(["symbol", "date"])
        .join_asof(
            annual,
            left_on="date",
            right_on="financial_available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("date") - pl.col("financial_available_date"))
            .dt.total_days()
            .alias("financial_age_days"),
            (pl.col("total_equity") / pl.col("market_cap")).alias(
                "book_to_market"
            ),
            (
                pl.col("net_income_attributable") / pl.col("market_cap")
            ).alias("earnings_yield"),
            (
                pl.col("net_operating_cash_flow") / pl.col("market_cap")
            ).alias("cashflow_yield"),
        )
    )


def build_candidates(panel: pl.DataFrame, factor_id: str) -> pl.DataFrame:
    base = panel.filter(
        (pl.col("market_cap") >= MIN_MARKET_CAP)
        & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
        & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
    )
    financial_ready = pl.col("financial_age_days").is_between(
        0, MAX_FINANCIAL_AGE_DAYS, closed="both"
    )
    if factor_id == "high_52week":
        selected = base.filter(
            (pl.col("date").dt.month() != 2)
            & (pl.col("high_52week_proximity") >= 0.90)
            & (pl.col("momentum_120d") > 0)
        )
        sort_columns = ["high_52week_proximity", "momentum_120d"]
        descending = [True, True]
        value = pl.col("high_52week_proximity")
    elif factor_id == "low_lottery":
        selected = base.filter(pl.col("max_daily_return_20d").is_finite())
        sort_columns = ["max_daily_return_20d", "volatility_20d"]
        descending = [False, False]
        value = pl.col("max_daily_return_20d")
    elif factor_id == "low_volatility_trend":
        selected = base.filter(
            (pl.col("momentum_120d") > 0)
            & (pl.col("close") > pl.col("ma120"))
            & pl.col("volatility_20d").is_finite()
        )
        sort_columns = ["volatility_20d", "momentum_120d"]
        descending = [False, True]
        value = pl.col("volatility_20d")
    elif factor_id == "cheap_quality":
        selected = base.filter(
            financial_ready
            & pl.col("book_to_market").is_between(0.1, 2.0, closed="both")
            & (pl.col("earnings_yield") > 0)
            & (pl.col("roe_proxy") >= 0.10)
            & (pl.col("debt_ratio") <= 0.70)
        )
        sort_columns = ["book_to_market", "roe_proxy"]
        descending = [True, True]
        value = pl.col("book_to_market")
    elif factor_id == "conservative_investment":
        selected = base.filter(
            financial_ready
            & pl.col("asset_growth").is_between(-0.5, 0.1, closed="both")
            & (pl.col("earnings_yield") > 0)
            & (pl.col("cashflow_yield") > 0)
        )
        sort_columns = ["asset_growth", "roe_proxy"]
        descending = [False, True]
        value = pl.col("asset_growth")
    elif factor_id == "low_accrual":
        selected = base.filter(
            financial_ready
            & (pl.col("earnings_yield") > 0)
            & pl.col("accrual_ratio").is_between(-1.0, 0.1, closed="both")
        )
        sort_columns = ["accrual_ratio", "roe_proxy"]
        descending = [False, True]
        value = pl.col("accrual_ratio")
    elif factor_id == "gross_profitability":
        selected = base.filter(
            financial_ready
            & (pl.col("gross_profitability") > 0)
            & (pl.col("cashflow_yield") > 0)
        )
        sort_columns = ["gross_profitability", "roe_proxy"]
        descending = [True, True]
        value = pl.col("gross_profitability")
    elif factor_id == "cashflow_yield":
        selected = base.filter(
            financial_ready
            & pl.col("cashflow_yield").is_between(0.0, 1.0, closed="right")
        )
        sort_columns = ["cashflow_yield", "roe_proxy"]
        descending = [True, True]
        value = pl.col("cashflow_yield")
    elif factor_id == "earnings_yield":
        selected = base.filter(
            financial_ready
            & pl.col("earnings_yield").is_between(0.0, 1.0, closed="right")
        )
        sort_columns = ["earnings_yield", "roe_proxy"]
        descending = [True, True]
        value = pl.col("earnings_yield")
    else:
        raise ValueError(f"unknown factor: {factor_id}")
    ordering = ["date", *sort_columns, "market_cap", "symbol"]
    directions = [False, *descending, True, False]
    return (
        selected.sort(ordering, descending=directions)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"),
            value.alias("factor_value"),
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "factor_value",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def simulate_factor(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    action_dates: list[date],
    data_dir: Path,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=shared.INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=action_dates,
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
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
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


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    raw_source = raw_all.filter(pl.col("date") >= DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = attach_price_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= DEVELOPMENT_START)
    )
    monthly, action_dates = monthly_signal_panel(panel)
    monthly = attach_annual_factors(monthly, load_annual_factors(data_dir))
    del panel
    gc.collect()
    results = {}
    promoted = []
    for factor_id in FACTOR_IDS:
        candidates = build_candidates(monthly, factor_id)
        result = simulate_factor(
            candidates, raw_source, all_dates, action_dates, data_dir
        )
        decision = shared.evaluate_gate(result, benchmark)
        results[factor_id] = {
            "data": {
                "signal_rows": candidates.height,
                "signal_symbols": candidates.get_column("symbol").n_unique(),
                "rebalance_days": candidates.get_column("entry_date").n_unique(),
            },
            "strategy": result,
            "decision": decision,
        }
        if decision["passed"]:
            promoted.append(factor_id)
    payload = {
        "schema_version": "p0-academic-factor-development-screen-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "benchmark": benchmark,
        "factor_ids": FACTOR_IDS,
        "results": results,
        "promoted_to_independent_validation": promoted,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        factor_id: {
            "metrics": row["strategy"]["metrics"],
            "execution": row["strategy"]["execution"],
            "integrity": row["strategy"]["integrity"],
            "account": row["strategy"]["account"],
            "decision": row["decision"],
        }
        for factor_id, row in results.items()
    }
    print(
        json.dumps(
            {
                "benchmark": benchmark,
                "results": summary,
                "promoted_to_independent_validation": promoted,
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
            "/app/data/research/p0_academic_factor_development_screen.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
