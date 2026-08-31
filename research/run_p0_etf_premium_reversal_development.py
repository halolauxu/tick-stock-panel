"""Run the frozen development-only stock-ETF discount correction study."""
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
TARGET_POSITIONS = 10
MIN_LISTING_DAYS = 180
LIQUIDITY_DAYS = 20
MIN_MEAN_AMOUNT = 20_000_000.0
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
        .join(
            master.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
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
            pl.col("amount")
            .rolling_mean(window_size=LIQUIDITY_DAYS, min_samples=LIQUIDITY_DAYS)
            .over("symbol")
            .alias("_mean_amount_20d"),
            pl.col("_global_index")
            .shift(LIQUIDITY_DAYS - 1)
            .over("symbol")
            .alias("_index_19d"),
            pl.col("close").shift(1).over("symbol").alias("_prev_raw_close"),
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
            & (pl.col("entry_date") <= DEVELOPMENT_END)
        )
    )


def build_candidate_legs(
    panel: pl.DataFrame,
    nav: pl.DataFrame,
    schedule: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    scheduled = panel.join(
        schedule, left_on="date", right_on="signal_date", how="inner"
    )
    joined = scheduled.join(
        nav.select("symbol", "nav_date", "ann_date", "unit_nav"),
        left_on=["symbol", "date"],
        right_on=["symbol", "nav_date"],
        how="inner",
    )
    late_rows = joined.filter(pl.col("ann_date") >= pl.col("entry_date")).height
    eligible = (
        joined.filter(
            (pl.col("ann_date") < pl.col("entry_date"))
            & (pl.col("unit_nav") > 0)
            & (pl.col("raw_close") > 0)
            & (pl.col("listing_days") >= MIN_LISTING_DAYS)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT)
        )
        .with_columns(
            (pl.col("raw_close") / pl.col("unit_nav") - 1.0).alias("premium")
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("premium")
            .rank(method="ordinal")
            .over("date")
            .alias("premium_rank_low"),
            pl.col("premium")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("premium_rank_high"),
        )
        .with_columns(
            (pl.col("universe_count") * 0.10)
            .ceil()
            .clip(1, TARGET_POSITIONS)
            .cast(pl.UInt32)
            .alias("leg_size")
        )
    )

    def leg(rank_column: str, *, require_discount: bool) -> pl.DataFrame:
        filtered = eligible.filter(pl.col(rank_column) <= pl.col("leg_size"))
        if require_discount:
            filtered = filtered.filter(pl.col("premium") < 0)
        return (
            filtered.sort(["date", rank_column, "symbol"])
            .with_columns(pl.col(rank_column).cast(pl.UInt32).alias("cap_rank"))
            .select(
                "date",
                "entry_date",
                "symbol",
                "premium",
                "ann_date",
                "unit_nav",
                pl.col("amount").alias("signal_amount"),
                "mean_amount_20d",
                "cap_rank",
            )
            .sort(["entry_date", "cap_rank", "symbol"])
        )

    low = leg("premium_rank_low", require_discount=True)
    high = leg("premium_rank_high", require_discount=False)
    return low, high, {
        "scheduled_price_rows": scheduled.height,
        "exact_nav_join_rows": joined.height,
        "same_day_or_late_announcement_rows": late_rows,
        "eligible_rows": eligible.height,
    }


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
    low_accounts: dict[str, dict[str, Any]],
    high_accounts: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
    active_rebalances: int,
) -> dict[str, Any]:
    checks = {"at_least_100_active_rebalances": active_rebalances >= 100}
    market_annualized = benchmark.get("annualized")
    for name, result in low_accounts.items():
        metrics = result["metrics"]
        strategy_annualized = metrics.get("annualized")
        control_annualized = high_accounts[name]["metrics"].get("annualized")
        checks.update(
            {
                f"{name}_annualized_at_least_50pct": (
                    strategy_annualized or -math.inf
                )
                >= 0.50,
                f"{name}_direction_excess_at_least_20pp": (
                    strategy_annualized is not None
                    and control_annualized is not None
                    and strategy_annualized - control_annualized >= 0.20
                ),
                f"{name}_market_excess_at_least_20pp": (
                    strategy_annualized is not None
                    and market_annualized is not None
                    and strategy_annualized - market_annualized >= 0.20
                ),
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
                f"{name}_ending_unresolved_positions_zero": result[
                    "integrity"
                ]["ending_unresolved_positions"]
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


def _load_nav(root: Path) -> pl.DataFrame:
    files = sorted((root / "nav").glob("symbol=*/part.parquet"))
    if not files:
        raise FileNotFoundError(f"no NAV partitions under {root / 'nav'}")
    return pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    price_root = data_dir / "research" / "etf_cross_asset_v2"
    nav_root = data_dir / "research" / "etf_premium_reversal"
    daily = pl.read_parquet(price_root / "daily_raw.parquet")
    adjustments = pl.read_parquet(price_root / "adjustments.parquet")
    master = pl.read_parquet(price_root / "master.parquet").filter(
        pl.col("fund_type") == "股票型"
    )
    daily = daily.filter(pl.col("symbol").is_in(master["symbol"].to_list()))
    nav = _load_nav(nav_root)
    panel = prepare_panel(daily, adjustments, master)
    schedule = weekly_schedule(panel)
    low_candidates, high_candidates, point_in_time = build_candidate_legs(
        panel, nav, schedule
    )
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
    low_accounts = {
        f"cny_{int(cash / 1000)}k": simulate(
            low_candidates, panel, all_dates, action_dates, cash
        )
        for cash in CAPITAL_LEVELS
    }
    high_accounts = {
        f"cny_{int(cash / 1000)}k": simulate(
            high_candidates, panel, all_dates, action_dates, cash
        )
        for cash in CAPITAL_LEVELS
    }
    benchmark = benchmark_metrics(panel)
    active_rebalances = low_candidates["entry_date"].n_unique()
    decision = evaluate_gate(
        low_accounts, high_accounts, benchmark, active_rebalances
    )
    payload = {
        "schema_version": "p0-etf-premium-reversal-development-v1",
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
            "minimum_listing_days": MIN_LISTING_DAYS,
            "liquidity_days": LIQUIDITY_DAYS,
            "minimum_mean_amount_cny": MIN_MEAN_AMOUNT,
            "premium_price": "unadjusted signal close",
            "nav": "same nav_date unit_nav; ann_date strictly before entry",
            "selection": "lowest premium decile, negative premium only, max 10",
            "direction_control": "highest premium decile, max 10",
            "commission_pct": baseline.COMMISSION_PCT,
            "minimum_commission_cny": account.MIN_COMMISSION,
            "slippage_pct": baseline.SLIPPAGE_PCT,
            "stamp_tax_pct": 0.0,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "execution": "weekly prior-close signal, next trade day open",
        },
        "data": {
            "master_symbols": master.height,
            "daily_symbols": daily["symbol"].n_unique(),
            "nav_symbols": nav["symbol"].n_unique(),
            "scheduled_rebalances": schedule.height,
            "active_rebalances": active_rebalances,
            "low_signal_rows": low_candidates.height,
            "low_signal_symbols": low_candidates["symbol"].n_unique(),
            "high_signal_rows": high_candidates.height,
            "high_signal_symbols": high_candidates["symbol"].n_unique(),
            **point_in_time,
        },
        "benchmark": benchmark,
        "low_discount_accounts": low_accounts,
        "high_premium_control_accounts": high_accounts,
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
                "low_discount_accounts": low_accounts,
                "high_premium_control_accounts": high_accounts,
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
            "/app/data/research/p0_etf_premium_reversal_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
