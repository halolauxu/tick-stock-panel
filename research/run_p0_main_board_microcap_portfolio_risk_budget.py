"""Test a capped, volatility-targeted main-board micro-cap portfolio sleeve."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

STUDY_END = date(2026, 8, 28)
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
MAX_MICROCAP_WEIGHT = 0.20
PORTFOLIO_VOLATILITY_TARGET = 0.06
VOLATILITY_WINDOW = 20


def _period_bounds(mode: str, last_date: date) -> tuple[date, date, str]:
    if mode == "development":
        return baseline.START, baseline.DEVELOPMENT_END, "development"
    if mode == "validation":
        return date(2021, 1, 1), baseline.VALIDATION_END, "validation"
    if mode == "stress":
        return date(2024, 1, 1), last_date, "known_stress"
    raise ValueError(f"unsupported mode: {mode}")


def build_volatility_exposure_schedule(
    panel: pl.DataFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    eligible = (
        panel.filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile")
        )
        .filter(pl.col("cap_decile") == 0)
    )
    daily = (
        eligible.group_by("date")
        .agg(pl.col("daily_return").mean().alias("microcap_return"))
        .sort("date")
        .with_columns(
            (
                pl.col("microcap_return").rolling_std(
                    window_size=VOLATILITY_WINDOW,
                    min_samples=VOLATILITY_WINDOW,
                )
                * math.sqrt(252.0)
            ).alias("realized_volatility")
        )
    )
    rebalances = (
        candidates.select(
            pl.col("date").alias("signal_date"), "entry_date"
        )
        .unique()
        .sort("entry_date")
    )
    return (
        rebalances.join(
            daily.rename({"date": "signal_date"}),
            on="signal_date",
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col("realized_volatility").is_null()
                | (pl.col("realized_volatility") <= 0)
            )
            .then(pl.lit(MAX_MICROCAP_WEIGHT))
            .otherwise(
                (
                    pl.lit(PORTFOLIO_VOLATILITY_TARGET)
                    / pl.col("realized_volatility")
                ).clip(0.0, MAX_MICROCAP_WEIGHT)
            )
            .alias("target_exposure")
        )
        .select(
            "signal_date",
            "entry_date",
            "microcap_return",
            "realized_volatility",
            "target_exposure",
        )
    )


def load_delist_dates(data_dir: Path, symbols: list[str]) -> dict[str, date]:
    master = data_dir / "research" / "historical_stock_universe_all_a.parquet"
    if not master.is_file():
        raise ValueError("all-A PIT security master is required")
    rows = (
        pl.read_parquet(master)
        .with_columns(pl.col("delist_date").cast(pl.Date, strict=False))
        .filter(
            pl.col("symbol").is_in(symbols)
            & pl.col("delist_date").is_not_null()
        )
        .select("symbol", "delist_date")
        .unique(subset=["symbol"], keep="last")
        .to_dicts()
    )
    return {row["symbol"]: row["delist_date"] for row in rows}


def _exposure_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    active_positions = [
        int(row["position_count"])
        for row in snapshots
        if int(row["position_count"]) > 0
    ]
    return {
        "rebalance_count": len(snapshots),
        "mean_target_exposure": statistics.fmean(
            float(row["target_exposure"]) for row in snapshots
        ),
        "mean_actual_exposure": statistics.fmean(
            float(row["actual_exposure"]) for row in snapshots
        ),
        "max_actual_exposure": max(
            float(row["actual_exposure"]) for row in snapshots
        ),
        "median_active_positions": (
            statistics.median(active_positions) if active_positions else 0.0
        ),
        "risk_budget_blocked_slots": sum(
            int(row["risk_budget_blocked_slots"]) for row in snapshots
        ),
    }


def _simulate_period(
    *,
    period: str,
    capital: float,
    candidates: pl.DataFrame,
    execution_grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
    weekly_market: pl.DataFrame,
    exposure_by_date: dict[date, float],
    delist_dates: dict[str, date],
) -> dict[str, Any]:
    scoped_dates = account.period_dates(all_dates, period)
    first_date, last_date = scoped_dates[0], scoped_dates[-1]
    scoped_candidates = candidates.filter(
        (pl.col("entry_date") >= pl.lit(first_date))
        & (pl.col("entry_date") <= pl.lit(last_date))
    )
    scoped_grid = execution_grid.filter(
        (pl.col("entry_date") >= pl.lit(first_date))
        & (pl.col("entry_date") <= pl.lit(last_date))
    )
    action_dates = (
        scoped_candidates.get_column("entry_date").unique().sort().to_list()
    )
    simulation = account.simulate_account(
        scoped_candidates,
        scoped_grid,
        initial_cash=capital,
        action_dates=action_dates,
        delist_dates=delist_dates,
        target_exposure_by_date=exposure_by_date,
    )
    daily, stale = account.build_daily_equity(
        simulation,
        quotes,
        scoped_dates,
        initial_cash=capital,
    )
    metric = next(
        row
        for row in account.account_period_metrics(daily, weekly_market)
        if row["period"] == period
    )
    integrity = {
        **stale,
        "max_cash_reconciliation_error": simulation[
            "max_cash_reconciliation_error"
        ],
    }
    return {
        "capital": capital,
        "period": period,
        "first_date": first_date,
        "last_date": last_date,
        "metrics": metric,
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": integrity,
        "account": account.account_summary(simulation, daily),
        "exposure": _exposure_summary(simulation["snapshots"]),
        "drawdown_episode": main_board.drawdown_episode(
            daily.select("date", "equity").to_dicts()
        ),
        "daily_equity": daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts(),
        "rebalance_snapshots": simulation["snapshots"],
        "orders": simulation["orders"],
        "settlements": simulation["settlements"],
        "worst_weeks": account.worst_weeks(daily),
    }


def evaluate(
    period: str,
    dynamic: dict[str, Any],
    fixed: dict[str, Any],
) -> dict[str, Any]:
    metrics = dynamic["metrics"]
    fixed_metrics = fixed["metrics"]
    execution = dynamic["execution"]
    integrity = dynamic["integrity"]
    exposure = dynamic["exposure"]
    annualized = metrics.get("account_annualized")
    fixed_annualized = fixed_metrics.get("account_annualized")
    drawdown = metrics.get("account_max_drawdown")
    fixed_drawdown = fixed_metrics.get("account_max_drawdown")
    buy_orders = execution["buy"]["orders"]
    zero_lot = execution["buy"]["rejection_reasons"].get(
        "zero_lot_or_cash", 0
    )
    zero_lot_rate = zero_lot / buy_orders if buy_orders else 0.0
    positive_years_required = 5 if period == "development" else 2
    checks = {
        "annualized_at_least_3pct": (annualized or -math.inf) >= 0.03,
        "max_drawdown_within_12pct": (drawdown or -math.inf) >= -0.12,
        "annualized_loss_vs_fixed_within_2pp": (
            annualized is not None
            and fixed_annualized is not None
            and annualized - fixed_annualized >= -0.02
        ),
        "drawdown_increment_is_useful": (
            fixed_drawdown is not None
            and drawdown is not None
            and (
                fixed_drawdown >= -0.10
                or drawdown - fixed_drawdown >= 0.01
            )
        ),
        "positive_years": (
            metrics.get("positive_account_years", 0)
            >= positive_years_required
        ),
        "median_active_positions_at_least_10": (
            exposure["median_active_positions"] >= 10
        ),
        "buy_execution_at_least_80pct": (
            execution["buy"]["execution_rate"] >= 0.80
        ),
        "sell_execution_at_least_80pct": (
            execution["sell"]["execution_rate"] >= 0.80
        ),
        "zero_lot_rejection_at_most_30pct": zero_lot_rate <= 0.30,
        "no_unresolved_positions": (
            integrity["ending_unresolved_positions"] == 0
        ),
        "cash_reconciled": (
            integrity["max_cash_reconciliation_error"] <= 0.01
        ),
    }
    passed = all(checks.values())
    next_verdict = {
        "development": "PROMOTE_TO_VALIDATION",
        "validation": "PROMOTE_TO_STRESS",
        "known_stress": "PROMOTE_TO_FORWARD_SHADOW",
    }[period]
    return {
        "passed": passed,
        "verdict": next_verdict if passed else "TERMINATE",
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "zero_lot_rejection_rate": zero_lot_rate,
        "annualized_delta_vs_fixed": (
            annualized - fixed_annualized
            if annualized is not None and fixed_annualized is not None
            else None
        ),
        "drawdown_delta_vs_fixed": (
            drawdown - fixed_drawdown
            if drawdown is not None and fixed_drawdown is not None
            else None
        ),
    }


def _summary_only(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "daily_equity",
            "rebalance_snapshots",
            "orders",
            "settlements",
        }
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(
    data_dir: Path,
    output: Path,
    *,
    mode: str,
    end: date = STUDY_END,
) -> dict[str, Any]:
    load_end = {
        "development": baseline.DEVELOPMENT_END,
        "validation": baseline.VALIDATION_END,
        "stress": end,
    }[mode]
    source = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=load_end)
    )
    if source.is_empty():
        raise ValueError("no main-board daily data")
    all_dates = source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    candidates = account.build_signal_candidates(panel)
    schedule = build_volatility_exposure_schedule(panel, candidates)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=load_end)
    ).filter(pl.col("symbol").is_in(candidate_symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(candidates, quotes)
    delist_dates = load_delist_dates(data_dir, candidate_symbols)

    start, finish, period = _period_bounds(mode, all_dates[-1])
    scoped_schedule = schedule.filter(
        (pl.col("entry_date") >= pl.lit(start))
        & (pl.col("entry_date") <= pl.lit(finish))
    )
    dynamic_exposure = {
        row["entry_date"]: float(row["target_exposure"])
        for row in scoped_schedule.select(
            "entry_date", "target_exposure"
        ).to_dicts()
    }
    fixed_exposure = {
        day: MAX_MICROCAP_WEIGHT for day in dynamic_exposure
    }
    tiers = []
    for capital in CAPITALS:
        dynamic = _simulate_period(
            period=period,
            capital=capital,
            candidates=candidates,
            execution_grid=execution_grid,
            quotes=quotes,
            all_dates=all_dates,
            weekly_market=weekly_market,
            exposure_by_date=dynamic_exposure,
            delist_dates=delist_dates,
        )
        fixed = _simulate_period(
            period=period,
            capital=capital,
            candidates=candidates,
            execution_grid=execution_grid,
            quotes=quotes,
            all_dates=all_dates,
            weekly_market=weekly_market,
            exposure_by_date=fixed_exposure,
            delist_dates=delist_dates,
        )
        tiers.append(
            {
                "capital": capital,
                "dynamic": (
                    dynamic
                    if capital == PRIMARY_CAPITAL
                    else _summary_only(dynamic)
                ),
                "fixed_20pct": (
                    fixed
                    if capital == PRIMARY_CAPITAL
                    else _summary_only(fixed)
                ),
            }
        )
    primary = next(row for row in tiers if row["capital"] == PRIMARY_CAPITAL)
    decision = evaluate(period, primary["dynamic"], primary["fixed_20pct"])
    payload = {
        "schema_version": "p0-main-board-microcap-portfolio-risk-budget-v1",
        "mode": mode,
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "signal": "unchanged_weekly_pit_market_cap_bottom_decile_20",
            "max_microcap_weight": MAX_MICROCAP_WEIGHT,
            "portfolio_volatility_target": PORTFOLIO_VOLATILITY_TARGET,
            "volatility_window": VOLATILITY_WINDOW,
            "risk_action": "normal_sells_then_buy_budget_only_no_forced_sale",
            "capital_ladder": list(CAPITALS),
            "primary_capital": PRIMARY_CAPITAL,
            "delisting_settlement": (
                "zero_recovery_on_first_action_at_or_after_delist_date"
            ),
        },
        "data": {
            "first_loaded_date": all_dates[0],
            "last_loaded_date": all_dates[-1],
            "period_start": start,
            "period_end": finish,
            "period_trading_days": len(
                [day for day in all_dates if start <= day <= finish]
            ),
            "candidate_symbols": len(candidate_symbols),
            "rebalance_count": scoped_schedule.height,
            "candidate_board_counts": baseline.board_symbol_counts(
                pl.DataFrame({"symbol": candidate_symbols})
            ),
        },
        "exposure_schedule": scoped_schedule.to_dicts(),
        "capital_tiers": tiers,
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
                "mode": mode,
                "data": payload["data"],
                "primary": {
                    "dynamic": _summary_only(primary["dynamic"]),
                    "fixed_20pct": _summary_only(primary["fixed_20pct"]),
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
        "--mode",
        choices=("development", "validation", "stress"),
        required=True,
    )
    parser.add_argument("--end", type=date.fromisoformat, default=STUDY_END)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "/app/data/research/"
        f"p0_main_board_microcap_portfolio_risk_budget_{args.mode}_v1.json"
    )
    run(args.data_dir, output, mode=args.mode, end=args.end)


if __name__ == "__main__":
    main()
