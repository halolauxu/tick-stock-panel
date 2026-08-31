"""Run the frozen micro-cap low absolute-price-limit-beta development study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
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

from app.price_limits import polars_limit_price  # noqa: E402

WARMUP_START = date(2013, 1, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
MIN_REGRESSION_DAYS = 15
MIN_AMOUNT_20D = 50_000_000.0
MAX_EXIT_DELAY = 20


def load_daily(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    return (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "raw_close",
        )
        .filter(
            pl.col("date").is_between(WARMUP_START, DEVELOPMENT_END)
            & pl.col("symbol").str.contains(baseline.SYMBOL_PATTERN)
        )
        .collect(engine="streaming")
    )


def prepare_panel(source: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    dates = source.select("date").unique().sort("date").with_row_index("_date_index")
    indexed = source.join(dates, on="date", how="left").sort(["symbol", "date"])
    indexed = indexed.with_columns(
        pl.col("_date_index").shift(19).over("symbol").alias("_index_19d"),
        pl.col("amount").rolling_mean(20).over("symbol").alias("mean_amount_20d"),
    ).with_columns(
        pl.when(pl.col("_date_index") == pl.col("_index_19d") + 19)
        .then(pl.col("mean_amount_20d"))
        .otherwise(None)
        .alias("mean_amount_20d")
    )
    pit = baseline.attach_point_in_time_data(indexed, data_dir)
    work = pit.sort(["symbol", "date"]).with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_adj_factor"),
        pl.col("_date_index").shift(1).over("symbol").alias("_prev_index"),
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_prev_raw_close"),
        (pl.col("close").shift(1).over("symbol") / pl.col("raw_close").shift(1).over("symbol")).alias("_prev_adj_factor"),
    ).with_columns(
        (pl.col("_date_index") == pl.col("_prev_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_adj_factor")).alias("raw_open"),
        (pl.col("high") / pl.col("_adj_factor")).alias("raw_high"),
        (pl.col("low") / pl.col("_adj_factor")).alias("raw_low"),
    )
    changed = (pl.col("_adj_factor") - pl.col("_prev_adj_factor")).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(pl.when(changed).then(pl.col("_prev_close")).otherwise(pl.col("_prev_raw_close")))
        .otherwise(None)
        .alias("_reference_close"),
        baseline.price_limit_pct().alias("_limit_pct"),
        pl.when(pl.col("_adjacent"))
        .then(pl.col("close") / pl.col("_prev_close") - 1.0)
        .otherwise(None)
        .alias("daily_return"),
    ).with_columns(
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=True).alias("limit_up_price"),
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=False).alias("limit_down_price"),
        (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
    ).with_columns(
        (
            (pl.col("volume") > 0)
            & pl.col("_reference_close").is_not_null()
            & (
                (pl.col("raw_high") >= pl.col("limit_up_price") - 0.005)
                | (pl.col("raw_low") <= pl.col("limit_down_price") + 0.005)
            )
        ).alias("limit_hit")
    )
    market = (
        work.filter((pl.col("volume") > 0) & pl.col("daily_return").is_not_null())
        .group_by("date")
        .agg(
            pl.col("daily_return").mean().alias("market_return"),
            pl.col("limit_hit").mean().alias("limit_hit_fraction"),
        )
    )
    return (
        work.join(market, on="date", how="left")
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .select(
            "symbol",
            "date",
            "month",
            "open",
            "raw_open",
            "close",
            "raw_close",
            "volume",
            "amount",
            "mean_amount_20d",
            "market_cap",
            "daily_return",
            "market_return",
            "limit_hit_fraction",
        )
    )


def compute_apl_betas(
    panel: pl.DataFrame, *, minimum_days: int = MIN_REGRESSION_DAYS
) -> pl.DataFrame:
    x1 = pl.col("market_return")
    x2 = pl.col("limit_hit_fraction")
    y = pl.col("daily_return")
    moments = (
        panel.drop_nulls(["daily_return", "market_return", "limit_hit_fraction"])
        .group_by("symbol", "month")
        .agg(
            pl.len().alias("observations"),
            x1.sum().alias("sx1"),
            x2.sum().alias("sx2"),
            y.sum().alias("sy"),
            (x1 * x1).sum().alias("sx1x1"),
            (x2 * x2).sum().alias("sx2x2"),
            (x1 * x2).sum().alias("sx1x2"),
            (x1 * y).sum().alias("sx1y"),
            (x2 * y).sum().alias("sx2y"),
        )
        .filter(pl.col("observations") >= minimum_days)
        .with_columns(
            (pl.col("sx1x1") - pl.col("sx1") ** 2 / pl.col("observations")).alias("c11"),
            (pl.col("sx2x2") - pl.col("sx2") ** 2 / pl.col("observations")).alias("c22"),
            (pl.col("sx1x2") - pl.col("sx1") * pl.col("sx2") / pl.col("observations")).alias("c12"),
            (pl.col("sx1y") - pl.col("sx1") * pl.col("sy") / pl.col("observations")).alias("c1y"),
            (pl.col("sx2y") - pl.col("sx2") * pl.col("sy") / pl.col("observations")).alias("c2y"),
        )
        .with_columns((pl.col("c11") * pl.col("c22") - pl.col("c12") ** 2).alias("denominator"))
        .filter(pl.col("denominator").abs() > 1e-12)
        .with_columns(
            ((pl.col("c2y") * pl.col("c11") - pl.col("c1y") * pl.col("c12")) / pl.col("denominator"))
            .abs()
            .alias("apl_beta")
        )
    )
    return moments.select("symbol", "month", "observations", "apl_beta").sort(["month", "symbol"])


def build_monthly_observations(
    panel: pl.DataFrame, betas: pl.DataFrame
) -> pl.DataFrame:
    dates = panel.select("date").unique().sort("date").with_row_index("_date_index")
    month_calendar = (
        dates.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("_date_index").max().alias("signal_index"),
        )
        .sort("signal_date")
        .with_columns((pl.col("signal_index") + 1).alias("entry_index"))
        .join(
            dates.rename({"date": "entry_date"}),
            left_on="entry_index",
            right_on="_date_index",
            how="left",
        )
        .with_columns(pl.col("entry_date").shift(-1).alias("next_rebalance_date"))
    )
    date_list = dates.get_column("date").to_list()
    date_index = {day: offset for offset, day in enumerate(date_list)}
    valid_months = [
        row
        for row in month_calendar.to_dicts()
        if row.get("entry_date") is not None
        and row.get("next_rebalance_date") is not None
        and date_index[row["next_rebalance_date"]] + MAX_EXIT_DELAY < len(date_list)
        and row["signal_date"] >= DEVELOPMENT_START
    ]
    calendar = pl.DataFrame(valid_months, infer_schema_length=None).select(
        "month", "signal_date", "entry_date", "next_rebalance_date"
    )
    signal_rows = panel.join(
        calendar.select("month", "signal_date"),
        left_on=["month", "date"],
        right_on=["month", "signal_date"],
        how="inner",
    )
    return (
        signal_rows.join(betas, on=["symbol", "month"], how="inner")
        .join(calendar, on="month", how="inner")
        .filter(
            (pl.col("mean_amount_20d") >= MIN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("market_cap") > 0)
        )
        .select(
            "signal_date",
            "entry_date",
            "next_rebalance_date",
            "symbol",
            "apl_beta",
            "market_cap",
            "amount",
        )
    )


def build_candidates(
    observations: pl.DataFrame, *, strategy: str, top_n: int = TARGET_POSITIONS
) -> pl.DataFrame:
    work = observations.sort(["signal_date", "market_cap", "symbol"]).with_columns(
        pl.len().over("signal_date").alias("universe_count"),
        pl.int_range(1, pl.len() + 1).over("signal_date").alias("market_cap_rank"),
    ).filter(pl.col("market_cap_rank") <= (pl.col("universe_count") * 0.10).ceil())
    if strategy == "low_apl":
        sort_columns = ["signal_date", "apl_beta", "market_cap", "symbol"]
    elif strategy == "microcap":
        sort_columns = ["signal_date", "market_cap", "symbol"]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return (
        work.sort(sort_columns)
        .with_columns(pl.int_range(1, pl.len() + 1).over("signal_date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= top_n)
        .select(
            "signal_date",
            "entry_date",
            "next_rebalance_date",
            "symbol",
            "apl_beta",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _rejection(
    quote: dict[str, Any] | None, *, side: str, gross: float
) -> str | None:
    if quote is None or quote.get("raw_open") is None:
        return "missing_market_data"
    if side == "BUY" and quote.get("is_excluded_name"):
        return "risk_warning"
    if not quote.get("volume") or quote["volume"] <= 0:
        return "suspended"
    raw_open = float(quote["raw_open"])
    if raw_open <= 0:
        return "missing_open"
    limit = quote.get("limit_up_price" if side == "BUY" else "limit_down_price")
    if limit is not None:
        if side == "BUY" and raw_open >= float(limit) - 0.005:
            return "limit_up"
        if side == "SELL" and raw_open <= float(limit) + 0.005:
            return "limit_down"
    if gross > float(quote.get("amount") or 0.0) * baseline.DAILY_PARTICIPATION:
        return "insufficient_capacity"
    return None


def simulate_monthly_account(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    trading_dates: list[date],
    *,
    rebalance_dates: list[date],
    initial_cash: float,
    target_positions: int = TARGET_POSITIONS,
    max_exit_delay: int = MAX_EXIT_DELAY,
) -> dict[str, Any]:
    candidate_groups = {
        key[0] if isinstance(key, tuple) else key: group.sort(["cap_rank", "symbol"])
        for key, group in candidates.partition_by("entry_date", as_dict=True).items()
    }
    quote_lookup = {(row["date"], row["symbol"]): row for row in quotes.to_dicts()}
    date_index = {day: offset for offset, day in enumerate(trading_dates)}
    rebalances = set(rebalance_dates)
    positions: dict[str, dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    cash = float(initial_cash)
    cash_ledger = float(initial_cash)
    max_cash_error = 0.0
    position_id = 0

    for day in trading_dates:
        rows = candidate_groups.get(day)
        if day in rebalances:
            desired = set(rows.get_column("symbol").to_list()) if rows is not None else set()
            for symbol, position in positions.items():
                if symbol in desired:
                    if not position.get("terminal_exit_failure"):
                        position["exit_due_date"] = None
                elif position.get("exit_due_date") is None:
                    position["exit_due_date"] = day

        for symbol in list(positions):
            position = positions[symbol]
            due = position.get("exit_due_date")
            if due is None or day < due or position.get("terminal_exit_failure"):
                continue
            quote = quote_lookup.get((day, symbol))
            raw_open = float(quote.get("raw_open") or 0.0) if quote else 0.0
            gross = position["raw_shares"] * raw_open
            reason = _rejection(quote, side="SELL", gross=gross)
            exit_delay = date_index[day] - date_index[due]
            order = {
                "date": day,
                "symbol": symbol,
                "side": "SELL",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
            }
            orders.append(order)
            if reason:
                if exit_delay >= max_exit_delay:
                    position["terminal_exit_failure"] = reason
                continue
            commission_fee = account.commission(gross)
            stamp_tax = gross * (baseline.STAMP_TAX_OLD if day < baseline.STAMP_TAX_CUT else baseline.STAMP_TAX_CURRENT)
            slippage = gross * baseline.SLIPPAGE_PCT
            cash_delta = gross - commission_fee - stamp_tax - slippage
            cash += cash_delta
            cash_ledger += cash_delta
            order.update(
                gross=gross,
                commission=commission_fee,
                stamp_tax=stamp_tax,
                slippage=slippage,
                cash_delta=cash_delta,
            )
            trades.append(order.copy())
            completed.append(
                {
                    "symbol": symbol,
                    "entry_date": position["start_date"],
                    "exit_date": day,
                    "exit_delay_days": exit_delay,
                    "pnl": cash_delta - position["entry_cash_out"],
                }
            )
            intervals.append(
                {
                    "position_id": position["position_id"],
                    "symbol": symbol,
                    "units": position["units"],
                    "start_date": position["start_date"],
                    "end_date": day,
                }
            )
            del positions[symbol]

        for symbol, position in positions.items():
            quote = quote_lookup.get((day, symbol))
            if quote and quote.get("raw_close") is not None:
                position["last_raw_price"] = float(quote["raw_close"])

        if day in rebalances and rows is not None:
            equity = cash
            for symbol, position in positions.items():
                quote = quote_lookup.get((day, symbol))
                mark = float(quote.get("raw_open") or position["last_raw_price"]) if quote else position["last_raw_price"]
                equity += position["raw_shares"] * mark
            target_notional = equity / target_positions
            slots = max(0, target_positions - len(positions))
            for candidate in rows.to_dicts():
                symbol = candidate["symbol"]
                if symbol in positions:
                    orders.append({"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "ALREADY_HELD"})
                    continue
                if slots <= 0:
                    orders.append({"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "NO_SLOT"})
                    continue
                if target_notional > float(candidate["signal_amount"]) * baseline.DAILY_PARTICIPATION:
                    orders.append({"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "signal_capacity"})
                    continue
                quote = quote_lookup.get((day, symbol))
                raw_open = float(quote.get("raw_open") or 0.0) if quote else 0.0
                shares = account.affordable_shares(raw_open, target_notional, cash)
                gross = shares * raw_open
                reason = "zero_lot_or_cash" if shares <= 0 else _rejection(quote, side="BUY", gross=gross)
                order = {
                    "date": day,
                    "signal_date": candidate["signal_date"],
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "REJECTED" if reason else "FILLED",
                    "reason": reason,
                }
                orders.append(order)
                if reason:
                    continue
                commission_fee = account.commission(gross)
                slippage = gross * baseline.SLIPPAGE_PCT
                cash_delta = -(gross + commission_fee + slippage)
                cash += cash_delta
                cash_ledger += cash_delta
                adjusted_open = float(quote["open"])
                position_id += 1
                positions[symbol] = {
                    "position_id": position_id,
                    "symbol": symbol,
                    "raw_shares": shares,
                    "units": gross / adjusted_open,
                    "start_date": day,
                    "entry_cash_out": -cash_delta,
                    "last_raw_price": raw_open,
                    "exit_due_date": None,
                }
                order.update(
                    gross=gross,
                    commission=commission_fee,
                    stamp_tax=0.0,
                    slippage=slippage,
                    cash_delta=cash_delta,
                )
                trades.append(order.copy())
                slots -= 1

        snapshots.append({"date": day, "cash": cash})
        max_cash_error = max(max_cash_error, abs(cash - cash_ledger))

    for position in positions.values():
        intervals.append(
            {
                "position_id": position["position_id"],
                "symbol": position["symbol"],
                "units": position["units"],
                "start_date": position["start_date"],
                "end_date": None,
            }
        )
    return {
        "orders": orders,
        "trades": trades,
        "completed_positions": completed,
        "snapshots": snapshots,
        "intervals": intervals,
        "ending_positions": list(positions.values()),
        "ending_cash": cash,
        "max_cash_reconciliation_error": max_cash_error,
    }


def summarize_account(
    simulation: dict[str, Any], quotes: pl.DataFrame, trading_dates: list[date], initial_cash: float
) -> dict[str, Any]:
    daily, stale = account.build_daily_equity(simulation, quotes, trading_dates, initial_cash=initial_cash)
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    for year in range(2014, 2021):
        values = daily.filter(pl.col("date").dt.year() == year).get_column("daily_return").drop_nulls().to_list()
        yearly.append({"year": year, "return": baseline._compound(values)})
    completed = simulation["completed_positions"]
    positive = [max(0.0, float(row["pnl"])) for row in completed]
    profit_by_symbol: Counter[str] = Counter()
    for row in completed:
        profit_by_symbol[row["symbol"]] += max(0.0, float(row["pnl"]))
    total_positive = sum(positive)
    execution = account.execution_summary(simulation["orders"])
    planned = sum(row["side"] == "BUY" for row in simulation["orders"])
    capacity_skips = sum(row.get("reason") in {"signal_capacity", "insufficient_capacity"} for row in simulation["orders"] if row["side"] == "BUY")
    return {
        "annualized": _daily_annualized(returns),
        "total_return": baseline._compound(returns),
        "max_drawdown": baseline._max_drawdown(returns),
        "positive_years": sum((row["return"] or 0.0) > 0 for row in yearly),
        "yearly": yearly,
        "execution": execution,
        "capacity_feasibility_rate": 1.0 - capacity_skips / planned if planned else 0.0,
        "unresolved_exits": len(simulation["ending_positions"]),
        "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        "max_positive_profit_contribution": max(profit_by_symbol.values(), default=0.0) / total_positive if total_positive > 0 else None,
        "stale": stale,
        "ending_equity": float(daily.get_column("equity")[-1]),
        "completed_positions": len(completed),
    }


def _daily_annualized(values: list[float]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(value)]
    total = baseline._compound(valid)
    if total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(valid)) - 1.0


def evaluate_gate(candidate: dict[str, Any], control: dict[str, Any], signal_months: int) -> dict[str, Any]:
    primary = candidate[str(int(CAPITALS[0]))]
    primary_control = control[str(int(CAPITALS[0]))]
    checks = {
        "signal_months_at_least_75": signal_months >= 75,
        "annualized_at_least_50pct": (primary["annualized"] or -math.inf) >= 0.50,
        "annualized_advantage_at_least_10pp": (primary["annualized"] or -math.inf) - (primary_control["annualized"] or math.inf) >= 0.10,
        "max_drawdown_at_least_minus_30pct": (primary["max_drawdown"] or -math.inf) >= -0.30,
        "positive_years_at_least_5": primary["positive_years"] >= 5,
        "all_buy_execution_at_least_90pct": min(row["execution"]["buy"]["execution_rate"] for row in candidate.values()) >= 0.90,
        "all_sell_execution_at_least_90pct": min(row["execution"]["sell"]["execution_rate"] for row in candidate.values()) >= 0.90,
        "all_capacity_at_least_95pct": min(row["capacity_feasibility_rate"] for row in candidate.values()) >= 0.95,
        "all_unresolved_exits_zero": max(row["unresolved_exits"] for row in candidate.values()) == 0,
        "cash_reconciliation_at_most_one_cent": max(row["max_cash_reconciliation_error"] for row in candidate.values()) <= 0.01,
        "profit_contribution_at_most_25pct": (primary["max_positive_profit_contribution"] or math.inf) <= 0.25,
    }
    passed = all(checks.values())
    return {"passed": passed, "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE", "checks": checks}


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source = load_daily(data_dir)
    quote_panel = account.prepare_quote_panel(account.attach_quote_names(source, data_dir))
    panel = prepare_panel(source, data_dir)
    del source
    gc.collect()
    betas = compute_apl_betas(panel)
    observations = build_monthly_observations(panel, betas)
    candidate_rows = build_candidates(observations, strategy="low_apl")
    control_rows = build_candidates(observations, strategy="microcap")
    symbols = set(candidate_rows.get_column("symbol").to_list()) | set(control_rows.get_column("symbol").to_list())
    quotes = quote_panel.filter(pl.col("symbol").is_in(symbols))
    all_trading_dates = panel.get_column("date").unique().sort().to_list()
    del panel, quote_panel
    gc.collect()
    first_entry = min(candidate_rows.get_column("entry_date").min(), control_rows.get_column("entry_date").min())
    final_rebalance = max(candidate_rows.get_column("next_rebalance_date").max(), control_rows.get_column("next_rebalance_date").max())
    trading_dates = [
        day for day in all_trading_dates if first_entry <= day <= DEVELOPMENT_END
    ]
    rebalance_dates = sorted(set(candidate_rows.get_column("entry_date").to_list()) | set(control_rows.get_column("entry_date").to_list()) | {final_rebalance})
    results: dict[str, dict[str, Any]] = {"low_apl": {}, "microcap": {}}
    for strategy, rows in (("low_apl", candidate_rows), ("microcap", control_rows)):
        for capital in CAPITALS:
            simulation = simulate_monthly_account(
                rows,
                quotes,
                trading_dates,
                rebalance_dates=rebalance_dates,
                initial_cash=capital,
            )
            results[strategy][str(int(capital))] = summarize_account(
                simulation, quotes, trading_dates, capital
            )
    signal_months = candidate_rows.get_column("entry_date").n_unique()
    decision = evaluate_gate(results["low_apl"], results["microcap"], signal_months)
    payload = {
        "schema_version": "p0-microcap-apl-beta-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "warmup_start": WARMUP_START,
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "validation_read": False,
            "stress_read": False,
        },
        "data": {
            "apl_beta_rows": betas.height,
            "observation_rows": observations.height,
            "signal_months": signal_months,
            "candidate_rows": candidate_rows.height,
            "control_rows": control_rows.height,
            "candidate_symbols": candidate_rows.get_column("symbol").n_unique(),
        },
        "accounts": results,
        "decision": decision,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "decision": decision, "strict_qualified_count": 0}, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backend" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "data" / "research" / "p0_microcap_apl_beta_development.json")
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
