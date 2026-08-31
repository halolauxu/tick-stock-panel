"""Run the frozen 2014-2018 Alpha101 long-only feasibility screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import alpha101_formulas as formulas  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

from app.price_limits import polars_limit_price  # noqa: E402

ALPHA_IDS = formulas.ALPHA_IDS
WARMUP_START = date(2013, 1, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2018, 12, 31)
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
TARGET_POSITIONS = 10
MIN_AMOUNT_20D = 50_000_000.0
MAX_EXIT_DELAY = 20
FAMILY_ALPHA = 0.05
TOP_N = 10


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


def prepare_alpha_panel(source: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
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
    )
    changed = (pl.col("_adj_factor") - pl.col("_prev_adj_factor")).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(pl.when(changed).then(pl.col("_prev_close")).otherwise(pl.col("_prev_raw_close")))
        .otherwise(None)
        .alias("_reference_close"),
        baseline.price_limit_pct().alias("_limit_pct"),
    ).with_columns(
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=True).alias("limit_up_price"),
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=False).alias("limit_down_price"),
        (
            (pl.col("date") >= DEVELOPMENT_START)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("mean_amount_20d") >= MIN_AMOUNT_20D)
            & pl.col("_adjacent")
        ).alias("eligible"),
        pl.lit(False).alias("is_excluded_name"),
    )
    return work.select(
        "symbol",
        "date",
        "_date_index",
        "open",
        "high",
        "low",
        "close",
        "raw_open",
        "raw_close",
        "volume",
        "amount",
        "eligible",
        "limit_up_price",
        "limit_down_price",
        "is_excluded_name",
    )


def panel_matrices(
    panel: pl.DataFrame,
) -> tuple[formulas.Alpha101Context, np.ndarray, np.ndarray, list[date], np.ndarray]:
    dates = panel.get_column("date").unique().sort().to_list()
    symbols = np.asarray(panel.get_column("symbol").unique().sort().to_list())
    date_ids = pl.DataFrame({"date": dates}).with_row_index("_row")
    symbol_ids = pl.DataFrame({"symbol": symbols}).with_row_index("_column")
    indexed = panel.join(date_ids, on="date", how="left").join(
        symbol_ids, on="symbol", how="left"
    )
    row_ids = indexed.get_column("_row").to_numpy()
    column_ids = indexed.get_column("_column").to_numpy()
    shape = (len(dates), len(symbols))

    def matrix(column: str, *, dtype: Any = np.float32) -> np.ndarray:
        result = np.full(shape, np.nan, dtype=dtype)
        result[row_ids, column_ids] = indexed.get_column(column).to_numpy()
        return result

    context = formulas.Alpha101Context.from_arrays(
        open=matrix("open"),
        high=matrix("high"),
        low=matrix("low"),
        close=matrix("close"),
        volume=matrix("volume"),
        amount=matrix("amount"),
    )
    eligible = np.zeros(shape, dtype=bool)
    eligible[row_ids, column_ids] = indexed.get_column("eligible").to_numpy()
    amount = matrix("amount", dtype=np.float64)
    return context, eligible, amount, dates, symbols


def select_candidates(
    alpha_id: int,
    values: np.ndarray,
    eligible: np.ndarray,
    amount: np.ndarray,
    dates: list[date],
    symbols: np.ndarray,
    *,
    development_start: date = DEVELOPMENT_START,
    top_n: int = TOP_N,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_id, signal_date in enumerate(dates):
        if signal_date < development_start:
            continue
        valid = eligible[row_id] & np.isfinite(values[row_id])
        asset_ids = np.flatnonzero(valid)
        if not asset_ids.size:
            continue
        order = np.lexsort((symbols[asset_ids], -values[row_id, asset_ids]))
        for rank, asset_id in enumerate(asset_ids[order[:top_n]], start=1):
            rows.append(
                {
                    "alpha_id": alpha_id,
                    "signal_date": signal_date,
                    "symbol": str(symbols[asset_id]),
                    "rank": rank,
                    "alpha_value": float(values[row_id, asset_id]),
                    "signal_amount": float(amount[row_id, asset_id]),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def attach_execution_dates_and_benchmark(
    candidates: pl.DataFrame,
    context: formulas.Alpha101Context,
    eligible: np.ndarray,
    dates: list[date],
    *,
    max_exit_delay: int = MAX_EXIT_DELAY,
) -> pl.DataFrame:
    calendar = pl.DataFrame({"signal_date": dates}).with_columns(
        pl.col("signal_date").shift(-1).alias("entry_date"),
        pl.col("signal_date").shift(-2).alias("planned_exit_date"),
    )
    gross_benchmark: list[dict[str, Any]] = []
    for row_id, signal_date in enumerate(dates[:-2]):
        if signal_date < DEVELOPMENT_START:
            continue
        returns = context.open[row_id + 2] / context.open[row_id + 1] - 1.0
        valid = eligible[row_id] & np.isfinite(returns)
        gross_benchmark.append(
            {
                "signal_date": signal_date,
                "benchmark_return": float(np.median(returns[valid])) if valid.any() else None,
            }
        )
    liquidation_cutoff = dates[-(max_exit_delay + 1)]
    return (
        candidates.join(calendar, on="signal_date", how="left")
        .join(pl.DataFrame(gross_benchmark), on="signal_date", how="left")
        .filter(
            pl.col("planned_exit_date")
            <= min(DEVELOPMENT_END, liquidation_cutoff)
        )
        .drop_nulls(["entry_date", "planned_exit_date", "benchmark_return"])
        .sort(["entry_date", "rank", "symbol"])
    )


def build_execution_quotes(
    candidates: pl.DataFrame,
    panel: pl.DataFrame,
    trading_dates: list[date],
    *,
    max_exit_delay: int = MAX_EXIT_DELAY,
) -> pl.DataFrame:
    index = {day: offset for offset, day in enumerate(trading_dates)}
    wanted: set[tuple[str, date]] = set()
    for row in candidates.select("symbol", "entry_date", "planned_exit_date").to_dicts():
        entry_id = index[row["entry_date"]]
        exit_id = index[row["planned_exit_date"]]
        end_id = min(len(trading_dates), exit_id + max_exit_delay + 1)
        for day in trading_dates[entry_id:end_id]:
            wanted.add((row["symbol"], day))
    wanted_frame = pl.DataFrame(
        [{"symbol": symbol, "date": day} for symbol, day in sorted(wanted)],
        infer_schema_length=None,
    )
    return wanted_frame.join(panel, on=["symbol", "date"], how="left")


def _quote_rejection(quote: dict[str, Any] | None, *, side: str, gross: float) -> str | None:
    if quote is None or quote.get("raw_open") is None:
        return "missing_market_data"
    if quote.get("is_excluded_name"):
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


def simulate_daily_account(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    trading_dates: list[date],
    *,
    initial_cash: float,
    target_positions: int = TARGET_POSITIONS,
    max_exit_delay: int = MAX_EXIT_DELAY,
) -> dict[str, Any]:
    candidate_groups = {
        key[0] if isinstance(key, tuple) else key: group.sort(["rank", "symbol"])
        for key, group in candidates.partition_by("entry_date", as_dict=True).items()
    }
    quote_lookup = {
        (row["date"], row["symbol"]): row for row in quotes.to_dicts()
    }
    date_index = {day: offset for offset, day in enumerate(trading_dates)}
    positions: dict[str, dict[str, Any]] = {}
    cash = float(initial_cash)
    cash_ledger = float(initial_cash)
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    max_cash_error = 0.0

    for day in trading_dates:
        if day < min(candidate_groups, default=day):
            continue
        for symbol in list(positions):
            position = positions[symbol]
            if day < position["planned_exit_date"]:
                continue
            if position.get("terminal_exit_failure"):
                continue
            quote = quote_lookup.get((day, symbol))
            raw_open = float(quote.get("raw_open") or 0.0) if quote else 0.0
            gross = position["raw_shares"] * raw_open
            reason = _quote_rejection(quote, side="SELL", gross=gross)
            exit_delay = date_index[day] - date_index[position["planned_exit_date"]]
            order = {
                "date": day,
                "symbol": symbol,
                "side": "SELL",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
                "exit_delay_days": exit_delay,
            }
            orders.append(order)
            if reason:
                if exit_delay >= max_exit_delay:
                    position["terminal_exit_failure"] = reason
                continue
            commission_fee = account.commission(gross)
            stamp_tax = gross * (
                baseline.STAMP_TAX_OLD if day < baseline.STAMP_TAX_CUT else baseline.STAMP_TAX_CURRENT
            )
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
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "planned_exit_date": position["planned_exit_date"],
                    "exit_date": day,
                    "exit_delay_days": exit_delay,
                    "net_return": cash_delta / position["entry_cash_out"] - 1.0,
                    "benchmark_return": position["benchmark_return"],
                    "excess_return": cash_delta / position["entry_cash_out"] - 1.0 - position["benchmark_return"],
                }
            )
            del positions[symbol]

        rows = candidate_groups.get(day)
        if rows is not None:
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
                    orders.append(
                        {"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "ALREADY_HELD"}
                    )
                    continue
                if slots <= 0:
                    orders.append(
                        {"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "NO_SLOT"}
                    )
                    continue
                if target_notional > float(candidate["signal_amount"]) * baseline.DAILY_PARTICIPATION:
                    orders.append(
                        {"date": day, "symbol": symbol, "side": "BUY", "status": "PRETRADE_SKIPPED", "reason": "signal_capacity"}
                    )
                    continue
                quote = quote_lookup.get((day, symbol))
                raw_open = float(quote.get("raw_open") or 0.0) if quote else 0.0
                shares = account.affordable_shares(raw_open, target_notional, cash)
                gross = shares * raw_open
                reason = "zero_lot_or_cash" if shares <= 0 else _quote_rejection(quote, side="BUY", gross=gross)
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
                entry_cash_out = -cash_delta
                positions[symbol] = {
                    "symbol": symbol,
                    "raw_shares": shares,
                    "signal_date": candidate["signal_date"],
                    "entry_date": day,
                    "planned_exit_date": candidate["planned_exit_date"],
                    "entry_cash_out": entry_cash_out,
                    "benchmark_return": float(candidate["benchmark_return"]),
                    "last_raw_price": raw_open,
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
        max_cash_error = max(max_cash_error, abs(cash - cash_ledger))

    return {
        "orders": orders,
        "trades": trades,
        "completed_trades": completed,
        "ending_cash": cash,
        "ending_positions": list(positions.values()),
        "unresolved_exits": len(positions),
        "max_cash_reconciliation_error": max_cash_error,
    }


def _cluster_t(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return mean / (std / math.sqrt(len(values))) if std > 0 else None


def _two_sided_normal_p(t_value: float | None) -> float:
    if t_value is None or not math.isfinite(t_value):
        return 1.0
    return math.erfc(abs(t_value) / math.sqrt(2.0))


def holm_bonferroni(
    p_values: dict[int, float], *, family_alpha: float = FAMILY_ALPHA
) -> dict[int, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    output: dict[int, dict[str, Any]] = {}
    still_rejecting = True
    family_size = len(ordered)
    for rank, (alpha_id, p_value) in enumerate(ordered, start=1):
        threshold = family_alpha / (family_size - rank + 1)
        rejected = still_rejecting and p_value <= threshold
        if not rejected:
            still_rejecting = False
        output[alpha_id] = {
            "rank": rank,
            "p_value": p_value,
            "threshold": threshold,
            "rejected": rejected,
        }
    return output


def summarize_simulation(
    simulation: dict[str, Any], candidates: pl.DataFrame
) -> dict[str, Any]:
    completed = simulation["completed_trades"]
    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_day[row["entry_date"]].append(row)
    daily = [
        {
            "date": day,
            "net_return": float(np.mean([row["net_return"] for row in rows])),
            "excess_return": float(np.mean([row["excess_return"] for row in rows])),
        }
        for day, rows in sorted(by_day.items())
    ]
    yearly = []
    positive_contributions = []
    for year in range(2014, 2019):
        values = [row["excess_return"] for row in daily if row["date"].year == year]
        mean = float(np.mean(values)) if values else None
        contribution = float(sum(values)) if values else 0.0
        if contribution > 0:
            positive_contributions.append(contribution)
        yearly.append({"year": year, "mean_excess": mean, "excess_sum": contribution})
    buy_orders = [row for row in simulation["orders"] if row["side"] == "BUY"]
    buy_attempts = [row for row in buy_orders if row["status"] != "PRETRADE_SKIPPED"]
    buy_filled = sum(row["status"] == "FILLED" for row in buy_attempts)
    capacity_skips = sum(row.get("reason") in {"signal_capacity", "insufficient_capacity"} for row in buy_orders)
    planned = candidates.height
    excess_values = [row["excess_return"] for row in daily]
    t_value = _cluster_t(excess_values)
    return {
        "plan_days": candidates.get_column("entry_date").n_unique(),
        "planned_signals": planned,
        "completed_trades": len(completed),
        "buy_execution_rate": buy_filled / len(buy_attempts) if buy_attempts else 0.0,
        "sell_execution_rate": len(completed) / buy_filled if buy_filled else 0.0,
        "capacity_feasibility_rate": 1.0 - capacity_skips / planned if planned else 0.0,
        "unresolved_exits": simulation["unresolved_exits"],
        "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        "mean_daily_net_return": float(np.mean([row["net_return"] for row in daily])) if daily else None,
        "mean_daily_excess_return": float(np.mean(excess_values)) if excess_values else None,
        "excess_daily_cluster_t": t_value,
        "two_sided_normal_p": _two_sided_normal_p(t_value),
        "positive_years": sum((row["mean_excess"] or 0.0) > 0 for row in yearly),
        "max_positive_year_contribution": max(positive_contributions) / sum(positive_contributions) if positive_contributions else None,
        "yearly": yearly,
        "skip_reasons": dict(sorted(Counter(row.get("reason") for row in buy_orders if row.get("reason")).items())),
    }


def evaluate_formula(
    primary: dict[str, Any], scaling: dict[str, dict[str, Any]], holm: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "plan_days_at_least_900": primary["plan_days"] >= 900,
        "buy_execution_at_least_90pct": min(row["buy_execution_rate"] for row in scaling.values()) >= 0.90,
        "sell_execution_at_least_90pct": min(row["sell_execution_rate"] for row in scaling.values()) >= 0.90,
        "capacity_feasibility_at_least_95pct": min(row["capacity_feasibility_rate"] for row in scaling.values()) >= 0.95,
        "unresolved_exits_zero": max(row["unresolved_exits"] for row in scaling.values()) == 0,
        "mean_daily_net_at_least_20bps": (primary["mean_daily_net_return"] or -math.inf) >= 0.0020,
        "mean_daily_excess_at_least_15bps": (primary["mean_daily_excess_return"] or -math.inf) >= 0.0015,
        "excess_cluster_t_at_least_3_5": (primary["excess_daily_cluster_t"] or -math.inf) >= 3.5,
        "holm_bonferroni_5pct": bool(holm["rejected"]),
        "positive_years_at_least_4": primary["positive_years"] >= 4,
        "max_positive_year_contribution_at_most_35pct": (primary["max_positive_year_contribution"] or math.inf) <= 0.35,
    }
    passed = all(checks.values())
    return {"passed": passed, "verdict": "FREEZE_UNIQUE_CANDIDATE" if passed else "TERMINATE", "checks": checks}


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source = load_daily(data_dir)
    panel = prepare_alpha_panel(source, data_dir)
    del source
    gc.collect()
    context, eligible, amount, dates, symbols = panel_matrices(panel)
    raw_results: dict[int, dict[str, Any]] = {}
    p_values: dict[int, float] = {}
    for alpha_id in ALPHA_IDS:
        values = formulas.compute_alpha101(context, alpha_id)
        candidates = select_candidates(alpha_id, values, eligible, amount, dates, symbols)
        candidates = attach_execution_dates_and_benchmark(candidates, context, eligible, dates)
        quotes = build_execution_quotes(candidates, panel, dates)
        scaling: dict[str, dict[str, Any]] = {}
        for capital in CAPITALS:
            simulation = simulate_daily_account(
                candidates,
                quotes,
                [day for day in dates if DEVELOPMENT_START <= day <= DEVELOPMENT_END],
                initial_cash=capital,
            )
            scaling[str(int(capital))] = summarize_simulation(simulation, candidates)
        primary = scaling[str(int(CAPITALS[0]))]
        p_values[alpha_id] = primary["two_sided_normal_p"]
        raw_results[alpha_id] = {"primary": primary, "scaling": scaling}
        del values, candidates, quotes
        gc.collect()
    holm = holm_bonferroni(p_values)
    promoted = []
    results: dict[str, Any] = {}
    for alpha_id in ALPHA_IDS:
        decision = evaluate_formula(raw_results[alpha_id]["primary"], raw_results[alpha_id]["scaling"], holm[alpha_id])
        results[str(alpha_id)] = {**raw_results[alpha_id], "holm": holm[alpha_id], "decision": decision}
        if decision["passed"]:
            promoted.append(alpha_id)
    unique_candidate = None
    if promoted:
        unique_candidate = min(promoted, key=lambda alpha_id: (p_values[alpha_id], alpha_id))
    payload = {
        "schema_version": "p0-alpha101-long-only-screen-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "warmup_start": WARMUP_START,
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "confirmation_read": False,
            "validation_read": False,
            "stress_read": False,
        },
        "method": {
            "formula_ids": ALPHA_IDS,
            "primary_capital": CAPITALS[0],
            "scaling_capitals": CAPITALS,
            "benchmark": "same_signal_day_eligible_universe_median_gross_next_open_to_following_open",
            "p_value": "two_sided_normal_approximation_from_entry_day_cluster_t",
            "holm_family_alpha": FAMILY_ALPHA,
        },
        "data": {
            "trading_days": len(dates),
            "symbols": len(symbols),
            "eligible_cells": int(eligible.sum()),
        },
        "results": results,
        "formulas_passing_all_development_gates": promoted,
        "unique_candidate_frozen_for_confirmation": unique_candidate,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "promoted": promoted,
                "unique_candidate": unique_candidate,
                "strict_qualified_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backend" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "data" / "research" / "p0_alpha101_long_only_screen.json")
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
