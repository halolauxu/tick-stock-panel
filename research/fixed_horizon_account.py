"""Reusable point-in-time A-share account with a fixed trading-day horizon."""

from __future__ import annotations

import sys
from collections import defaultdict
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


def prepare_candidates(
    candidates: pl.DataFrame,
    trading_dates: list[date],
    *,
    holding_trading_days: int,
    maximum_exit_delay: int,
) -> pl.DataFrame:
    calendar = pl.DataFrame(
        {
            "entry_date": trading_dates,
            "entry_index": list(range(len(trading_dates))),
        }
    )
    last_entry_index = len(trading_dates) - holding_trading_days - maximum_exit_delay - 1
    return (
        candidates.join(calendar, on="entry_date", how="inner")
        .filter(pl.col("entry_index") <= last_entry_index)
        .with_columns(
            (pl.col("entry_index") + holding_trading_days).alias(
                "planned_exit_index"
            ),
            pl.col("cap_rank").alias("rank"),
        )
        .sort(["entry_index", "rank", "symbol"])
    )


def prepare_quotes(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    data_dir: Path,
) -> pl.DataFrame:
    symbols = candidates.get_column("symbol").unique().to_list()
    return account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )


def _quote_rejection(
    quote: dict[str, Any] | None,
    *,
    side: str,
    gross: float,
) -> str | None:
    if quote is None or quote.get("raw_open") is None:
        return "missing_market_data"
    if side == "BUY" and quote.get("is_excluded_name"):
        return "risk_warning"
    if not quote.get("volume") or float(quote["volume"]) <= 0:
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


def _yearly_metrics(
    daily: pl.DataFrame,
    *,
    start: date,
    end: date,
) -> tuple[int, list[dict[str, Any]]]:
    yearly: list[dict[str, Any]] = []
    positive_years = 0
    for year in range(start.year, end.year + 1):
        values = (
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return positive_years, yearly


def simulate(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    trading_dates: list[date],
    *,
    initial_cash: float,
    target_positions: int,
    holding_trading_days: int,
    maximum_exit_delay: int,
    period_start: date,
    period_end: date,
) -> dict[str, Any]:
    prepared = prepare_candidates(
        candidates,
        trading_dates,
        holding_trading_days=holding_trading_days,
        maximum_exit_delay=maximum_exit_delay,
    )
    candidate_groups = {
        int(key[0] if isinstance(key, tuple) else key): frame.sort(["rank", "symbol"])
        for key, frame in prepared.partition_by("entry_index", as_dict=True).items()
    }
    quote_lookup = {
        (row["date"], row["symbol"]): row for row in quotes.to_dicts()
    }
    positions: dict[str, dict[str, Any]] = {}
    cash = float(initial_cash)
    cash_ledger = float(initial_cash)
    max_cash_error = 0.0
    orders: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    position_id = 0
    stale_run = 0
    longest_stale_run = 0
    stale_position_days = 0

    for trade_index, trade_date in enumerate(trading_dates):
        for symbol, position in positions.items():
            quote = quote_lookup.get((trade_date, symbol))
            if quote and quote.get("close") is not None:
                position["last_mark"] = float(quote["close"])

        pre_open_equity = cash
        for symbol, position in positions.items():
            quote = quote_lookup.get((trade_date, symbol))
            price = (
                float(quote["open"])
                if quote and quote.get("open") is not None
                else float(position["last_mark"])
            )
            pre_open_equity += float(position["units"]) * price

        for symbol in list(positions):
            position = positions[symbol]
            if trade_index < int(position["planned_exit_index"]) or position.get(
                "terminal_exit_failure"
            ):
                continue
            quote = quote_lookup.get((trade_date, symbol))
            adjusted_open = float(quote.get("open") or 0.0) if quote else 0.0
            gross = float(position["units"]) * adjusted_open
            reason = _quote_rejection(quote, side="SELL", gross=gross)
            delay = trade_index - int(position["planned_exit_index"])
            status = "REJECTED" if reason else "FILLED"
            if reason and delay >= maximum_exit_delay:
                status = "UNRESOLVED"
                position["terminal_exit_failure"] = reason
            order = {
                "date": trade_date,
                "symbol": symbol,
                "side": "SELL",
                "status": status,
                "reason": reason,
                "position_id": position["position_id"],
                "planned_holding_days": holding_trading_days,
                "exit_delay_days": delay,
            }
            orders.append(order)
            if reason:
                continue
            commission_fee = account.commission(gross)
            stamp_tax = gross * (
                baseline.STAMP_TAX_OLD
                if trade_date < baseline.STAMP_TAX_CUT
                else baseline.STAMP_TAX_CURRENT
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
            del positions[symbol]

        target_notional = pre_open_equity / target_positions if pre_open_equity > 0 else 0.0
        slots = max(0, target_positions - len(positions))
        rows = candidate_groups.get(trade_index)
        if rows is not None:
            for candidate in rows.to_dicts():
                if slots <= 0:
                    break
                symbol = str(candidate["symbol"])
                if symbol in positions:
                    orders.append(
                        {
                            "date": trade_date,
                            "symbol": symbol,
                            "side": "BUY",
                            "status": "PRETRADE_SKIPPED",
                            "reason": "already_held",
                            "rank": int(candidate["rank"]),
                        }
                    )
                    continue
                if target_notional > float(candidate["signal_amount"]) * baseline.DAILY_PARTICIPATION:
                    orders.append(
                        {
                            "date": trade_date,
                            "symbol": symbol,
                            "side": "BUY",
                            "status": "PRETRADE_SKIPPED",
                            "reason": "signal_capacity",
                            "rank": int(candidate["rank"]),
                        }
                    )
                    continue
                quote = quote_lookup.get((trade_date, symbol))
                raw_open = float(quote.get("raw_open") or 0.0) if quote else 0.0
                shares = account.affordable_shares(raw_open, target_notional, cash)
                gross = shares * raw_open
                reason = (
                    "zero_lot_or_cash"
                    if shares <= 0
                    else _quote_rejection(quote, side="BUY", gross=gross)
                )
                order = {
                    "date": trade_date,
                    "signal_date": candidate["date"],
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "REJECTED" if reason else "FILLED",
                    "reason": reason,
                    "rank": int(candidate["rank"]),
                }
                orders.append(order)
                if reason:
                    continue
                commission_fee = account.commission(gross)
                slippage = gross * baseline.SLIPPAGE_PCT
                cash_delta = -(gross + commission_fee + slippage)
                cash += cash_delta
                cash_ledger += cash_delta
                position_id += 1
                adjusted_open = float(quote["open"])
                positions[symbol] = {
                    "position_id": position_id,
                    "symbol": symbol,
                    "entry_date": trade_date,
                    "entry_index": trade_index,
                    "planned_exit_index": trade_index + holding_trading_days,
                    "raw_shares": shares,
                    "units": gross / adjusted_open,
                    "last_mark": float(quote["close"]),
                    "terminal_exit_failure": None,
                }
                order.update(
                    position_id=position_id,
                    planned_exit_index=trade_index + holding_trading_days,
                    raw_shares=shares,
                    gross=gross,
                    commission=commission_fee,
                    stamp_tax=0.0,
                    slippage=slippage,
                    cash_delta=cash_delta,
                )
                slots -= 1

        close_equity = cash
        stale_today = 0
        for symbol, position in positions.items():
            quote = quote_lookup.get((trade_date, symbol))
            if quote and quote.get("close") is not None:
                mark = float(quote["close"])
                position["last_mark"] = mark
            else:
                mark = float(position["last_mark"])
                stale_today += 1
            close_equity += float(position["units"]) * mark
        stale_position_days += stale_today
        stale_run = stale_run + 1 if stale_today else 0
        longest_stale_run = max(longest_stale_run, stale_run)
        daily_rows.append(
            {
                "date": trade_date,
                "cash": cash,
                "equity": close_equity,
                "position_count": len(positions),
                "cash_ratio": cash / close_equity if close_equity > 0 else 1.0,
            }
        )
        max_cash_error = max(max_cash_error, abs(cash - cash_ledger))

    daily = pl.DataFrame(daily_rows, infer_schema_length=None).with_columns(
        (
            pl.col("equity") / pl.col("equity").shift(1).fill_null(initial_cash)
            - 1.0
        ).alias("daily_return")
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    positive_years, yearly = _yearly_metrics(
        daily, start=period_start, end=period_end
    )
    filled_orders = [row for row in orders if row["status"] == "FILLED"]
    total_costs = sum(
        float(row.get("commission") or 0.0)
        + float(row.get("stamp_tax") or 0.0)
        + float(row.get("slippage") or 0.0)
        for row in filled_orders
    )
    hold_counts = defaultdict(int)
    for row in orders:
        if row["side"] == "SELL" and row["status"] == "FILLED":
            hold_counts[int(row["planned_holding_days"]) + int(row["exit_delay_days"])] += 1
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
        "execution": account.execution_summary(orders),
        "integrity": {
            "stale_position_days": stale_position_days,
            "longest_stale_trading_days": longest_stale_run,
            "ending_unresolved_positions": len(positions),
            "max_cash_reconciliation_error": max_cash_error,
            "holding_days_distribution": dict(sorted(hold_counts.items())),
            "maximum_completed_holding_days": max(hold_counts, default=None),
        },
        "account": {
            "ending_equity": float(daily.get_column("equity")[-1]),
            "ending_cash": cash,
            "ending_positions": len(positions),
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "max_position_count": daily.get_column("position_count").max(),
            "trade_count": len(filled_orders),
            "total_gross_turnover": sum(
                float(row.get("gross") or 0.0) for row in filled_orders
            ),
            "total_costs": total_costs,
        },
    }
