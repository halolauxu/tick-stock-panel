"""Run the preregistered P0-B2 CNY 300k micro-cap cash-account study."""
from __future__ import annotations

import argparse
import gc
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

import run_p0_microcap_baseline as baseline  # noqa: E402

from app.price_limits import polars_limit_price  # noqa: E402

INITIAL_CASH = 300_000.0
TARGET_POSITIONS = 20
LOT_SIZE = 100
MIN_COMMISSION = 5.0


def commission(gross: float) -> float:
    return max(MIN_COMMISSION, gross * baseline.COMMISSION_PCT)


def affordable_shares(
    raw_open: float,
    target: float,
    cash: float,
    *,
    lot_size: int = LOT_SIZE,
) -> int:
    if raw_open <= 0 or target <= 0 or cash <= MIN_COMMISSION:
        return 0
    budget = min(target, cash - MIN_COMMISSION)
    shares = math.floor(
        budget / (raw_open * (1.0 + baseline.SLIPPAGE_PCT)) / lot_size
    ) * lot_size
    while shares > 0:
        gross = shares * raw_open
        total = gross * (1.0 + baseline.SLIPPAGE_PCT) + commission(gross)
        if total <= cash + 1e-9:
            return shares
        shares -= lot_size
    return 0


def attach_quote_names(source: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    names = (
        pl.read_parquet(
            data_dir / "research" / "historical_stock_names_all_a.parquet"
        )
        .with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "name", "start_date", "end_date")
        .sort(["symbol", "start_date"])
    )
    return (
        source.sort(["symbol", "date"])
        .join_asof(
            names,
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            pl.when(
                pl.col("end_date").is_null()
                | (pl.col("date") <= pl.col("end_date"))
            )
            .then(pl.col("name"))
            .otherwise(None)
            .alias("name")
        )
        .drop("start_date", "end_date")
    )


def prepare_quote_panel(source: pl.DataFrame) -> pl.DataFrame:
    dates = source.select("date").unique().sort("date").with_row_index("_index")
    work = source.join(dates, on="date", how="left").sort(["symbol", "date"])
    name_is_excluded = (
        pl.col("name")
        .fill_null("")
        .str.to_uppercase()
        .str.contains(r"(?:\*?ST|退)")
        if "name" in work.columns
        else pl.lit(False)
    )
    main_board_st = name_is_excluded & (
        pl.col("symbol").str.starts_with("00")
        | pl.col("symbol").str.starts_with("60")
    )
    work = work.with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_adj_factor"),
        pl.col("_index").shift(1).over("symbol").alias("_prev_index"),
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_prev_raw_close"),
        (
            pl.col("close").shift(1).over("symbol")
            / pl.col("raw_close").shift(1).over("symbol")
        ).alias("_prev_adj_factor"),
    ).with_columns(
        (pl.col("_index") == pl.col("_prev_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_adj_factor")).alias("raw_open"),
    )
    factor_changed = (
        pl.col("_adj_factor") - pl.col("_prev_adj_factor")
    ).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(
            pl.when(factor_changed)
            .then(pl.col("_prev_close"))
            .otherwise(pl.col("_prev_raw_close"))
        )
        .otherwise(None)
        .alias("_reference_close"),
        pl.when(main_board_st)
        .then(pl.lit(0.05))
        .otherwise(baseline.price_limit_pct())
        .alias("_limit_pct"),
        name_is_excluded.alias("is_excluded_name"),
    ).with_columns(
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=True)
        .alias("limit_up_price"),
        polars_limit_price(
            pl.col("_reference_close"), pl.col("_limit_pct"), up=False
        ).alias("limit_down_price"),
    )
    return work.select(
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


def build_signal_candidates(panel: pl.DataFrame) -> pl.DataFrame:
    dates = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("entry_date"))
    )
    weekly = (
        dates.with_columns(pl.col("date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )
    return (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"),
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
        .select(
            "date",
            "entry_date",
            "symbol",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def build_execution_grid(
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
) -> pl.DataFrame:
    symbols = candidates.select("symbol").unique().sort("symbol")
    entry_dates = candidates.select("entry_date").unique().sort("entry_date")
    grid = symbols.join(entry_dates, how="cross").sort(["symbol", "entry_date"])
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
            (pl.col("quote_date") == pl.col("entry_date")).alias("exact_quote")
        )
        .sort(["entry_date", "symbol"])
    )


def _partition_rows(frame: pl.DataFrame, column: str) -> dict[date, pl.DataFrame]:
    output: dict[date, pl.DataFrame] = {}
    for key, group in frame.partition_by(column, as_dict=True).items():
        day = key[0] if isinstance(key, tuple) else key
        output[day] = group
    return output


def _sell_rejection(position: dict[str, Any], quote: dict[str, Any] | None) -> str | None:
    if quote is None or not quote.get("exact_quote"):
        return "missing_market_data"
    if not quote.get("entry_volume") or quote["entry_volume"] <= 0:
        return "suspended"
    raw_open = quote.get("raw_open")
    limit_down = quote.get("limit_down_price")
    if raw_open is None or raw_open <= 0:
        return "missing_open"
    if limit_down is not None and raw_open <= limit_down + 0.005:
        return "limit_down"
    gross = position["units"] * quote["open"]
    if gross > float(quote.get("entry_amount") or 0.0) * baseline.DAILY_PARTICIPATION:
        return "insufficient_capacity"
    return None


def _buy_rejection(
    quote: dict[str, Any] | None,
    gross: float,
) -> str | None:
    if quote is None or not quote.get("exact_quote"):
        return "missing_market_data"
    if quote.get("is_excluded_name"):
        return "became_risk_warning"
    if not quote.get("entry_volume") or quote["entry_volume"] <= 0:
        return "suspended"
    raw_open = quote.get("raw_open")
    limit_up = quote.get("limit_up_price")
    if raw_open is None or raw_open <= 0:
        return "missing_open"
    if limit_up is not None and raw_open >= limit_up - 0.005:
        return "limit_up"
    capacity = float(
        quote.get("entry_amount") or 0.0
    ) * baseline.DAILY_PARTICIPATION
    if gross > capacity:
        return "insufficient_capacity"
    return None


def simulate_account(
    candidates: pl.DataFrame,
    execution_grid: pl.DataFrame,
    *,
    initial_cash: float = INITIAL_CASH,
    target_positions: int = TARGET_POSITIONS,
    action_dates: list[date] | None = None,
    stamp_tax_rate: float | None = None,
    lot_size: int = LOT_SIZE,
    delist_dates: dict[str, date] | None = None,
    delist_settlement_status: str = "DELISTED_WRITE_OFF",
    settle_only_after_delist_date: bool = False,
    delist_recovery_per_raw_share: float = 0.0,
) -> dict[str, Any]:
    candidate_groups = _partition_rows(candidates, "entry_date")
    quote_groups = _partition_rows(execution_grid, "entry_date")
    positions: dict[str, dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    settlements: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    cash = float(initial_cash)
    cash_ledger = float(initial_cash)
    position_id = 0
    max_cash_error = 0.0

    scheduled_dates = action_dates or sorted(candidate_groups)
    for entry_date in scheduled_dates:
        candidate_frame = candidate_groups.get(entry_date)
        candidate_rows = (
            candidate_frame.sort(["cap_rank", "symbol"]).to_dicts()
            if candidate_frame is not None
            else []
        )
        quote_frame = quote_groups.get(entry_date, pl.DataFrame())
        quotes = {
            row["symbol"]: row
            for row in quote_frame.to_dicts()
        }
        for symbol in list(positions):
            delist_date = (delist_dates or {}).get(symbol)
            if (
                delist_date is None
                or entry_date < delist_date
                or (
                    settle_only_after_delist_date
                    and entry_date == delist_date
                )
            ):
                continue
            position = positions.pop(symbol)
            last_book_value = position["units"] * position["last_mark"]
            recovery_value = (
                position["raw_shares"] * delist_recovery_per_raw_share
            )
            cash += recovery_value
            cash_ledger += recovery_value
            settlements.append(
                {
                    "date": entry_date,
                    "effective_delist_date": delist_date,
                    "symbol": symbol,
                    "status": delist_settlement_status,
                    "raw_shares": position["raw_shares"],
                    "recovery_value": recovery_value,
                    "last_book_value": last_book_value,
                    "recognized_loss": recovery_value - last_book_value,
                }
            )
            intervals.append(
                {
                    "position_id": position["position_id"],
                    "symbol": symbol,
                    "units": position["units"],
                    "start_date": position["start_date"],
                    "end_date": entry_date,
                }
            )
        for symbol, position in positions.items():
            quote = quotes.get(symbol)
            if quote and quote.get("close") is not None:
                position["last_mark"] = float(quote["close"])

        pre_open_equity = cash
        for symbol, position in positions.items():
            quote = quotes.get(symbol)
            price = (
                float(quote["open"])
                if quote and quote.get("exact_quote") and quote.get("open")
                else float(position["last_mark"])
            )
            pre_open_equity += position["units"] * price

        desired = {
            row["symbol"] for row in candidate_rows[:target_positions]
        }
        sold_today: set[str] = set()
        for symbol in list(positions):
            if symbol in desired:
                continue
            position = positions[symbol]
            quote = quotes.get(symbol)
            reason = _sell_rejection(position, quote)
            order = {
                "date": entry_date,
                "symbol": symbol,
                "side": "SELL",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
            }
            if reason:
                orders.append(order)
                continue
            gross = position["units"] * float(quote["open"])
            commission_fee = commission(gross)
            effective_stamp_tax = (
                stamp_tax_rate
                if stamp_tax_rate is not None
                else (
                    baseline.STAMP_TAX_OLD
                    if entry_date < baseline.STAMP_TAX_CUT
                    else baseline.STAMP_TAX_CURRENT
                )
            )
            stamp_tax = gross * effective_stamp_tax
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
            orders.append(order)
            trades.append(order.copy())
            intervals.append(
                {
                    "position_id": position["position_id"],
                    "symbol": symbol,
                    "units": position["units"],
                    "start_date": position["start_date"],
                    "end_date": entry_date,
                }
            )
            sold_today.add(symbol)
            del positions[symbol]

        slots = max(0, target_positions - len(positions))
        target_notional = pre_open_equity / target_positions
        for candidate in candidate_rows:
            if slots <= 0:
                break
            symbol = candidate["symbol"]
            if symbol in positions or symbol in sold_today:
                continue
            signal_capacity = float(
                candidate.get("signal_amount") or 0.0
            ) * baseline.DAILY_PARTICIPATION
            if target_notional > signal_capacity:
                orders.append(
                    {
                        "date": entry_date,
                        "signal_date": candidate["date"],
                        "symbol": symbol,
                        "side": "BUY",
                        "status": "PRETRADE_SKIPPED",
                        "reason": "signal_capacity",
                        "rank": candidate["cap_rank"],
                        "target_notional": target_notional,
                        "signal_capacity": signal_capacity,
                    }
                )
                continue
            quote = quotes.get(symbol)
            raw_open = float(quote["raw_open"]) if quote and quote.get("raw_open") else 0.0
            shares = affordable_shares(
                raw_open, target_notional, cash, lot_size=lot_size
            )
            gross = shares * raw_open
            reason = (
                "zero_lot_or_cash"
                if shares <= 0
                else _buy_rejection(quote, gross)
            )
            order = {
                "date": entry_date,
                "signal_date": candidate["date"],
                "symbol": symbol,
                "side": "BUY",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
                "rank": candidate["cap_rank"],
            }
            if reason:
                orders.append(order)
                continue
            commission_fee = commission(gross)
            slippage = gross * baseline.SLIPPAGE_PCT
            cash_delta = -(gross + commission_fee + slippage)
            cash += cash_delta
            cash_ledger += cash_delta
            adjusted_open = float(quote["open"])
            units = gross / adjusted_open
            position_id += 1
            positions[symbol] = {
                "position_id": position_id,
                "symbol": symbol,
                "units": units,
                "raw_shares": shares,
                "start_date": entry_date,
                "last_mark": float(quote["close"]),
            }
            order.update(
                raw_shares=shares,
                gross=gross,
                commission=commission_fee,
                stamp_tax=0.0,
                slippage=slippage,
                cash_delta=cash_delta,
            )
            orders.append(order)
            trades.append(order.copy())
            slots -= 1

        max_cash_error = max(max_cash_error, abs(cash - cash_ledger))
        snapshots.append(
            {
                "date": entry_date,
                "cash": cash,
                "position_count": len(positions),
                "pre_open_equity": pre_open_equity,
                "target_notional": target_notional,
            }
        )

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
        "settlements": settlements,
        "snapshots": snapshots,
        "intervals": intervals,
        "ending_positions": list(positions.values()),
        "ending_cash": cash,
        "max_cash_reconciliation_error": max_cash_error,
    }


def build_daily_equity(
    simulation: dict[str, Any],
    quotes: pl.DataFrame,
    all_dates: list[date],
    *,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    date_index = {day: index for index, day in enumerate(all_dates)}
    wanted: list[dict[str, Any]] = []
    final_end = len(all_dates)
    for interval in simulation["intervals"]:
        start = date_index[interval["start_date"]]
        end_day = interval.get("end_date")
        end = date_index.get(end_day, final_end)
        for index in range(start, end):
            wanted.append(
                {
                    "position_id": interval["position_id"],
                    "symbol": interval["symbol"],
                    "date": all_dates[index],
                    "units": interval["units"],
                }
            )
    if wanted:
        marks = (
            pl.DataFrame(wanted, infer_schema_length=None)
            .join(
                quotes.select("symbol", "date", "close"),
                on=["symbol", "date"],
                how="left",
            )
            .sort(["position_id", "date"])
            .with_columns(
                pl.col("close").is_null().alias("stale"),
                pl.col("close").forward_fill().over("position_id").alias("mark_price"),
            )
            .with_columns(
                (pl.col("units") * pl.col("mark_price")).alias("market_value")
            )
        )
        if marks.get_column("mark_price").null_count():
            raise ValueError("position interval starts without a reliable mark")
        daily_positions = marks.group_by("date").agg(
            pl.col("market_value").sum().alias("position_value"),
            pl.col("position_id").n_unique().alias("position_count"),
            pl.col("stale").sum().alias("stale_positions"),
        )
        stale_rows = marks.filter(pl.col("stale")).height
        longest_stale = 0
        streaks: dict[int, int] = {}
        for row in marks.select("position_id", "stale").to_dicts():
            key = row["position_id"]
            streaks[key] = streaks.get(key, 0) + 1 if row["stale"] else 0
            longest_stale = max(longest_stale, streaks[key])
    else:
        daily_positions = pl.DataFrame(
            schema={
                "date": pl.Date,
                "position_value": pl.Float64,
                "position_count": pl.UInt32,
                "stale_positions": pl.UInt32,
            }
        )
        stale_rows = 0
        longest_stale = 0

    cash_events = pl.DataFrame(
        simulation["snapshots"], infer_schema_length=None
    ).select("date", "cash")
    daily = (
        pl.DataFrame({"date": all_dates})
        .join(cash_events, on="date", how="left")
        .with_columns(pl.col("cash").forward_fill().fill_null(initial_cash))
        .join(daily_positions, on="date", how="left")
        .with_columns(
            pl.col("position_value").fill_null(0.0),
            pl.col("position_count").fill_null(0),
            pl.col("stale_positions").fill_null(0),
        )
        .with_columns(
            (pl.col("cash") + pl.col("position_value")).alias("equity")
        )
        .with_columns(
            (
                pl.col("equity")
                / pl.col("equity").shift(1).fill_null(initial_cash)
                - 1.0
            ).alias("daily_return"),
            (pl.col("cash") / pl.col("equity")).alias("cash_ratio"),
        )
    )
    final_date = all_dates[-1]
    ending_symbols = {
        row["symbol"] for row in simulation["ending_positions"]
    }
    final_exact = set(
        quotes.filter(pl.col("date") == pl.lit(final_date))
        .get_column("symbol")
        .to_list()
    )
    return daily, {
        "stale_position_days": stale_rows,
        "longest_stale_trading_days": longest_stale,
        "ending_unresolved_positions": len(ending_symbols - final_exact),
    }


def _period_filter(period: str) -> pl.Expr:
    if period == "development":
        return pl.col("date") <= pl.lit(baseline.DEVELOPMENT_END)
    if period == "validation":
        return (pl.col("date") > pl.lit(baseline.DEVELOPMENT_END)) & (
            pl.col("date") <= pl.lit(baseline.VALIDATION_END)
        )
    return pl.col("date") > pl.lit(baseline.VALIDATION_END)


def account_period_metrics(
    daily: pl.DataFrame,
    weekly_market: pl.DataFrame,
) -> list[dict[str, Any]]:
    output = []
    for period in ("development", "validation", "known_stress"):
        scoped = daily.filter(_period_filter(period))
        market = weekly_market.filter(pl.col("period") == period)
        returns = scoped.get_column("daily_return").drop_nulls().to_list()
        account_total = baseline._compound(returns)
        account_annual = (
            (1.0 + account_total) ** (252.0 / len(returns)) - 1.0
            if returns and account_total is not None and account_total > -1.0
            else None
        )
        market_annual = baseline._annualized(
            market.get_column("market_net").to_list()
        )
        yearly = []
        positive_years = 0
        for year in sorted(scoped.get_column("date").dt.year().unique().to_list()):
            year_return = baseline._compound(
                scoped.filter(pl.col("date").dt.year() == year)
                .get_column("daily_return")
                .drop_nulls()
                .to_list()
            )
            positive_years += int(year_return is not None and year_return > 0)
            yearly.append({"year": year, "account_return": year_return})
        output.append(
            {
                "period": period,
                "trading_days": scoped.height,
                "account_total_return": account_total,
                "account_annualized": account_annual,
                "market_annualized": market_annual,
                "annualized_excess": (
                    account_annual - market_annual
                    if account_annual is not None and market_annual is not None
                    else None
                ),
                "account_max_drawdown": baseline._max_drawdown(returns),
                "positive_account_years": positive_years,
                "yearly": yearly,
            }
        )
    return output


def execution_summary(orders: list[dict[str, Any]]) -> dict[str, Any]:
    by_side: dict[str, Any] = {}
    for side in ("BUY", "SELL"):
        side_rows = [row for row in orders if row["side"] == side]
        pretrade = [
            row for row in side_rows if row["status"] == "PRETRADE_SKIPPED"
        ]
        scoped = [
            row for row in side_rows if row["status"] != "PRETRADE_SKIPPED"
        ]
        filled = sum(row["status"] == "FILLED" for row in scoped)
        reasons = Counter(
            row["reason"] for row in scoped if row.get("reason") is not None
        )
        by_side[side.lower()] = {
            "orders": len(scoped),
            "filled": filled,
            "execution_rate": filled / len(scoped) if scoped else 1.0,
            "pretrade_skipped": len(pretrade),
            "pretrade_skip_reasons": dict(
                sorted(Counter(row["reason"] for row in pretrade).items())
            ),
            "rejection_reasons": dict(sorted(reasons.items())),
        }
    return by_side


def account_summary(
    simulation: dict[str, Any],
    daily: pl.DataFrame,
) -> dict[str, Any]:
    trades = simulation["trades"]
    return {
        "ending_equity": daily.get_column("equity")[-1],
        "ending_cash": simulation["ending_cash"],
        "ending_positions": len(simulation["ending_positions"]),
        "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
        "max_position_count": daily.get_column("position_count").max(),
        "trade_count": len(trades),
        "total_gross_turnover": sum(float(row["gross"]) for row in trades),
        "total_costs": sum(
            float(row.get("commission") or 0.0)
            + float(row.get("stamp_tax") or 0.0)
            + float(row.get("slippage") or 0.0)
            for row in trades
        ),
    }


def worst_weeks(daily: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        daily.with_columns(pl.col("date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("date"),
            pl.col("equity").last().alias("equity"),
        )
        .sort("date")
        .with_columns(
            (pl.col("equity") / pl.col("equity").shift(1) - 1.0).alias(
                "weekly_return"
            )
        )
        .drop_nulls("weekly_return")
        .sort("weekly_return")
        .head(10)
        .to_dicts()
    )


def period_dates(all_dates: list[date], period: str) -> list[date]:
    if period == "development":
        return [day for day in all_dates if day <= baseline.DEVELOPMENT_END]
    if period == "validation":
        return [
            day
            for day in all_dates
            if baseline.DEVELOPMENT_END < day <= baseline.VALIDATION_END
        ]
    return [day for day in all_dates if day > baseline.VALIDATION_END]


def run_independent_account(
    period: str,
    candidates: pl.DataFrame,
    execution_grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
    weekly_market: pl.DataFrame,
    *,
    initial_cash: float = INITIAL_CASH,
) -> dict[str, Any]:
    scoped_dates = period_dates(all_dates, period)
    first_date = scoped_dates[0]
    last_date = scoped_dates[-1]
    scoped_candidates = candidates.filter(
        (pl.col("entry_date") >= pl.lit(first_date))
        & (pl.col("entry_date") <= pl.lit(last_date))
    )
    scoped_grid = execution_grid.filter(
        (pl.col("entry_date") >= pl.lit(first_date))
        & (pl.col("entry_date") <= pl.lit(last_date))
    )
    simulation = simulate_account(
        scoped_candidates,
        scoped_grid,
        initial_cash=initial_cash,
    )
    daily, stale = build_daily_equity(
        simulation,
        quotes,
        scoped_dates,
        initial_cash=initial_cash,
    )
    metric = next(
        row
        for row in account_period_metrics(daily, weekly_market)
        if row["period"] == period
    )
    integrity = {
        **stale,
        "max_cash_reconciliation_error": simulation[
            "max_cash_reconciliation_error"
        ],
    }
    return {
        "period": period,
        "first_date": first_date,
        "last_date": last_date,
        "metrics": metric,
        "execution": execution_summary(simulation["orders"]),
        "integrity": integrity,
        "account": account_summary(simulation, daily),
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
        "worst_weeks": worst_weeks(daily),
    }


def evaluate_gate(
    independent_accounts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    passed = True
    failures = []
    for period in ("validation", "known_stress"):
        account = independent_accounts[period]
        row = account["metrics"]
        checks = {
            f"{period}_annualized": (row.get("account_annualized") or -99.0)
            >= 0.15,
            f"{period}_excess": (row.get("annualized_excess") or -99.0) >= 0.10,
            f"{period}_positive_years": row.get("positive_account_years", 0) >= 2,
        }
        failures.extend(name for name, ok in checks.items() if not ok)
        passed = passed and all(checks.values())
        for side in ("buy", "sell"):
            ok = account["execution"][side]["execution_rate"] >= 0.80
            if not ok:
                failures.append(f"{period}_{side}_execution_rate")
            passed = passed and ok
        integrity = account["integrity"]
        integrity_ok = (
            integrity["ending_unresolved_positions"] == 0
            and integrity["max_cash_reconciliation_error"] <= 0.01
        )
        if not integrity_ok:
            failures.append(f"{period}_account_integrity")
        passed = passed and integrity_ok
    return {
        "verdict": "CONTINUE_TO_ESCAPE" if passed else "DOWNGRADE",
        "passed": passed,
        "failures": failures,
        "reason": (
            "30万元统一账户通过收益、执行和完整性门槛,只允许进入独立冻结的灾难逃生研究"
            if passed
            else "30万元统一账户没有同时通过冻结的收益、执行和完整性门槛"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path, *, end: date | None = None) -> dict[str, Any]:
    source = baseline.load_daily(data_dir, end=end)
    if source.is_empty():
        raise ValueError("no daily data")
    all_dates = source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    signal_panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    candidates = build_signal_candidates(signal_panel)
    observations = baseline.build_weekly_observations(signal_panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del signal_panel, observations
    gc.collect()

    source_quotes = baseline.load_daily(data_dir, end=end).filter(
        pl.col("symbol").is_in(candidate_symbols)
    )
    source_quotes = attach_quote_names(source_quotes, data_dir)
    quotes = prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = build_execution_grid(candidates, quotes)
    simulation = simulate_account(candidates, execution_grid)
    daily, stale_integrity = build_daily_equity(simulation, quotes, all_dates)
    metrics = account_period_metrics(daily, weekly_market)
    execution = execution_summary(simulation["orders"])
    integrity = {
        **stale_integrity,
        "max_cash_reconciliation_error": simulation[
            "max_cash_reconciliation_error"
        ],
    }
    continuous_account = {
        "period_metrics": metrics,
        "execution": execution,
        "integrity": integrity,
        "account": account_summary(simulation, daily),
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
        "worst_weeks": worst_weeks(daily),
    }
    independent_accounts = {
        period: run_independent_account(
            period,
            candidates,
            execution_grid,
            quotes,
            all_dates,
            weekly_market,
        )
        for period in ("development", "validation", "known_stress")
    }
    decision = evaluate_gate(independent_accounts)
    payload = {
        "schema_version": "p0-microcap-account-v2",
        "contract": {
            "initial_cash": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "lot_size": LOT_SIZE,
            "minimum_commission": MIN_COMMISSION,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "signal": "weekly_pit_total_market_cap_bottom_decile",
            "execution": "next_trade_day_open_sells_before_buys",
            "validation": "independent_cny_300k_accounts_by_period",
            "signal_capacity": "pretrade_skip_not_open_order",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "candidate_symbols": len(candidate_symbols),
            "signal_rows": candidates.height,
            "rebalance_days": candidates.get_column("entry_date").n_unique(),
        },
        "continuous_account": continuous_account,
        "independent_accounts": independent_accounts,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "data": payload["data"],
                "continuous_account": {
                    "period_metrics": metrics,
                    "execution": execution,
                    "integrity": integrity,
                    "account": continuous_account["account"],
                },
                "independent_accounts": {
                    period: {
                        "metrics": result["metrics"],
                        "execution": result["execution"],
                        "integrity": result["integrity"],
                        "account": result["account"],
                    }
                    for period, result in independent_accounts.items()
                },
                "decision": decision,
                "output": str(output),
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
        default=Path("/app/data/research/p0_microcap_account.json"),
    )
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    run(args.data_dir, args.output, end=args.end)


if __name__ == "__main__":
    main()
