"""Run the frozen development-only main-board IPO first-open W-shape study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.price_limits import (  # noqa: E402
    polars_is_risk_warning_name,
    polars_limit_price,
)
from research.run_p0_microcap_account import (  # noqa: E402
    _buy_rejection,
    _sell_rejection,
    affordable_shares,
    commission,
)
from research.run_p0_microcap_baseline import (  # noqa: E402
    COMMISSION_PCT,
    DAILY_PARTICIPATION,
    SLIPPAGE_PCT,
    STAMP_TAX_OLD,
)

SIGNAL_START = date(2014, 6, 16)
DEVELOPMENT_END = date(2016, 12, 31)
PANEL_END = date(2017, 1, 31)
DEVELOPMENT_LIST_END = DEVELOPMENT_END
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
MIN_PRIOR_LIMIT_UPS = 3
MAX_LISTING_TRADING_DAYS = 35
INTRADAY_THRESHOLD = 0.05
TARGET_POSITIONS = 2
HOLD_TRADING_DAYS = 5
MAX_EXIT_DELAY = 5
LOT_SIZE = 100

ARMS = ("OPEN_STRENGTH", "OPEN_SELLOFF")


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_git_commit() -> str:
    configured = os.environ.get("RESEARCH_GIT_COMMIT", "").strip()
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_market_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    source = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "close",
            "volume",
            "amount",
            "raw_close",
            "consecutive_limit_ups",
        )
        .filter(
            pl.col("date").is_between(SIGNAL_START, PANEL_END, closed="both")
            & pl.col("symbol").str.contains(
                r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"
            )
        )
        .collect(engine="streaming")
    )
    return attach_point_in_time_security(source, data_dir)


def attach_point_in_time_security(panel: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    research = data_dir / "research"
    universe_path = research / "historical_stock_universe_all_a.parquet"
    names_path = research / "historical_stock_names_all_a.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError("all-A point-in-time security master is required")
    universe = (
        pl.read_parquet(universe_path)
        .with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "list_date", "delist_date")
    )
    names = (
        pl.read_parquet(names_path)
        .with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "name", "start_date", "end_date")
        .sort(["symbol", "start_date"])
    )
    return (
        panel.join(universe, on="symbol", how="left")
        .filter(
            pl.col("list_date").is_not_null()
            & (pl.col("date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("date") <= pl.col("delist_date"))
            )
        )
        .sort(["symbol", "date"])
        .join_asof(
            names,
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (
                pl.col("name").is_not_null()
                & (
                    pl.col("end_date").is_null()
                    | (pl.col("date") <= pl.col("end_date"))
                )
            ).alias("name_status_known")
        )
        .with_columns(
            (
                pl.col("name_status_known")
                & (
                    polars_is_risk_warning_name(pl.col("name").fill_null(""))
                    | pl.col("name").fill_null("").str.contains("退", literal=True)
                    | pl.col("name")
                    .fill_null("")
                    .str.contains("重新上市", literal=True)
                )
            ).alias("excluded_name")
        )
        .drop("delist_date", "start_date", "end_date")
    )


def prepare_panel(panel: pl.DataFrame) -> pl.DataFrame:
    calendar = panel.select("date").unique().sort("date").with_row_index("trade_index")
    work = panel.join(calendar, on="date", how="left").sort(["symbol", "date"])
    work = work.with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_factor"),
        pl.col("trade_index").shift(1).over("symbol").alias("_prior_index"),
        pl.col("close").shift(1).over("symbol").alias("_prior_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_prior_raw_close"),
        (
            pl.col("close").shift(1).over("symbol")
            / pl.col("raw_close").shift(1).over("symbol")
        ).alias("_prior_factor"),
        pl.col("consecutive_limit_ups")
        .shift(1)
        .over("symbol")
        .fill_null(0)
        .alias("prior_consecutive_limit_ups"),
        pl.int_range(pl.len()).over("symbol").alias("listing_trade_index"),
    ).with_columns(
        (pl.col("trade_index") == pl.col("_prior_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_factor")).alias("raw_open"),
    )
    factor_changed = (pl.col("_factor") - pl.col("_prior_factor")).abs() > 1e-6
    work = work.with_columns(
        pl.when(factor_changed)
        .then(pl.col("_prior_close"))
        .otherwise(pl.col("_prior_raw_close"))
        .alias("reference_close"),
        (pl.col("close") / pl.col("open") - 1.0).alias("intraday_return"),
    ).with_columns(
        polars_limit_price(pl.col("reference_close"), pl.lit(0.10), up=True).alias(
            "limit_up_price"
        ),
        polars_limit_price(pl.col("reference_close"), pl.lit(0.10), up=False).alias(
            "limit_down_price"
        ),
    )
    return work.select(
        "symbol",
        "date",
        "trade_index",
        "list_date",
        "listing_trade_index",
        "open",
        "raw_open",
        "close",
        "raw_close",
        "volume",
        "amount",
        "consecutive_limit_ups",
        "prior_consecutive_limit_ups",
        "intraday_return",
        "limit_up_price",
        "limit_down_price",
        "excluded_name",
        "name_status_known",
        "_adjacent",
    )


def build_first_open_events(panel: pl.DataFrame) -> pl.DataFrame:
    eligible = panel.filter(
        pl.col("symbol").str.starts_with("00") | pl.col("symbol").str.starts_with("60")
    ).filter(
        pl.col("list_date").is_between(
            SIGNAL_START, DEVELOPMENT_LIST_END, closed="both"
        )
        & pl.col("date").is_between(SIGNAL_START, DEVELOPMENT_END, closed="both")
        & pl.col("listing_trade_index").is_between(
            1, MAX_LISTING_TRADING_DAYS - 1, closed="both"
        )
        & (pl.col("prior_consecutive_limit_ups") >= MIN_PRIOR_LIMIT_UPS)
        & ~pl.col("excluded_name")
        & (pl.col("volume").fill_null(0) > 0)
        & (pl.col("amount").fill_null(0) > 0)
        & pl.col("raw_open").is_finite()
        & pl.col("close").is_finite()
        & (pl.col("raw_open") < pl.col("limit_up_price") - 0.005)
    )
    return (
        eligible.sort(["symbol", "date"])
        .unique(subset=["symbol"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("date").alias("signal_date"),
            (pl.col("trade_index") + 1).alias("entry_index"),
            (pl.col("trade_index") + 1 + HOLD_TRADING_DAYS).alias("planned_exit_index"),
        )
        .select(
            "symbol",
            "signal_date",
            "trade_index",
            "entry_index",
            "planned_exit_index",
            "list_date",
            "listing_trade_index",
            "prior_consecutive_limit_ups",
            "intraday_return",
            "raw_open",
            "raw_close",
            "amount",
            "name_status_known",
        )
        .sort(["signal_date", "symbol"])
    )


def rank_candidates(events: pl.DataFrame, arm: str, *, control: bool) -> pl.DataFrame:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    selected = events
    if not control:
        selected = selected.filter(
            pl.col("intraday_return") >= INTRADAY_THRESHOLD
            if arm == "OPEN_STRENGTH"
            else pl.col("intraday_return") <= -INTRADAY_THRESHOLD
        )
    descending = arm == "OPEN_STRENGTH"
    return (
        selected.sort(
            ["entry_index", "intraday_return", "symbol"],
            descending=[False, descending, False],
        )
        .with_columns(pl.int_range(pl.len()).over("entry_index").alias("rank"))
        .sort(["entry_index", "rank", "symbol"])
    )


def build_quote_lookup(
    panel: pl.DataFrame, symbols: list[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    wanted = panel.filter(pl.col("symbol").is_in(symbols))
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in wanted.iter_rows(named=True):
        output[(str(row["symbol"]), int(row["trade_index"]))] = {
            **row,
            "exact_quote": True,
            "entry_amount": row.get("amount"),
            "entry_volume": row.get("volume"),
            "is_excluded_name": row.get("excluded_name"),
        }
    return output


def _stamp_tax(_: date) -> float:
    return STAMP_TAX_OLD


def _empty_order_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.Date,
            "trade_index": pl.Int64,
            "symbol": pl.String,
            "side": pl.String,
            "status": pl.String,
            "reason": pl.String,
        }
    )


def simulate_account(
    ranked: pl.DataFrame,
    quote_lookup: dict[tuple[str, int], dict[str, Any]],
    calendar_dates: list[date],
    *,
    initial_capital: float,
    start_index: int,
    end_index: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    candidates: dict[int, list[dict[str, Any]]] = {}
    if not ranked.is_empty():
        for key, frame in ranked.partition_by("entry_index", as_dict=True).items():
            index = int(key[0] if isinstance(key, tuple) else key)
            candidates[index] = frame.sort(["rank", "symbol"]).to_dicts()
    positions: dict[str, dict[str, Any]] = {}
    cash = float(initial_capital)
    orders: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    position_id = 0

    for trade_index in range(start_index, end_index + 1):
        trade_date = calendar_dates[trade_index]
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            if quote and quote.get("close") is not None:
                position["last_mark"] = float(quote["close"])

        pre_open_equity = cash
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            price = (
                float(quote["open"])
                if quote and quote.get("open") is not None
                else float(position["last_mark"])
            )
            pre_open_equity += float(position["units"]) * price

        for symbol in list(positions):
            position = positions[symbol]
            if trade_index < int(position["planned_exit_index"]) or position.get(
                "unresolved"
            ):
                continue
            quote = quote_lookup.get((symbol, trade_index))
            reason = (
                "missing_adjusted_price"
                if quote and (quote.get("open") is None or quote.get("close") is None)
                else _sell_rejection(position, quote)
            )
            delay = trade_index - int(position["planned_exit_index"])
            if reason:
                unresolved = delay >= MAX_EXIT_DELAY
                position["unresolved"] = unresolved
                orders.append(
                    {
                        "date": trade_date,
                        "trade_index": trade_index,
                        "position_id": position["position_id"],
                        "symbol": symbol,
                        "side": "SELL",
                        "status": "UNRESOLVED" if unresolved else "REJECTED",
                        "reason": reason,
                        "delay": delay,
                    }
                )
                continue
            gross = float(position["units"]) * float(quote["open"])
            commission_fee = commission(gross)
            stamp_tax = gross * _stamp_tax(trade_date)
            slippage = gross * SLIPPAGE_PCT
            cash_delta = gross - commission_fee - stamp_tax - slippage
            cash += cash_delta
            net_pnl = cash_delta - float(position["cash_out"])
            record = {
                "date": trade_date,
                "trade_index": trade_index,
                "position_id": position["position_id"],
                "symbol": symbol,
                "side": "SELL",
                "status": "FILLED",
                "reason": None,
                "delay": delay,
                "gross": gross,
                "commission": commission_fee,
                "stamp_tax": stamp_tax,
                "slippage": slippage,
                "cash_delta": cash_delta,
                "net_pnl": net_pnl,
                "net_trade_return": net_pnl / float(position["cash_out"]),
                "entry_date": position["entry_date"],
                "signal_date": position["signal_date"],
            }
            orders.append(record)
            completed.append(record)
            del positions[symbol]

        slots = max(0, TARGET_POSITIONS - len(positions))
        target_notional = (
            pre_open_equity / TARGET_POSITIONS if pre_open_equity > 0 else 0.0
        )
        for candidate in candidates.get(trade_index, []):
            if slots <= 0:
                break
            symbol = str(candidate["symbol"])
            if symbol in positions:
                orders.append(
                    {
                        "date": trade_date,
                        "trade_index": trade_index,
                        "symbol": symbol,
                        "side": "BUY",
                        "status": "PRETRADE_SKIPPED",
                        "reason": "already_held",
                        "rank": int(candidate["rank"]),
                    }
                )
                continue
            quote = quote_lookup.get((symbol, trade_index))
            raw_open = (
                float(quote["raw_open"])
                if quote and quote.get("raw_open") is not None
                else 0.0
            )
            shares = affordable_shares(
                raw_open,
                target_notional,
                cash,
                lot_size=LOT_SIZE,
            )
            gross = shares * raw_open
            reason = "zero_lot_or_cash" if shares <= 0 else _buy_rejection(quote, gross)
            if (
                reason is None
                and quote
                and (quote.get("open") is None or quote.get("close") is None)
            ):
                reason = "missing_adjusted_price"
            order = {
                "date": trade_date,
                "trade_index": trade_index,
                "symbol": symbol,
                "side": "BUY",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
                "rank": int(candidate["rank"]),
                "signal_date": candidate["signal_date"],
                "intraday_return": candidate["intraday_return"],
            }
            if reason:
                orders.append(order)
                continue
            commission_fee = commission(gross)
            slippage = gross * SLIPPAGE_PCT
            cash_out = gross + commission_fee + slippage
            cash -= cash_out
            position_id += 1
            adjusted_open = float(quote["open"])
            positions[symbol] = {
                "position_id": position_id,
                "symbol": symbol,
                "entry_date": trade_date,
                "entry_index": trade_index,
                "planned_exit_index": trade_index + HOLD_TRADING_DAYS,
                "signal_date": candidate["signal_date"],
                "units": gross / adjusted_open,
                "raw_shares": shares,
                "cash_out": cash_out,
                "last_mark": float(quote["close"]),
                "unresolved": False,
            }
            order.update(
                position_id=position_id,
                planned_exit_index=trade_index + HOLD_TRADING_DAYS,
                raw_shares=shares,
                gross=gross,
                commission=commission_fee,
                stamp_tax=0.0,
                slippage=slippage,
                cash_delta=-cash_out,
            )
            orders.append(order)
            slots -= 1

        close_equity = cash
        stale_positions = 0
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            if quote and quote.get("close") is not None:
                mark = float(quote["close"])
                position["last_mark"] = mark
            else:
                mark = float(position["last_mark"])
                stale_positions += 1
            close_equity += float(position["units"]) * mark
        daily.append(
            {
                "date": trade_date,
                "trade_index": trade_index,
                "cash": cash,
                "equity": close_equity,
                "positions": len(positions),
                "stale_positions": stale_positions,
            }
        )

    daily_frame = pl.DataFrame(daily, infer_schema_length=None).sort("trade_index")
    daily_frame = daily_frame.with_columns(
        (
            pl.col("equity") / pl.col("equity").shift(1).fill_null(initial_capital)
            - 1.0
        ).alias("daily_return")
    )
    attempted_buys = [
        row
        for row in orders
        if row["side"] == "BUY" and row["status"] != "PRETRADE_SKIPPED"
    ]
    filled_buys = [row for row in attempted_buys if row["status"] == "FILLED"]
    total_return = float(daily_frame["equity"][-1]) / initial_capital - 1.0
    annualized = (
        (1.0 + total_return) ** (252.0 / daily_frame.height) - 1.0
        if total_return > -1.0
        else None
    )
    peak = float(initial_capital)
    max_drawdown = 0.0
    for equity in daily_frame["equity"].to_list():
        peak = max(peak, float(equity))
        max_drawdown = min(max_drawdown, float(equity) / peak - 1.0)

    yearly = []
    positive_years = 0
    for year in range(SIGNAL_START.year, DEVELOPMENT_END.year + 1):
        scoped = daily_frame.filter(pl.col("date").dt.year() == year)
        returns = scoped["daily_return"].to_list()
        year_return = math.prod(1.0 + float(value) for value in returns) - 1.0
        positive_years += int(year_return > 0)
        yearly.append({"year": year, "return": year_return})

    cluster_values: dict[date, list[float]] = defaultdict(list)
    positive_year_pnl: dict[int, float] = defaultdict(float)
    for row in completed:
        cluster_values[row["signal_date"]].append(float(row["net_trade_return"]))
        if float(row["net_pnl"]) > 0:
            positive_year_pnl[row["date"].year] += float(row["net_pnl"])
    clustered = [statistics.fmean(values) for values in cluster_values.values()]
    cluster_mean = statistics.fmean(clustered) if clustered else None
    cluster_t = (
        cluster_mean / (statistics.stdev(clustered) / math.sqrt(len(clustered)))
        if cluster_mean is not None
        and len(clustered) >= 2
        and statistics.stdev(clustered) > 0
        else None
    )
    trade_returns = [float(row["net_trade_return"]) for row in completed]
    total_positive = sum(positive_year_pnl.values())
    largest_positive_year_share = (
        max(positive_year_pnl.values()) / total_positive if total_positive else None
    )

    cash_delta_by_index: dict[int, float] = defaultdict(float)
    for row in orders:
        if row["status"] == "FILLED":
            cash_delta_by_index[int(row["trade_index"])] += float(row["cash_delta"])
    reconciled_cash = float(initial_capital)
    max_ledger_error = 0.0
    for row in daily:
        reconciled_cash += cash_delta_by_index[int(row["trade_index"])]
        max_ledger_error = max(
            max_ledger_error, abs(reconciled_cash - float(row["cash"]))
        )

    # Any position still open at the frozen account boundary is unresolved,
    # even if a future coding error shortened the configured exit buffer.
    unresolved = len(positions)
    sell_resolution_rate = len(completed) / len(filled_buys) if filled_buys else 0.0
    all_filled = [row for row in orders if row["status"] == "FILLED"]
    summary = {
        "initial_capital": initial_capital,
        "ending_equity": float(daily_frame["equity"][-1]),
        "ending_cash": cash,
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": max_drawdown,
        "positive_years": positive_years,
        "yearly": yearly,
        "attempted_buys": len(attempted_buys),
        "filled_buys": len(filled_buys),
        "buy_execution_rate": len(filled_buys) / len(attempted_buys)
        if attempted_buys
        else 0.0,
        "completed_sells": len(completed),
        "sell_execution_rate": sell_resolution_rate,
        "unresolved_exits": unresolved,
        "ending_positions": len(positions),
        "mean_net_trade_return": statistics.fmean(trade_returns)
        if trade_returns
        else None,
        "signal_day_clusters": len(clustered),
        "signal_day_cluster_mean": cluster_mean,
        "signal_day_cluster_t": cluster_t,
        "largest_positive_year_share": largest_positive_year_share,
        "total_cost": sum(
            float(row.get("commission") or 0.0)
            + float(row.get("stamp_tax") or 0.0)
            + float(row.get("slippage") or 0.0)
            for row in all_filled
        ),
        "maximum_cash_ledger_error": max_ledger_error,
        "capacity_rejections": sum(
            row.get("reason") == "insufficient_capacity" for row in attempted_buys
        ),
        "buy_reject_reasons": dict(
            sorted(
                Counter(
                    row["reason"] for row in attempted_buys if row.get("reason")
                ).items()
            )
        ),
        "sell_reject_reasons": dict(
            sorted(
                Counter(
                    row["reason"]
                    for row in orders
                    if row["side"] == "SELL" and row.get("reason")
                ).items()
            )
        ),
        "return_observations": daily_frame.height,
    }
    records = (
        pl.DataFrame(orders, infer_schema_length=None)
        if orders
        else _empty_order_frame()
    )
    return summary, records


def market_benchmark(
    panel: pl.DataFrame, *, start_index: int, end_index: int
) -> dict[str, Any]:
    daily = (
        panel.sort(["symbol", "trade_index"])
        .with_columns(
            pl.col("trade_index").shift(1).over("symbol").alias("_prior_index"),
            pl.col("close").shift(1).over("symbol").alias("_prior_close"),
        )
        .filter(
            pl.col("trade_index").is_between(start_index, end_index, closed="both")
            & (pl.col("trade_index") == pl.col("_prior_index") + 1)
            & ~pl.col("excluded_name")
            & (pl.col("_prior_close") > 0)
            & pl.col("close").is_finite()
        )
        .with_columns((pl.col("close") / pl.col("_prior_close") - 1.0).alias("return"))
        .group_by("date")
        .agg(pl.col("return").mean().alias("return"))
        .sort("date")
    )
    values = [float(value) for value in daily["return"].to_list()]
    total = math.prod(1.0 + value for value in values) - 1.0
    return {
        "trading_days": len(values),
        "total_return": total,
        "annualized_return": (1.0 + total) ** (252.0 / len(values)) - 1.0,
    }


def evaluate_gate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    annualized = candidate.get("annualized_return")
    checks = {
        "annualized_above_50pct": (annualized or -math.inf) > 0.50,
        "market_excess_at_least_20pp": (
            (annualized or -math.inf) - (benchmark.get("annualized_return") or math.inf)
            >= 0.20
        ),
        "direction_control_increment_at_least_15pp": (
            (annualized or -math.inf) - (control.get("annualized_return") or math.inf)
            >= 0.15
        ),
        "max_drawdown_no_worse_than_30pct": candidate.get("max_drawdown", -math.inf)
        >= -0.30,
        "positive_years_at_least_2": candidate.get("positive_years", 0) >= 2,
        "completed_sells_at_least_40": candidate.get("completed_sells", 0) >= 40,
        "buy_execution_at_least_90pct": candidate.get("buy_execution_rate", 0.0)
        >= 0.90,
        "sell_execution_at_least_90pct": candidate.get("sell_execution_rate", 0.0)
        >= 0.90,
        "no_unresolved_exits": candidate.get("unresolved_exits", 1) == 0,
        "mean_net_trade_return_at_least_2pct": (
            candidate.get("mean_net_trade_return") or -math.inf
        )
        >= 0.02,
        "signal_day_cluster_t_at_least_2": (
            candidate.get("signal_day_cluster_t") or -math.inf
        )
        >= 2.0,
        "largest_positive_year_share_at_most_60pct": (
            candidate.get("largest_positive_year_share") or math.inf
        )
        <= 0.60,
        "cash_ledger_error_at_most_one_cent": candidate.get(
            "maximum_cash_ledger_error", math.inf
        )
        <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def run(data_dir: Path, output: Path, artifact_dir: Path) -> dict[str, Any]:
    attached = load_market_panel(data_dir)
    panel = prepare_panel(attached)
    calendar = panel.select("date", "trade_index").unique().sort("trade_index")
    calendar_dates = calendar["date"].to_list()
    start_index = int(calendar.filter(pl.col("date") >= SIGNAL_START)["trade_index"][0])
    development_end_index = int(
        calendar.filter(pl.col("date") <= DEVELOPMENT_END)["trade_index"][-1]
    )
    end_index = min(
        len(calendar_dates) - 1,
        development_end_index + HOLD_TRADING_DAYS + MAX_EXIT_DELAY + 1,
    )
    events = build_first_open_events(panel)
    if events.is_empty():
        raise RuntimeError("no development first-open events were generated")
    quote_lookup = build_quote_lookup(panel, events["symbol"].unique().to_list())
    benchmark = market_benchmark(panel, start_index=start_index, end_index=end_index)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    events_path = artifact_dir / "development_first_open_events.parquet"
    events.write_parquet(events_path)
    artifacts = {str(events_path): _sha256(events_path)}
    accounts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    ranked_counts: dict[str, Any] = {}
    for arm in ARMS:
        ranked_candidate = rank_candidates(events, arm, control=False)
        ranked_control = rank_candidates(events, arm, control=True)
        ranked_counts[arm] = {
            "candidate_events": ranked_candidate.height,
            "candidate_signal_days": ranked_candidate["signal_date"].n_unique(),
            "control_events": ranked_control.height,
            "control_signal_days": ranked_control["signal_date"].n_unique(),
        }
        arm_accounts: dict[str, Any] = {}
        for capital in INITIAL_CAPITALS:
            candidate, candidate_orders = simulate_account(
                ranked_candidate,
                quote_lookup,
                calendar_dates,
                initial_capital=capital,
                start_index=start_index,
                end_index=end_index,
            )
            control, control_orders = simulate_account(
                ranked_control,
                quote_lookup,
                calendar_dates,
                initial_capital=capital,
                start_index=start_index,
                end_index=end_index,
            )
            candidate_path = (
                artifact_dir / f"{arm.lower()}_candidate_orders_{int(capital)}.parquet"
            )
            control_path = (
                artifact_dir / f"{arm.lower()}_control_orders_{int(capital)}.parquet"
            )
            candidate_orders.write_parquet(candidate_path)
            control_orders.write_parquet(control_path)
            artifacts[str(candidate_path)] = _sha256(candidate_path)
            artifacts[str(control_path)] = _sha256(control_path)
            arm_accounts[str(int(capital))] = {
                "candidate": candidate,
                "direction_control": control,
            }
        accounts[arm] = arm_accounts
        decisions[arm] = evaluate_gate(
            arm_accounts["200000"]["candidate"],
            arm_accounts["200000"]["direction_control"],
            benchmark,
        )

    passing = [arm for arm in ARMS if decisions[arm]["passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda arm: (
                accounts[arm]["200000"]["candidate"]["annualized_return"],
                arm == "OPEN_STRENGTH",
            ),
        )
    decision = {
        "passed_development": selected is not None,
        "selected_arm": selected,
        "counts_toward_50pct_goal": False,
        "strict_qualified_count": 0,
        "next_step": (
            f"freeze_{selected.lower()}_for_independent_validation"
            if selected
            else "terminate_ipo_first_open_wshape_family"
        ),
    }
    payload = {
        "schema_version": "p0-ipo-first-open-wshape-development-v1",
        "contract_frozen": "2026-08-31",
        "implementation_git_commit": implementation_git_commit(),
        "period": {
            "signal_start": SIGNAL_START,
            "development_end": DEVELOPMENT_END,
            "exit_buffer_end": calendar_dates[end_index],
            "validation_signal_or_account_read": False,
            "pressure_read": False,
        },
        "assumptions": {
            "main_board_prefixes": ["00", "60"],
            "minimum_prior_consecutive_limit_ups": MIN_PRIOR_LIMIT_UPS,
            "maximum_listing_trading_days": MAX_LISTING_TRADING_DAYS,
            "intraday_threshold": INTRADAY_THRESHOLD,
            "entry": "signal t close, buy t+1 open",
            "exit": "sell t+6 open; five complete open-to-open intervals",
            "target_positions": TARGET_POSITIONS,
            "lot_size": LOT_SIZE,
            "maximum_exit_delay": MAX_EXIT_DELAY,
            "commission_pct": COMMISSION_PCT,
            "minimum_commission_cny": 5.0,
            "slippage_pct_per_side": SLIPPAGE_PCT,
            "stamp_tax_pct": STAMP_TAX_OLD,
            "daily_participation": DAILY_PARTICIPATION,
            "direction_control": "same arm ranking, remove only the +/-5% state filter",
        },
        "data": {
            "panel_first_date": calendar_dates[0],
            "panel_last_date": calendar_dates[-1],
            "development_account_days": end_index - start_index + 1,
            "first_open_events": events.height,
            "first_open_symbols": events["symbol"].n_unique(),
            "first_open_events_with_unknown_name_status": events.filter(
                ~pl.col("name_status_known")
            ).height,
            "ranked": ranked_counts,
        },
        "benchmark": benchmark,
        "accounts": accounts,
        "arm_decisions": decisions,
        "artifacts": artifacts,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": _sha256(output)},
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
        default=Path("/app/data/research/p0_ipo_first_open_wshape_development.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/app/data/research/ipo_first_open_wshape"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output, args.artifact_dir)


if __name__ == "__main__":
    main()
