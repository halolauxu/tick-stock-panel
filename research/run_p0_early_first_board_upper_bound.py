"""Evaluate an optimistic fill upper bound for early first-board entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from research import run_p0_emotion_limit_up_study as emotion  # noqa: E402
from research.run_p0_forecast_drift_development import (  # noqa: E402
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    STAMP_TAX_CURRENT,
    STAMP_TAX_CUT,
    STAMP_TAX_OLD,
)

CONTEXT_START = date(2025, 8, 1)
SIGNAL_START = date(2025, 8, 27)
SIGNAL_END = date(2026, 7, 31)
EXECUTION_END = date(2026, 8, 31)
CAPITAL_TIERS = (200_000, 300_000, 500_000, 1_000_000)
MAX_POSITIONS = 4
LOT_SIZE = 100
PARTICIPATION_RATE = 0.01
MIN_PREVIOUS_AMOUNT = 50_000_000.0
MAX_EXIT_DELAY = 20
TICK = 0.005


def stamp_tax_rate(day: date) -> float:
    return STAMP_TAX_OLD if day < STAMP_TAX_CUT else STAMP_TAX_CURRENT


def commission(notional: float) -> float:
    return max(5.0, round(notional * COMMISSION_PCT, 2))


def prepare_context(data_dir: Path) -> pl.DataFrame:
    raw = emotion.load_daily_panel(
        data_dir,
        end=EXECUTION_END,
        start=CONTEXT_START,
    )
    pit = emotion.attach_point_in_time_universe(raw, data_dir)
    panel = emotion.prepare_market_panel(pit).sort(["symbol", "date"])
    return panel.with_columns(
        pl.col("amount").shift(1).over("symbol").alias("previous_amount"),
        (pl.col("close") / pl.col("raw_close")).alias("adj_factor"),
    ).with_columns(
        (
            pl.col("_adjacent")
            & ~pl.col("_is_excluded")
            & ~pl.col("_prev_limit_up")
            & pl.col("_reference_close").is_between(3.0, 100.0, closed="both")
            & (pl.col("previous_amount") >= MIN_PREVIOUS_AMOUNT)
        ).alias("universe_eligible")
    )


def detect_early_first_board(
    rows: list[dict[str, Any]],
    limit_up: float,
) -> dict[str, Any] | None:
    """Return the first causal 09:31-10:30 seal, excluding prior touches."""
    if not rows or not math.isfinite(limit_up) or limit_up <= 0:
        return None
    ordered = sorted(rows, key=lambda row: row["datetime"])
    for row in ordered:
        clock = row["datetime"].time()
        if clock > time(10, 30):
            break
        high = float(row.get("high") or 0.0)
        close = float(row.get("close") or 0.0)
        if clock == time(9, 30) and high >= limit_up - TICK:
            return None
        if clock < time(9, 31):
            continue
        if high >= limit_up - TICK:
            if close < limit_up - TICK:
                return None
            volume = float(row.get("volume") or 0.0)
            amount = float(row.get("amount") or 0.0)
            if volume <= 0 or amount <= 0:
                return None
            return {
                "signal_datetime": row["datetime"],
                "entry_price": limit_up,
                "signal_volume": volume,
                "signal_amount": amount,
            }
    return None


def _minute_path(data_dir: Path, day: date) -> Path:
    return data_dir / "kline_minute" / f"date={day.isoformat()}" / "part.parquet"


def load_events(
    data_dir: Path,
    context: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = context.filter(
        pl.col("date").is_between(SIGNAL_START, SIGNAL_END, closed="both")
        & pl.col("universe_eligible")
    )
    daily_groups = {
        key[0] if isinstance(key, tuple) else key: group
        for key, group in candidates.partition_by("date", as_dict=True).items()
    }
    events: list[dict[str, Any]] = []
    missing: list[date] = []
    for day, daily in sorted(daily_groups.items()):
        path = _minute_path(data_dir, day)
        if not path.is_file():
            missing.append(day)
            continue
        symbols = daily.get_column("symbol").to_list()
        minute = (
            pl.read_parquet(path)
            .filter(
                pl.col("symbol").is_in(symbols)
                & (pl.col("datetime").dt.time() <= time(10, 30))
            )
            .select(
                "symbol",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            )
            .sort(["symbol", "datetime"])
        )
        minute_groups = {
            key[0] if isinstance(key, tuple) else key: group.to_dicts()
            for key, group in minute.partition_by("symbol", as_dict=True).items()
        }
        for source in daily.to_dicts():
            detected = detect_early_first_board(
                minute_groups.get(source["symbol"], []),
                float(source["limit_up_price"]),
            )
            if detected is None:
                continue
            events.append(
                {
                    **detected,
                    "date": day,
                    "symbol": source["symbol"],
                    "previous_amount": source["previous_amount"],
                    "adj_factor": source["adj_factor"],
                    "adjusted_close": source["close"],
                }
            )
    ranked: list[dict[str, Any]] = []
    event_frame = pl.DataFrame(events) if events else pl.DataFrame()
    if not event_frame.is_empty():
        ranked = (
            event_frame.sort(
                ["date", "signal_datetime", "previous_amount", "symbol"],
                descending=[False, False, True, False],
            )
            .with_columns(pl.int_range(pl.len()).over("date").alias("daily_rank"))
            .filter(pl.col("daily_rank") < MAX_POSITIONS)
            .to_dicts()
        )
    return ranked, {
        "eligible_stock_days": candidates.height,
        "detected_events": len(events),
        "selected_events": len(ranked),
        "missing_minute_partitions": [value.isoformat() for value in missing],
    }


def load_exit_windows(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "kline_minute").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if SIGNAL_START <= day <= EXECUTION_END:
            paths.append(path)
    if not paths:
        raise ValueError("minute partitions are required")
    return (
        pl.scan_parquet(sorted(paths), hive_partitioning=False)
        .filter(
            (pl.col("datetime").dt.hour() == 9)
            & pl.col("datetime").dt.minute().is_between(30, 34, closed="both")
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .group_by("symbol", "date")
        .agg(
            pl.col("amount").sum().alias("window_amount"),
            pl.col("volume").sum().alias("window_volume"),
            pl.col("high").max().alias("window_high"),
            pl.len().alias("window_minutes"),
        )
        .with_columns(
            (pl.col("window_amount") / (pl.col("window_volume") * 100.0)).alias(
                "window_vwap"
            )
        )
        .collect(engine="streaming")
    )


@dataclass
class Position:
    symbol: str
    entry_date: date
    due_index: int
    adj_units: float
    entry_notional: float
    entry_fee: float


def _drawdown(values: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def simulate_account(
    capital: float,
    events: list[dict[str, Any]],
    context: pl.DataFrame,
    exit_windows: pl.DataFrame,
    *,
    signal_dates: set[date] | None = None,
) -> dict[str, Any]:
    if signal_dates:
        signal_calendar = sorted(signal_dates)
        full_calendar = (
            context.select("date", "_global_index")
            .unique()
            .sort("_global_index")
            .to_dicts()
        )
        index_by_signal_date = {
            row["date"]: int(row["_global_index"]) for row in full_calendar
        }
        first_index = index_by_signal_date[signal_calendar[0]]
        last_index = index_by_signal_date[signal_calendar[-1]] + MAX_EXIT_DELAY + 1
    else:
        first_index = None
        last_index = None
    daily_rows = context.filter(
        pl.col("date").is_between(SIGNAL_START, EXECUTION_END, closed="both")
    )
    if first_index is not None and last_index is not None:
        daily_rows = daily_rows.filter(
            pl.col("_global_index").is_between(
                first_index,
                last_index,
                closed="both",
            )
        )
    daily_rows = daily_rows.select(
        "symbol",
        "date",
        "_global_index",
        "close",
        "adj_factor",
        "limit_down_price",
    )
    rows = daily_rows.to_dicts()
    daily = {(row["symbol"], row["date"]): row for row in rows}
    calendar_rows = (
        daily_rows.select("date", "_global_index")
        .unique()
        .sort("_global_index")
        .to_dicts()
    )
    calendar = [row["date"] for row in calendar_rows]
    index_by_date = {row["date"]: int(row["_global_index"]) for row in calendar_rows}
    quotes = {(row["symbol"], row["date"]): row for row in exit_windows.to_dicts()}
    event_groups: dict[date, list[dict[str, Any]]] = {}
    for event in events:
        if signal_dates is not None and event["date"] not in signal_dates:
            continue
        event_groups.setdefault(event["date"], []).append(event)

    cash = float(capital)
    positions: dict[str, Position] = {}
    last_marks: dict[str, float] = {}
    curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rejected = Counter()
    candidate_orders = 0
    entry_capacity_ok = 0

    for day in calendar:
        current_index = index_by_date[day]
        for symbol in sorted(list(positions)):
            position = positions[symbol]
            if current_index < position.due_index:
                continue
            ctx = daily.get((symbol, day))
            quote = quotes.get((symbol, day))
            if ctx is None or quote is None:
                continue
            if int(quote["window_minutes"]) < 5:
                continue
            vwap = float(quote["window_vwap"] or 0.0)
            amount = float(quote["window_amount"] or 0.0)
            high = float(quote["window_high"] or 0.0)
            limit_down = float(ctx["limit_down_price"] or 0.0)
            factor = float(ctx["adj_factor"] or 0.0)
            raw_shares = position.adj_units * factor
            if vwap <= 0 or amount <= 0 or factor <= 0:
                continue
            if high <= limit_down + TICK:
                continue
            executable_notional = raw_shares * vwap
            if executable_notional > amount * PARTICIPATION_RATE:
                continue
            exit_price = vwap * (1.0 - SLIPPAGE_PCT)
            gross = raw_shares * exit_price
            exit_fee = commission(gross) + gross * stamp_tax_rate(day)
            cash += gross - exit_fee
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position.entry_date,
                    "exit_date": day,
                    "entry_notional": position.entry_notional,
                    "exit_notional": gross,
                    "fees": position.entry_fee + exit_fee,
                    "net_pnl": gross
                    - exit_fee
                    - position.entry_notional
                    - position.entry_fee,
                    "exit_delay": current_index - position.due_index,
                }
            )
            positions.pop(symbol)
            last_marks.pop(symbol, None)

        def equity_at_close() -> float:
            value = cash
            for symbol, position in positions.items():
                ctx = daily.get((symbol, day))
                if ctx is not None and float(ctx["close"] or 0.0) > 0:
                    last_marks[symbol] = position.adj_units * float(ctx["close"])
                value += last_marks.get(symbol, position.entry_notional)
            return value

        for event in event_groups.get(day, []):
            candidate_orders += 1
            if event["symbol"] in positions:
                rejected["already_held"] += 1
                continue
            if len(positions) >= MAX_POSITIONS:
                rejected["position_limit"] += 1
                continue
            equity = equity_at_close()
            target = equity / MAX_POSITIONS
            price = float(event["entry_price"])
            shares = math.floor(target / price / LOT_SIZE) * LOT_SIZE
            if shares <= 0:
                rejected["zero_lot"] += 1
                continue
            notional = shares * price
            fee = commission(notional)
            while shares > 0 and notional + fee > cash:
                shares -= LOT_SIZE
                notional = shares * price
                fee = commission(notional) if shares > 0 else 0.0
            if shares <= 0:
                rejected["cash"] += 1
                continue
            if notional > float(event["signal_amount"]) * PARTICIPATION_RATE:
                rejected["signal_capacity"] += 1
                continue
            entry_capacity_ok += 1
            factor = float(event["adj_factor"])
            if factor <= 0:
                rejected["invalid_factor"] += 1
                continue
            cash -= notional + fee
            due_index = current_index + 1
            positions[event["symbol"]] = Position(
                symbol=event["symbol"],
                entry_date=day,
                due_index=due_index,
                adj_units=shares / factor,
                entry_notional=notional,
                entry_fee=fee,
            )
            last_marks[event["symbol"]] = (
                shares / factor * float(event["adjusted_close"])
            )

        curve.append({"date": day, "equity": equity_at_close(), "cash": cash})

    equities = [float(row["equity"]) for row in curve]
    final_equity = equities[-1] if equities else capital
    span = max(len(curve), 1)
    annualized = (final_equity / capital) ** (252.0 / span) - 1.0
    capacity_checks = entry_capacity_ok + rejected["signal_capacity"]
    return {
        "initial_capital": capital,
        "final_equity": final_equity,
        "total_return": final_equity / capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _drawdown(equities),
        "candidate_orders": candidate_orders,
        "entry_fills": len(trades) + len(positions),
        "completed_trades": len(trades),
        "open_positions": len(positions),
        "entry_capacity_feasible_rate": (
            entry_capacity_ok / capacity_checks if capacity_checks else None
        ),
        "rejections": dict(sorted(rejected.items())),
        "total_fees": sum(float(row["fees"]) for row in trades),
        "net_realized_pnl": sum(float(row["net_pnl"]) for row in trades),
        "curve_start": curve[0]["date"].isoformat() if curve else None,
        "curve_end": curve[-1]["date"].isoformat() if curve else None,
    }


def evaluate(data_dir: Path) -> dict[str, Any]:
    context = prepare_context(data_dir)
    events, audit = load_events(data_dir, context)
    windows = load_exit_windows(data_dir)
    signal_days = sorted(
        context.filter(
            pl.col("date").is_between(SIGNAL_START, SIGNAL_END, closed="both")
        )
        .get_column("date")
        .unique()
        .to_list()
    )
    midpoint = len(signal_days) // 2
    halves = {
        "first_half": set(signal_days[:midpoint]),
        "second_half": set(signal_days[midpoint:]),
    }
    full = {
        str(capital): simulate_account(capital, events, context, windows)
        for capital in CAPITAL_TIERS
    }
    split = {
        name: simulate_account(
            200_000,
            events,
            context,
            windows,
            signal_dates=dates,
        )
        for name, dates in halves.items()
    }
    event_counts = {
        name: sum(event["date"] in dates for event in events)
        for name, dates in halves.items()
    }
    checks = {
        "each_half_at_least_80_events": all(
            value >= 80 for value in event_counts.values()
        ),
        "each_half_at_least_40_fills": all(
            value["entry_fills"] >= 40 for value in split.values()
        ),
        "each_half_20w_annualized_over_50pct": all(
            value["annualized_return"] > 0.50 for value in split.values()
        ),
        "all_full_capitals_annualized_over_50pct": all(
            value["annualized_return"] > 0.50 for value in full.values()
        ),
        "full_20w_drawdown_within_35pct": full["200000"]["max_drawdown"] >= -0.35,
        "full_20w_capacity_at_least_90pct": (
            (full["200000"]["entry_capacity_feasible_rate"] or 0.0) >= 0.90
        ),
        "all_accounts_flat": all(
            value["open_positions"] == 0 for value in full.values()
        ),
        "no_missing_signal_partitions": not audit["missing_minute_partitions"],
    }
    return {
        "study": "p0_early_first_board_upper_bound",
        "fill_model": "UPPER_BOUND",
        "signal_start": SIGNAL_START.isoformat(),
        "signal_end": SIGNAL_END.isoformat(),
        "execution_end": EXECUTION_END.isoformat(),
        "audit": audit,
        "event_counts": event_counts,
        "full_accounts": full,
        "split_20w_accounts": split,
        "checks": checks,
        "verdict": "ADVANCE_TO_L2" if all(checks.values()) else "TERMINATE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print("sha256", hashlib.sha256((payload + "\n").encode()).hexdigest())


if __name__ == "__main__":
    main()
