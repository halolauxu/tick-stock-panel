"""Run the frozen T+0 ETF overnight-to-intraday reversal development test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, time
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

DEV_START = date(2025, 8, 1)
DEV_END = date(2025, 12, 31)
QDII_SYMBOLS = (
    "159920.SZ", "510900.SH", "513100.SH", "513500.SH", "513030.SH",
    "513600.SH", "159941.SZ", "513050.SH", "159954.SZ", "513000.SH",
    "513520.SH", "513880.SH", "513080.SH", "159822.SZ", "513300.SH",
)
GOLD_SYMBOLS = ("518880.SH", "518800.SH", "159934.SZ", "159937.SZ")
SYMBOLS = QDII_SYMBOLS + GOLD_SYMBOLS
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
EXPECTED_BARS = 241
MIN_PREVIOUS_AMOUNT = 20_000_000.0
MIN_ELIGIBLE = 8
POSITIONS = 3
ALLOCATION_PER_POSITION = 0.30
ENTRY_START = time(9, 31)
ENTRY_END = time(9, 35)
EXIT_START = time(14, 55)
MAX_VOLUME_PARTICIPATION = 0.01
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
TICK = 0.001
LOT = 100


def _day_rows(frame: pl.DataFrame) -> dict[time, dict[str, dict[str, Any]]]:
    rows: dict[time, dict[str, dict[str, Any]]] = {}
    for row in frame.sort(["datetime", "symbol"]).iter_rows(named=True):
        rows.setdefault(row["datetime"].time(), {})[row["symbol"]] = row
    return rows


def _complete_days(minutes: pl.DataFrame) -> dict[date, pl.DataFrame]:
    frame = minutes.with_columns(pl.col("datetime").dt.date().alias("date"))
    output: dict[date, pl.DataFrame] = {}
    for trade_date in sorted(frame["date"].unique().to_list()):
        day = frame.filter(pl.col("date") == trade_date).drop("date")
        counts = day.group_by("symbol").len()
        if (
            day.height == len(SYMBOLS) * EXPECTED_BARS
            and counts.height == len(SYMBOLS)
            and counts.filter(pl.col("len") != EXPECTED_BARS).is_empty()
        ):
            output[trade_date] = day
    return output


def build_signals(minutes: pl.DataFrame) -> list[dict[str, Any]]:
    days = _complete_days(minutes)
    dates = sorted(days)
    signals: list[dict[str, Any]] = []
    for previous_date, trade_date in pairwise(dates):
        previous = days[previous_date]
        current = days[trade_date]
        previous_rows = _day_rows(previous)
        current_rows = _day_rows(current)
        previous_close = previous_rows.get(time(15, 0), {})
        opening = current_rows.get(time(9, 30), {})
        amount_frame = previous.group_by("symbol").agg(pl.col("amount").sum())
        previous_amounts = dict(
            zip(
                amount_frame["symbol"].to_list(),
                amount_frame["amount"].to_list(),
                strict=True,
            )
        )
        eligible: list[tuple[str, float]] = []
        for symbol in SYMBOLS:
            prior = previous_close.get(symbol)
            today = opening.get(symbol)
            if prior is None or today is None:
                continue
            prior_close = float(prior["close"])
            open_price = float(today["open"])
            if (
                prior_close <= 0
                or open_price <= 0
                or float(today["amount"]) <= 0
                or float(previous_amounts.get(symbol, 0.0)) < MIN_PREVIOUS_AMOUNT
            ):
                continue
            eligible.append((symbol, open_price / prior_close - 1.0))
        if len(eligible) < MIN_ELIGIBLE:
            continue
        ranked = sorted(eligible, key=lambda item: (item[1], item[0]))
        eligible_symbols = [item[0] for item in ranked]
        for rank, (symbol, overnight_return) in enumerate(ranked[:POSITIONS], start=1):
            signals.append(
                {
                    "date": trade_date,
                    "previous_date": previous_date,
                    "symbol": symbol,
                    "rank": rank,
                    "overnight_return": overnight_return,
                    "eligible_funds": len(eligible),
                    "eligible_symbols": eligible_symbols,
                    "signal_price": float(opening[symbol]["close"]),
                }
            )
    return signals


def signal_diagnostics(minutes: pl.DataFrame, signals: list[dict[str, Any]]) -> dict[str, Any]:
    days = _complete_days(minutes)
    selected: list[float] = []
    excess: list[float] = []
    for trade_date, day in days.items():
        day_signals = [row for row in signals if row["date"] == trade_date]
        if not day_signals:
            continue
        rows = _day_rows(day)
        entry = rows.get(ENTRY_START, {})
        exit_rows = rows.get(EXIT_START, {})
        eligible_symbols = [
            symbol for symbol in day_signals[0]["eligible_symbols"]
            if symbol in entry
            and symbol in exit_rows
            and float(entry[symbol]["open"]) > 0
            and float(exit_rows[symbol]["open"]) > 0
        ]
        if len(eligible_symbols) < MIN_ELIGIBLE:
            continue
        universe_returns = [
            float(exit_rows[symbol]["open"]) / float(entry[symbol]["open"]) - 1.0
            for symbol in eligible_symbols
        ]
        benchmark = statistics.median(universe_returns)
        for signal in day_signals:
            symbol = signal["symbol"]
            if symbol not in entry or symbol not in exit_rows:
                continue
            value = float(exit_rows[symbol]["open"]) / float(entry[symbol]["open"]) - 1.0
            selected.append(value)
            excess.append(value - benchmark)
    return {
        "observations": len(selected),
        "mean_gross_return": statistics.fmean(selected) if selected else None,
        "mean_excess_return": statistics.fmean(excess) if excess else None,
        "positive_gross_share": sum(value > 0 for value in selected) / len(selected)
        if selected else None,
    }


def _commission(notional: float) -> float:
    return max(MIN_COMMISSION, notional * COMMISSION_RATE) if notional > 0 else 0.0


def _capacity(volume: float) -> int:
    return max(0, math.floor(volume * MAX_VOLUME_PARTICIPATION / LOT) * LOT)


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _monthly_returns(daily_equity: dict[date, float], initial: float) -> dict[str, float]:
    result: dict[str, float] = {}
    previous = initial
    for month in sorted({value.strftime("%Y-%m") for value in daily_equity}):
        dates = [value for value in daily_equity if value.strftime("%Y-%m") == month]
        ending = daily_equity[max(dates)]
        result[month] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return result


def simulate_account(
    minutes: pl.DataFrame, signals: list[dict[str, Any]], initial_capital: float
) -> dict[str, Any]:
    days = _complete_days(minutes)
    signals_by_date: dict[date, list[dict[str, Any]]] = {}
    for signal in signals:
        signals_by_date.setdefault(signal["date"], []).append(signal)
    cash = initial_capital
    expected_cash = initial_capital
    positions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    equity_marks = [initial_capital]
    daily_equity: dict[date, float] = {}
    total_commission = 0.0
    total_slippage = 0.0
    entered = 0
    carry_days = 0

    def mark(rows: dict[str, dict[str, Any]] | None = None) -> float:
        value = cash
        for symbol, position in positions.items():
            row = rows.get(symbol) if rows else None
            price = float(row["close"]) if row is not None else float(position["last_price"])
            position["last_price"] = price
            value += int(position["shares"]) * price
        return value

    def sell_window(
        trade_date: date,
        rows: dict[time, dict[str, dict[str, Any]]],
        start: time,
        status: str,
    ) -> None:
        nonlocal cash, expected_cash, total_commission, total_slippage
        sold_notional: dict[str, float] = {symbol: 0.0 for symbol in positions}
        for clock in sorted(value for value in rows if value >= start):
            clock_rows = rows[clock]
            for symbol in list(positions):
                position = positions[symbol]
                row = clock_rows.get(symbol)
                if row is None or position["shares"] <= 0:
                    continue
                position["last_price"] = float(row["close"])
                shares = min(int(position["shares"]), _capacity(float(row["volume"])))
                if shares <= 0:
                    continue
                raw_price = float(row["open"])
                price = max(0.0, raw_price - TICK)
                proceeds = shares * price
                cash += proceeds
                expected_cash += proceeds
                total_slippage += shares * (raw_price - price)
                sold_notional[symbol] += proceeds
                position["shares"] -= shares
                position["sale_proceeds"] += proceeds
                position["raw_sale_proceeds"] += shares * raw_price
            equity_marks.append(mark(clock_rows))
            if all(position["shares"] == 0 for position in positions.values()):
                break
        for symbol in list(positions):
            position = positions[symbol]
            sale = sold_notional[symbol]
            if sale > 0:
                fee = _commission(sale)
                cash -= fee
                expected_cash -= fee
                total_commission += fee
                position["exit_fee"] += fee
            if position["shares"] == 0:
                net_pnl = (
                    position["sale_proceeds"]
                    - position["exit_fee"]
                    - position["purchase_cost"]
                    - position["entry_fee"]
                )
                gross_pnl = position["raw_sale_proceeds"] - position["raw_purchase_cost"]
                records.append(
                    {
                        "signal_date": position["signal"]["date"],
                        "entry_date": position["entry_date"],
                        "exit_date": trade_date,
                        "symbol": symbol,
                        "rank": position["signal"]["rank"],
                        "overnight_return": position["signal"]["overnight_return"],
                        "status": status,
                        "shares": position["entry_shares"],
                        "gross_pnl": gross_pnl,
                        "cost": gross_pnl - net_pnl,
                        "net_pnl": net_pnl,
                    }
                )
                del positions[symbol]

    for trade_date in sorted(days):
        day = days[trade_date]
        rows = _day_rows(day)
        if positions:
            carry_days += 1
            sell_window(trade_date, rows, time(9, 30), "CLOSED_AFTER_OVERNIGHT")
            daily_equity[trade_date] = mark(rows.get(time(15, 0), {}))
            continue

        day_signals = sorted(signals_by_date.get(trade_date, []), key=lambda row: row["rank"])
        if not day_signals:
            daily_equity[trade_date] = cash
            continue
        starting_equity = cash
        for signal in day_signals:
            symbol = signal["symbol"]
            target_shares = (
                math.floor(
                    starting_equity * ALLOCATION_PER_POSITION
                    / max(float(signal["signal_price"]) + TICK, TICK)
                    / LOT
                )
                * LOT
            )
            if target_shares <= 0:
                records.append({**signal, "status": "REJECTED_CAPITAL", "shares": 0})
                continue
            position = {
                "signal": signal,
                "entry_date": trade_date,
                "shares": 0,
                "entry_shares": 0,
                "purchase_cost": 0.0,
                "raw_purchase_cost": 0.0,
                "entry_fee": 0.0,
                "sale_proceeds": 0.0,
                "raw_sale_proceeds": 0.0,
                "exit_fee": 0.0,
                "last_price": float(signal["signal_price"]),
                "target_shares": target_shares,
            }
            positions[symbol] = position

        for clock in sorted(value for value in rows if ENTRY_START <= value <= ENTRY_END):
            clock_rows = rows[clock]
            for symbol in list(positions):
                position = positions[symbol]
                remaining = int(position["target_shares"] - position["entry_shares"])
                row = clock_rows.get(symbol)
                if remaining <= 0 or row is None:
                    continue
                position["last_price"] = float(row["close"])
                raw_price = float(row["open"])
                price = raw_price + TICK
                shares = min(remaining, _capacity(float(row["volume"])))
                affordable = max(0, math.floor((cash - MIN_COMMISSION) / price / LOT) * LOT)
                shares = min(shares, affordable)
                if shares <= 0:
                    continue
                cost = shares * price
                cash -= cost
                expected_cash -= cost
                total_slippage += shares * (price - raw_price)
                position["shares"] += shares
                position["entry_shares"] += shares
                position["purchase_cost"] += cost
                position["raw_purchase_cost"] += shares * raw_price
            equity_marks.append(mark(clock_rows))

        for symbol in list(positions):
            position = positions[symbol]
            if position["entry_shares"] <= 0:
                records.append(
                    {
                        **position["signal"],
                        "status": "REJECTED_NO_MINUTE_CAPACITY",
                        "shares": 0,
                    }
                )
                del positions[symbol]
                continue
            entered += 1
            fee = _commission(position["purchase_cost"])
            cash -= fee
            expected_cash -= fee
            total_commission += fee
            position["entry_fee"] = fee

        if positions:
            sell_window(trade_date, rows, EXIT_START, "CLOSED_INTRADAY")
        daily_equity[trade_date] = mark(rows.get(time(15, 0), {}))

    ending_market_value = sum(
        int(position["shares"]) * float(position["last_price"])
        for position in positions.values()
    )
    for symbol, position in positions.items():
        records.append(
            {
                "signal_date": position["signal"]["date"],
                "entry_date": position["entry_date"],
                "exit_date": None,
                "symbol": symbol,
                "status": "OPEN_RESIDUAL",
                "shares": position["shares"],
            }
        )
    ending_equity = cash + ending_market_value
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (
        (ending_equity / initial_capital) ** (1.0 / years) - 1.0
        if ending_equity > 0 else -1.0
    )
    monthly = _monthly_returns(daily_equity, initial_capital)
    closed = [row for row in records if str(row["status"]).startswith("CLOSED")]
    intraday = [row for row in records if row["status"] == "CLOSED_INTRADAY"]
    gains = [float(row["net_pnl"]) for row in closed if float(row["net_pnl"]) > 0]
    losses = [float(row["net_pnl"]) for row in closed if float(row["net_pnl"]) < 0]
    return {
        "initial_capital": initial_capital,
        "ending_cash": cash,
        "ending_market_value": ending_market_value,
        "ending_equity": ending_equity,
        "total_return": ending_equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown(equity_marks),
        "planned_legs": len(signals),
        "entered_legs": entered,
        "entry_execution_rate": entered / len(signals) if signals else 0.0,
        "intraday_trades": len(intraday),
        "closed_trades": len(closed),
        "win_rate": sum(float(row["net_pnl"]) > 0 for row in closed) / len(closed)
        if closed else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "positive_months": sum(value > 0 for value in monthly.values()),
        "month_returns": monthly,
        "carry_days": carry_days,
        "open_positions": len(positions),
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_cost": total_commission + total_slippage,
        "ledger_error": cash - expected_cash,
        "record_statuses": dict(Counter(row["status"] for row in records)),
        "records": records,
    }


def evaluate_gate(account: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, bool]:
    return {
        "annualized_at_least_50pct": account["annualized_return"] >= 0.50,
        "max_drawdown_no_worse_than_25pct": account["max_drawdown"] >= -0.25,
        "at_least_180_intraday_trades": account["intraday_trades"] >= 180,
        "at_least_4_positive_months": account["positive_months"] >= 4,
        "entry_execution_rate_at_least_90pct": account["entry_execution_rate"] >= 0.90,
        "mean_excess_at_least_10bps": diagnostics["mean_excess_return"] is not None
        and diagnostics["mean_excess_return"] >= 0.001,
        "no_overnight_or_open_residual": account["carry_days"] == 0
        and account["open_positions"] == 0
        and account["ending_market_value"] == 0,
        "ledger_balanced": abs(account["ledger_error"]) <= 0.01,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    for audit_name in (
        "p0_qdii_etf_minute_data_audit.json",
        "p0_gold_etf_minute_data_audit.json",
    ):
        audit = json.loads((data_dir / "research" / audit_name).read_text(encoding="utf-8"))
        if audit.get("status") != "DATA_QUALIFIED":
            raise RuntimeError(f"{audit_name} has not passed the frozen audit")
    qdii_root = (
        data_dir / "research" / "qdii_etf_intraday_momentum" / "phases" / "development"
    )
    gold_root = data_dir / "research" / "gold_etf_dispersion" / "phases" / "development"
    minutes = pl.concat(
        [
            pl.read_parquet(str(qdii_root / "symbol=*" / "part.parquet")),
            pl.read_parquet(str(gold_root / "symbol=*" / "part.parquet")),
        ],
        how="vertical_relaxed",
    )
    if minutes.filter(pl.col("datetime").dt.date() > DEV_END).height:
        raise RuntimeError("sealed validation or pressure rows entered development input")
    if set(minutes["symbol"].unique().to_list()) != set(SYMBOLS):
        raise RuntimeError("frozen T+0 ETF universe is incomplete")
    signals = build_signals(minutes)
    diagnostics = signal_diagnostics(minutes, signals)
    accounts = {
        str(int(capital)): simulate_account(minutes, signals, capital)
        for capital in INITIAL_CAPITALS
    }
    gate = evaluate_gate(accounts["200000"], diagnostics)
    payload = {
        "schema_version": "p0-t0-etf-overnight-intraday-reversal-development-v1",
        "contract_frozen": "2026-09-01",
        "period": {"start": DEV_START, "end": DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "signal_legs": len(signals),
        "diagnostics": diagnostics,
        "gate": gate,
        "decision": "CONTINUE_TO_VALIDATION" if all(gate.values()) else "TERMINATE_DEVELOPMENT",
        "accounts": accounts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    compact = {
        **{key: value for key, value in payload.items() if key != "accounts"},
        "accounts": {
            key: {field: value[field] for field in value if field != "records"}
            for key, value in accounts.items()
        },
        "output": str(output),
        "sha256": digest,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=_json_default))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_t0_etf_overnight_intraday_reversal_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
