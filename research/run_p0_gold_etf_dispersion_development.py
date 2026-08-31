"""Run the frozen gold-ETF T+0 dispersion strategy on development data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

DEV_START = date(2025, 8, 1)
DEV_END = date(2025, 12, 31)
SYMBOLS = ("518880.SH", "518800.SH", "159934.SZ", "159937.SZ")
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
EXPECTED_BARS = 241
SIGNAL_TIME = time(10, 0)
ENTRY_START = time(10, 1)
ENTRY_END = time(10, 5)
SCHEDULED_EXIT = time(14, 55)
MIN_CUMULATIVE_AMOUNT = 1_000_000.0
MIN_DISPERSION = 0.002
CONVERGENCE_GAP = 0.0005
MAX_ALLOCATION = 0.50
MAX_VOLUME_PARTICIPATION = 0.01
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
TICK = 0.001
LOT = 100


def _day_rows(frame: pl.DataFrame) -> dict[time, dict[str, dict[str, Any]]]:
    output: dict[time, dict[str, dict[str, Any]]] = {}
    for row in frame.sort(["datetime", "symbol"]).iter_rows(named=True):
        output.setdefault(row["datetime"].time(), {})[row["symbol"]] = row
    return output


def build_signals(minutes: pl.DataFrame) -> list[dict[str, Any]]:
    """Freeze one daily signal from complete, contemporaneous 10:00 bars."""
    frame = minutes.with_columns(pl.col("datetime").dt.date().alias("date"))
    signals: list[dict[str, Any]] = []
    for trade_date in sorted(frame["date"].unique().to_list()):
        day = frame.filter(pl.col("date") == trade_date).drop("date")
        if day.height != len(SYMBOLS) * EXPECTED_BARS:
            continue
        rows = _day_rows(day)
        opening = rows.get(time(9, 30), {})
        signal_bars = rows.get(SIGNAL_TIME, {})
        if set(opening) != set(SYMBOLS) or set(signal_bars) != set(SYMBOLS):
            continue
        cumulative = (
            day.filter(pl.col("datetime").dt.time() <= SIGNAL_TIME)
            .group_by("symbol")
            .agg(pl.col("amount").sum().alias("amount"))
        )
        amount_by_symbol = dict(
            zip(cumulative["symbol"].to_list(), cumulative["amount"].to_list(), strict=True)
        )
        if any(float(amount_by_symbol.get(symbol, 0.0)) < MIN_CUMULATIVE_AMOUNT for symbol in SYMBOLS):
            continue
        bases = {symbol: float(opening[symbol]["open"]) for symbol in SYMBOLS}
        if any(price <= 0 for price in bases.values()):
            continue
        returns = {
            symbol: float(signal_bars[symbol]["close"]) / bases[symbol] - 1.0
            for symbol in SYMBOLS
        }
        median_return = statistics.median(returns.values())
        laggard = min(SYMBOLS, key=lambda symbol: (returns[symbol], symbol))
        gap = median_return - returns[laggard]
        if median_return < 0 or gap < MIN_DISPERSION:
            continue
        signals.append(
            {
                "date": trade_date,
                "symbol": laggard,
                "median_return": median_return,
                "laggard_return": returns[laggard],
                "dispersion": gap,
                "base_prices": bases,
                "signal_close": float(signal_bars[laggard]["close"]),
            }
        )
    return signals


def _commission(notional: float) -> float:
    return max(MIN_COMMISSION, notional * COMMISSION_RATE) if notional > 0 else 0.0


def _capacity(volume: float) -> int:
    return max(0, math.floor(volume * MAX_VOLUME_PARTICIPATION / LOT) * LOT)


def _median_gap(
    rows: dict[str, dict[str, Any]], symbol: str, bases: dict[str, float]
) -> float | None:
    if set(rows) != set(SYMBOLS):
        return None
    returns = {
        candidate: float(rows[candidate]["close"]) / bases[candidate] - 1.0
        for candidate in SYMBOLS
    }
    return statistics.median(returns.values()) - returns[symbol]


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _monthly_returns(daily_equity: dict[date, float], initial: float) -> dict[str, float]:
    output: dict[str, float] = {}
    previous = initial
    months = sorted({trade_date.strftime("%Y-%m") for trade_date in daily_equity})
    for month in months:
        dates = [trade_date for trade_date in daily_equity if trade_date.strftime("%Y-%m") == month]
        ending = daily_equity[max(dates)]
        output[month] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return output


def simulate_account(
    minutes: pl.DataFrame, signals: list[dict[str, Any]], initial_capital: float
) -> dict[str, Any]:
    frame = minutes.with_columns(pl.col("datetime").dt.date().alias("date"))
    signal_by_date = {signal["date"]: signal for signal in signals}
    dates = sorted(frame["date"].unique().to_list())
    cash = initial_capital
    expected_cash = initial_capital
    position: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    equity_marks = [initial_capital]
    daily_equity: dict[date, float] = {}
    entered_signals = 0
    overnight_failures = 0
    total_commission = 0.0
    total_slippage = 0.0

    def mark(row: dict[str, Any] | None = None) -> float:
        if position is None:
            return cash
        price = float(row["close"]) if row is not None else float(position["last_price"])
        position["last_price"] = price
        return cash + int(position["shares"]) * price

    def finish_position(exit_date: date, status: str) -> None:
        nonlocal position
        assert position is not None
        net_pnl = (
            float(position["sale_proceeds"])
            - float(position["exit_fees"])
            - float(position["purchase_cost"])
            - float(position["entry_fee"])
        )
        gross_pnl = float(position["raw_sale_proceeds"]) - float(position["raw_purchase_cost"])
        records.append(
            {
                "signal_date": position["signal"]["date"],
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "symbol": position["symbol"],
                "status": status,
                "shares": position["entry_shares"],
                "entry_average": position["purchase_cost"] / position["entry_shares"],
                "exit_average": position["sale_proceeds"] / position["entry_shares"],
                "gross_pnl": gross_pnl,
                "cost": gross_pnl - net_pnl,
                "net_pnl": net_pnl,
                "dispersion": position["signal"]["dispersion"],
                "convergence_triggered": position["convergence_triggered"],
            }
        )
        position = None

    for trade_date in dates:
        day = frame.filter(pl.col("date") == trade_date).drop("date").sort(["datetime", "symbol"])
        rows = _day_rows(day)
        had_residual = position is not None
        if had_residual:
            assert position is not None
            symbol = position["symbol"]
            day_sale = 0.0
            for timestamp in sorted(rows):
                row = rows[timestamp].get(symbol)
                if row is None:
                    continue
                position["last_price"] = float(row["close"])
                if position["shares"] > 0:
                    shares = min(position["shares"], _capacity(float(row["volume"])))
                    if shares > 0:
                        raw_price = float(row["open"])
                        price = max(0.0, raw_price - TICK)
                        proceeds = shares * price
                        cash += proceeds
                        expected_cash += proceeds
                        day_sale += proceeds
                        total_slippage += shares * (raw_price - price)
                        position["shares"] -= shares
                        position["sale_proceeds"] += proceeds
                        position["raw_sale_proceeds"] += shares * raw_price
                equity_marks.append(mark(row))
                if position["shares"] == 0:
                    break
            fee = _commission(day_sale)
            cash -= fee
            expected_cash -= fee
            total_commission += fee
            position["exit_fees"] += fee
            if position["shares"] == 0:
                finish_position(trade_date, "CLOSED_AFTER_OVERNIGHT")
            if position is not None:
                last_row = rows[max(rows)].get(position["symbol"])
                daily_equity[trade_date] = mark(last_row)
            else:
                daily_equity[trade_date] = cash
            continue

        signal = signal_by_date.get(trade_date)
        if signal is None:
            daily_equity[trade_date] = cash
            continue
        symbol = signal["symbol"]
        target_shares = (
            math.floor(
                (cash * MAX_ALLOCATION) / (float(signal["signal_close"]) + TICK) / LOT
            )
            * LOT
        )
        if target_shares <= 0:
            records.append(
                {
                    "signal_date": trade_date,
                    "entry_date": None,
                    "exit_date": None,
                    "symbol": symbol,
                    "status": "REJECTED_CAPITAL",
                    "shares": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "dispersion": signal["dispersion"],
                }
            )
            daily_equity[trade_date] = cash
            continue
        position = {
            "signal": signal,
            "symbol": symbol,
            "shares": 0,
            "entry_shares": 0,
            "purchase_cost": 0.0,
            "raw_purchase_cost": 0.0,
            "entry_fee": 0.0,
            "sale_proceeds": 0.0,
            "raw_sale_proceeds": 0.0,
            "exit_fees": 0.0,
            "entry_date": trade_date,
            "last_price": float(signal["signal_close"]),
            "convergence_triggered": False,
        }
        entry_finalized = False
        exit_active = False
        exit_start: time | None = None
        exit_order_notional = 0.0
        for timestamp in sorted(rows):
            if timestamp < ENTRY_START:
                continue
            symbol_row = rows[timestamp].get(symbol)
            if symbol_row is None:
                continue
            position["last_price"] = float(symbol_row["close"])
            if timestamp <= ENTRY_END and position["entry_shares"] < target_shares:
                remaining = target_shares - position["entry_shares"]
                shares = min(remaining, _capacity(float(symbol_row["volume"])))
                if shares > 0:
                    raw_price = float(symbol_row["open"])
                    price = raw_price + TICK
                    affordable = max(0, math.floor((cash - MIN_COMMISSION) / price / LOT) * LOT)
                    shares = min(shares, affordable)
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        expected_cash -= cost
                        total_slippage += shares * (price - raw_price)
                        position["shares"] += shares
                        position["entry_shares"] += shares
                        position["purchase_cost"] += cost
                        position["raw_purchase_cost"] += shares * raw_price
            if timestamp == ENTRY_END and not entry_finalized:
                entry_finalized = True
                if position["entry_shares"] > 0:
                    entered_signals += 1
                    fee = _commission(float(position["purchase_cost"]))
                    cash -= fee
                    expected_cash -= fee
                    total_commission += fee
                    position["entry_fee"] = fee
                else:
                    records.append(
                        {
                            "signal_date": trade_date,
                            "entry_date": None,
                            "exit_date": None,
                            "symbol": symbol,
                            "status": "REJECTED_NO_MINUTE_CAPACITY",
                            "shares": 0,
                            "gross_pnl": 0.0,
                            "cost": 0.0,
                            "net_pnl": 0.0,
                            "dispersion": signal["dispersion"],
                        }
                    )
                    position = None
                    break
            if position is None:
                break
            if entry_finalized and position["entry_shares"] > 0 and not exit_active:
                if (exit_start is not None and timestamp >= exit_start) or timestamp >= SCHEDULED_EXIT:
                    exit_active = True
                elif timestamp >= ENTRY_END:
                    gap = _median_gap(rows[timestamp], symbol, signal["base_prices"])
                    if gap is not None and gap <= CONVERGENCE_GAP:
                        position["convergence_triggered"] = True
                        later = [candidate for candidate in sorted(rows) if candidate > timestamp]
                        exit_start = later[0] if later else None
            if exit_active and position["shares"] > 0:
                shares = min(position["shares"], _capacity(float(symbol_row["volume"])))
                if shares > 0:
                    raw_price = float(symbol_row["open"])
                    price = max(0.0, raw_price - TICK)
                    proceeds = shares * price
                    cash += proceeds
                    expected_cash += proceeds
                    total_slippage += shares * (raw_price - price)
                    exit_order_notional += proceeds
                    position["shares"] -= shares
                    position["sale_proceeds"] += proceeds
                    position["raw_sale_proceeds"] += shares * raw_price
            equity_marks.append(mark(symbol_row))
            if exit_active and position["shares"] == 0:
                fee = _commission(exit_order_notional)
                cash -= fee
                expected_cash -= fee
                total_commission += fee
                position["exit_fees"] += fee
                finish_position(trade_date, "CLOSED_INTRADAY")
                equity_marks.append(cash)
                break
        if position is not None:
            if position["entry_shares"] > 0 and position["shares"] > 0:
                fee = _commission(exit_order_notional)
                cash -= fee
                expected_cash -= fee
                total_commission += fee
                position["exit_fees"] += fee
                overnight_failures += 1
            last_row = rows[max(rows)].get(symbol)
            daily_equity[trade_date] = mark(last_row)
        else:
            daily_equity[trade_date] = cash

    ending_market_value = 0.0
    if position is not None:
        ending_market_value = int(position["shares"]) * float(position["last_price"])
        records.append(
            {
                "signal_date": position["signal"]["date"],
                "entry_date": position["entry_date"],
                "exit_date": None,
                "symbol": position["symbol"],
                "status": "OPEN_RESIDUAL",
                "shares": position["shares"],
                "gross_pnl": None,
                "cost": None,
                "net_pnl": None,
                "dispersion": position["signal"]["dispersion"],
            }
        )
    ending_equity = cash + ending_market_value
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (
        (ending_equity / initial_capital) ** (1.0 / years) - 1.0
        if ending_equity > 0
        else -1.0
    )
    monthly = _monthly_returns(daily_equity, initial_capital)
    closed = [record for record in records if str(record["status"]).startswith("CLOSED")]
    intraday = [record for record in records if record["status"] == "CLOSED_INTRADAY"]
    gains = [float(record["net_pnl"]) for record in closed if float(record["net_pnl"]) > 0]
    losses = [float(record["net_pnl"]) for record in closed if float(record["net_pnl"]) < 0]
    return {
        "initial_capital": initial_capital,
        "ending_cash": cash,
        "ending_market_value": ending_market_value,
        "ending_equity": ending_equity,
        "total_return": ending_equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown(equity_marks),
        "signals": len(signals),
        "entered_signals": entered_signals,
        "signal_execution_rate": entered_signals / len(signals) if signals else 0.0,
        "intraday_trades": len(intraday),
        "closed_trades": len(closed),
        "win_rate": sum(record["net_pnl"] > 0 for record in closed) / len(closed) if closed else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "positive_months": sum(value > 0 for value in monthly.values()),
        "month_returns": monthly,
        "overnight_failures": overnight_failures,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_cost": total_commission + total_slippage,
        "ledger_error": cash - expected_cash,
        "record_statuses": dict(Counter(record["status"] for record in records)),
        "records": records,
    }


def evaluate_gate(account: dict[str, Any]) -> dict[str, bool]:
    return {
        "annualized_at_least_50pct": account["annualized_return"] >= 0.50,
        "max_drawdown_no_worse_than_25pct": account["max_drawdown"] >= -0.25,
        "at_least_40_intraday_trades": account["intraday_trades"] >= 40,
        "at_least_4_positive_months": account["positive_months"] >= 4,
        "signal_execution_rate_at_least_80pct": account["signal_execution_rate"] >= 0.80,
        "no_overnight_residual": account["overnight_failures"] == 0 and account["ending_market_value"] == 0,
        "ledger_balanced": abs(account["ledger_error"]) <= 0.01,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit = json.loads(
        (data_dir / "research" / "p0_gold_etf_minute_data_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "DATA_QUALIFIED":
        raise RuntimeError("gold ETF minute data has not passed the frozen audit")
    root = data_dir / "research" / "gold_etf_dispersion" / "phases" / "development"
    minutes = pl.read_parquet(str(root / "symbol=*" / "part.parquet"))
    if minutes.filter(pl.col("datetime").dt.date() > DEV_END).height:
        raise RuntimeError("sealed validation or pressure rows entered development input")
    signals = build_signals(minutes)
    accounts = {
        str(int(capital)): simulate_account(minutes, signals, capital)
        for capital in INITIAL_CAPITALS
    }
    primary = accounts[str(int(INITIAL_CAPITALS[0]))]
    gate = evaluate_gate(primary)
    payload = {
        "schema_version": "p0-gold-etf-dispersion-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": DEV_START, "end": DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "signal_count": len(signals),
        "gate": gate,
        "decision": "CONTINUE_TO_VALIDATION" if all(gate.values()) else "TERMINATE_DEVELOPMENT",
        "accounts": accounts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
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
        default=Path("/app/data/research/p0_gold_etf_dispersion_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
