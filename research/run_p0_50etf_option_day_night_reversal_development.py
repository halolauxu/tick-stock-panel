"""Run the frozen daily-proxy 50ETF option day/night reversal development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

DEV_START = date(2015, 2, 9)
DEV_END = date(2020, 12, 31)
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 500_000.0
OPTION_ACCOUNT_MIN_ASSETS = 500_000.0
STANDARD_MULTIPLIER = 10_000.0
MIN_VOLUME = 100.0
MIN_OPEN_INTEREST = 500.0
FEE_PER_LEG_SIDE = 5.0
MAX_SYNTHETIC_NOTIONAL_FRACTION = 1.0
MAX_RIGHTS_POSITION = 20


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _quote_lookup(options: pl.DataFrame) -> dict[tuple[str, date], dict[str, Any]]:
    return {
        (str(row["contract"]), row["date"]): row
        for row in options.iter_rows(named=True)
    }


def _reject(signal_date: date, trade_date: date, reason: str) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "trade_date": trade_date,
        "status": "REJECTED",
        "reason": reason,
    }


def build_sessions(
    master: pl.DataFrame, fund: pl.DataFrame, options: pl.DataFrame
) -> list[dict[str, Any]]:
    """Select each next-day pair from the prior close and prior-day liquidity only."""
    fund = fund.filter(
        pl.col("date").is_between(DEV_START, DEV_END, closed="both")
    ).sort("date")
    fund_rows = fund.to_dicts()
    quotes = _quote_lookup(options)
    standard = master.filter(pl.col("opt_multiplier") == STANDARD_MULTIPLIER)
    sessions: list[dict[str, Any]] = []
    for signal, current in zip(fund_rows[:-1], fund_rows[1:], strict=True):
        signal_date = signal["date"]
        trade_date = current["date"]
        chain = standard.filter(
            (pl.col("list_date") <= signal_date)
            & (pl.col("delist_date") >= trade_date)
            & (pl.col("maturity_date") >= trade_date)
        )
        if chain.is_empty():
            sessions.append(
                _reject(signal_date, trade_date, "NO_ACTIVE_STANDARD_CHAIN")
            )
            continue
        maturity = min(chain["maturity_date"].to_list())
        lead = chain.filter(pl.col("maturity_date") == maturity)
        call_strikes = set(
            lead.filter(pl.col("call_put") == "C")["exercise_price"].to_list()
        )
        put_strikes = set(
            lead.filter(pl.col("call_put") == "P")["exercise_price"].to_list()
        )
        paired = sorted(call_strikes.intersection(put_strikes))
        if not paired:
            sessions.append(_reject(signal_date, trade_date, "NO_PAIRED_STRIKE"))
            continue
        signal_close = float(signal["close"])
        strike = min(
            paired, key=lambda value: (abs(float(value) - signal_close), value)
        )
        selected: dict[str, dict[str, Any]] = {}
        for side, call_put in (("call", "C"), ("put", "P")):
            rows = (
                lead.filter(
                    (pl.col("call_put") == call_put)
                    & (pl.col("exercise_price") == strike)
                )
                .sort("contract")
                .to_dicts()
            )
            if not rows:
                break
            selected[side] = rows[0]
        if len(selected) != 2:
            sessions.append(_reject(signal_date, trade_date, "PAIR_CONTRACT_MISSING"))
            continue
        signal_quotes = {
            side: quotes.get((str(contract["contract"]), signal_date))
            for side, contract in selected.items()
        }
        if any(quote is None for quote in signal_quotes.values()):
            sessions.append(_reject(signal_date, trade_date, "SIGNAL_QUOTE_MISSING"))
            continue
        if any(
            not _positive(quote.get("close"))
            or not _positive(quote.get("pre_settle"))
            or float(quote.get("volume") or 0.0) < MIN_VOLUME
            or float(quote.get("open_interest") or 0.0) < MIN_OPEN_INTEREST
            for quote in signal_quotes.values()
            if quote is not None
        ):
            sessions.append(_reject(signal_date, trade_date, "SIGNAL_LIQUIDITY_FAILED"))
            continue
        current_quotes = {
            side: quotes.get((str(contract["contract"]), trade_date))
            for side, contract in selected.items()
        }
        if any(
            quote is None
            or not _positive(quote.get("open"))
            or not _positive(quote.get("close"))
            or not _positive(quote.get("pre_settle"))
            for quote in current_quotes.values()
        ):
            sessions.append(_reject(signal_date, trade_date, "NEXT_DAY_QUOTE_MISSING"))
            continue
        legs = {
            side: {
                "contract": str(contract["contract"]),
                "strike": float(contract["exercise_price"]),
                "tick": float(contract["min_price_chg"]),
                "multiplier": float(contract["opt_multiplier"]),
            }
            for side, contract in selected.items()
        }
        sessions.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "status": "READY",
                "reason": None,
                "maturity_date": maturity,
                "strike": float(strike),
                "signal_underlying_close": signal_close,
                "signal_underlying_pre_close": float(signal["pre_close"]),
                "trade_underlying_pre_close": float(current["pre_close"]),
                "legs": legs,
                "signal_quotes": signal_quotes,
                "trade_quotes": current_quotes,
            }
        )
    return sessions


def short_option_margin(
    *,
    call_put: str,
    pre_settle: float,
    underlying_pre_close: float,
    strike: float,
    multiplier: float,
) -> float:
    """SSE minimum opening margin for one uncovered ETF option contract."""
    if call_put == "C":
        out_of_money = max(strike - underlying_pre_close, 0.0)
        per_unit = pre_settle + max(
            0.12 * underlying_pre_close - out_of_money,
            0.07 * underlying_pre_close,
        )
    elif call_put == "P":
        out_of_money = max(underlying_pre_close - strike, 0.0)
        per_unit = min(
            pre_settle + max(0.12 * underlying_pre_close - out_of_money, 0.07 * strike),
            strike,
        )
    else:
        raise ValueError(f"unsupported option side: {call_put}")
    return per_unit * multiplier


def _max_sets(
    *,
    equity: float,
    long_entry: float,
    short_quote: dict[str, Any],
    short_call_put: str,
    underlying_pre_close: float,
    strike: float,
    multiplier: float,
) -> int:
    margin = short_option_margin(
        call_put=short_call_put,
        pre_settle=float(short_quote["pre_settle"]),
        underlying_pre_close=underlying_pre_close,
        strike=strike,
        multiplier=multiplier,
    )
    round_trip_fees = 4.0 * FEE_PER_LEG_SIDE
    conservative_cash = long_entry * multiplier + margin + round_trip_fees
    synthetic_notional = strike * multiplier
    if conservative_cash <= 0 or synthetic_notional <= 0:
        return 0
    return max(
        0,
        min(
            math.floor(equity / conservative_cash),
            math.floor(equity * MAX_SYNTHETIC_NOTIONAL_FRACTION / synthetic_notional),
            MAX_RIGHTS_POSITION,
        ),
    )


def _segment_pnl(
    *,
    long_entry_raw: float,
    long_exit_raw: float,
    short_entry_raw: float,
    short_exit_raw: float,
    long_tick: float,
    short_tick: float,
    multiplier: float,
    sets: int,
) -> dict[str, float]:
    long_entry = long_entry_raw + long_tick
    long_exit = long_exit_raw - long_tick
    short_entry = short_entry_raw - short_tick
    short_exit = short_exit_raw + short_tick
    if min(long_entry, long_exit, short_entry, short_exit) <= 0:
        raise ValueError("adverse execution price is not positive")
    gross = (
        ((long_exit_raw - long_entry_raw) - (short_exit_raw - short_entry_raw))
        * multiplier
        * sets
    )
    execution_pnl = (
        ((long_exit - long_entry) - (short_exit - short_entry)) * multiplier * sets
    )
    fees = 4.0 * FEE_PER_LEG_SIDE * sets
    net = execution_pnl - fees
    return {
        "gross_pnl": gross,
        "slippage": gross - execution_pnl,
        "fees": fees,
        "cost": gross - net,
        "net_pnl": net,
    }


def _year_returns(
    fund_dates: list[date], close_equity: dict[date, float], initial: float
) -> dict[str, float]:
    output: dict[str, float] = {}
    previous = initial
    for year in range(DEV_START.year, DEV_END.year + 1):
        dates = [trade_date for trade_date in fund_dates if trade_date.year == year]
        ending = close_equity.get(dates[-1], previous) if dates else previous
        output[str(year)] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return output


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def simulate_account(
    sessions: list[dict[str, Any]], fund_dates: list[date], initial_capital: float
) -> dict[str, Any]:
    equity = initial_capital
    curve = [initial_capital]
    close_equity: dict[date, float] = {}
    records: list[dict[str, Any]] = []
    for session in sessions:
        record = {
            "signal_date": session["signal_date"],
            "trade_date": session["trade_date"],
            "status": session["status"],
            "reason": session["reason"],
            "equity_before": equity,
        }
        if session["status"] != "READY":
            record.update(
                {
                    "overnight_sets": 0,
                    "intraday_sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            records.append(record)
            close_equity[session["trade_date"]] = equity
            continue
        call_leg = session["legs"]["call"]
        put_leg = session["legs"]["put"]
        signal_call = session["signal_quotes"]["call"]
        signal_put = session["signal_quotes"]["put"]
        trade_call = session["trade_quotes"]["call"]
        trade_put = session["trade_quotes"]["put"]
        multiplier = float(call_leg["multiplier"])
        overnight_sets = _max_sets(
            equity=equity,
            long_entry=float(signal_call["close"]) + float(call_leg["tick"]),
            short_quote=signal_put,
            short_call_put="P",
            underlying_pre_close=float(session["signal_underlying_pre_close"]),
            strike=float(session["strike"]),
            multiplier=multiplier,
        )
        if overnight_sets <= 0:
            record.update(
                {
                    "status": "REJECTED",
                    "reason": "OVERNIGHT_CAPITAL_OR_MARGIN",
                    "overnight_sets": 0,
                    "intraday_sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            records.append(record)
            close_equity[session["trade_date"]] = equity
            continue
        try:
            overnight = _segment_pnl(
                long_entry_raw=float(signal_call["close"]),
                long_exit_raw=float(trade_call["open"]),
                short_entry_raw=float(signal_put["close"]),
                short_exit_raw=float(trade_put["open"]),
                long_tick=float(call_leg["tick"]),
                short_tick=float(put_leg["tick"]),
                multiplier=multiplier,
                sets=overnight_sets,
            )
        except ValueError:
            record.update(
                {
                    "status": "REJECTED",
                    "reason": "OVERNIGHT_ADVERSE_PRICE_INVALID",
                    "overnight_sets": 0,
                    "intraday_sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            records.append(record)
            close_equity[session["trade_date"]] = equity
            continue
        equity += overnight["net_pnl"]
        curve.append(equity)
        intraday_sets = _max_sets(
            equity=equity,
            long_entry=float(trade_put["open"]) + float(put_leg["tick"]),
            short_quote=trade_call,
            short_call_put="C",
            underlying_pre_close=float(session["trade_underlying_pre_close"]),
            strike=float(session["strike"]),
            multiplier=multiplier,
        )
        if intraday_sets <= 0:
            record.update(
                {
                    "status": "PARTIAL_OVERNIGHT",
                    "reason": "INTRADAY_CAPITAL_OR_MARGIN",
                    "overnight_sets": overnight_sets,
                    "intraday_sets": 0,
                    "overnight": overnight,
                    "gross_pnl": overnight["gross_pnl"],
                    "cost": overnight["cost"],
                    "net_pnl": overnight["net_pnl"],
                    "equity_after": equity,
                }
            )
            records.append(record)
            close_equity[session["trade_date"]] = equity
            continue
        try:
            intraday = _segment_pnl(
                long_entry_raw=float(trade_put["open"]),
                long_exit_raw=float(trade_put["close"]),
                short_entry_raw=float(trade_call["open"]),
                short_exit_raw=float(trade_call["close"]),
                long_tick=float(put_leg["tick"]),
                short_tick=float(call_leg["tick"]),
                multiplier=multiplier,
                sets=intraday_sets,
            )
        except ValueError:
            record.update(
                {
                    "status": "PARTIAL_OVERNIGHT",
                    "reason": "INTRADAY_ADVERSE_PRICE_INVALID",
                    "overnight_sets": overnight_sets,
                    "intraday_sets": 0,
                    "overnight": overnight,
                    "gross_pnl": overnight["gross_pnl"],
                    "cost": overnight["cost"],
                    "net_pnl": overnight["net_pnl"],
                    "equity_after": equity,
                }
            )
            records.append(record)
            close_equity[session["trade_date"]] = equity
            continue
        equity += intraday["net_pnl"]
        curve.append(equity)
        gross = overnight["gross_pnl"] + intraday["gross_pnl"]
        cost = overnight["cost"] + intraday["cost"]
        net = overnight["net_pnl"] + intraday["net_pnl"]
        record.update(
            {
                "status": "FILLED",
                "reason": None,
                "maturity_date": session["maturity_date"],
                "strike": session["strike"],
                "call_contract": call_leg["contract"],
                "put_contract": put_leg["contract"],
                "overnight_sets": overnight_sets,
                "intraday_sets": intraday_sets,
                "overnight": overnight,
                "intraday": intraday,
                "gross_pnl": gross,
                "cost": cost,
                "net_pnl": net,
                "equity_after": equity,
            }
        )
        records.append(record)
        close_equity[session["trade_date"]] = equity
    ready = [
        record
        for record in records
        if record["status"] in {"FILLED", "PARTIAL_OVERNIGHT"}
    ]
    filled = [record for record in records if record["status"] == "FILLED"]
    planned = [session for session in sessions if session["status"] == "READY"]
    net_total = sum(float(record["net_pnl"]) for record in records)
    ledger_error = equity - initial_capital - net_total
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (
        (equity / initial_capital) ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    )
    yearly = _year_returns(fund_dates, close_equity, initial_capital)
    gains = [float(record["net_pnl"]) for record in ready if record["net_pnl"] > 0]
    losses = [float(record["net_pnl"]) for record in ready if record["net_pnl"] < 0]
    return {
        "initial_capital": initial_capital,
        "option_account_eligible": initial_capital >= OPTION_ACCOUNT_MIN_ASSETS,
        "eligibility_reason": (
            None
            if initial_capital >= OPTION_ACCOUNT_MIN_ASSETS
            else "NOT_ELIGIBLE_FOR_OPTION_ACCOUNT"
        ),
        "ending_equity": equity,
        "total_return": equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown(curve),
        "planned_sessions": len(planned),
        "complete_sessions": len(filled),
        "execution_rate": len(filled) / len(planned) if planned else 0.0,
        "win_rate": sum(record["net_pnl"] > 0 for record in filled) / len(filled)
        if filled
        else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "positive_years": sum(value > 0 for value in yearly.values()),
        "year_returns": yearly,
        "overnight_net_pnl": sum(
            float(record.get("overnight", {}).get("net_pnl", 0.0)) for record in records
        ),
        "intraday_net_pnl": sum(
            float(record.get("intraday", {}).get("net_pnl", 0.0)) for record in records
        ),
        "gross_pnl": sum(float(record["gross_pnl"]) for record in records),
        "total_cost": sum(float(record["cost"]) for record in records),
        "ledger_error": ledger_error,
        "reject_reasons": {
            reason: sum(record["reason"] == reason for record in records)
            for reason in sorted(
                {record["reason"] for record in records if record["reason"]}
            )
        },
        "records": records,
    }


def evaluate_gate(account: dict[str, Any]) -> dict[str, bool]:
    return {
        "option_account_eligible": bool(account["option_account_eligible"]),
        "annualized_at_least_50pct": account["annualized_return"] >= 0.50,
        "max_drawdown_no_worse_than_25pct": account["max_drawdown"] >= -0.25,
        "at_least_1000_complete_sessions": account["complete_sessions"] >= 1_000,
        "at_least_4_positive_years": account["positive_years"] >= 4,
        "execution_rate_at_least_90pct": account["execution_rate"] >= 0.90,
        "ledger_balanced": abs(account["ledger_error"]) <= 0.01,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "50etf_option_vrp"
    audit = json.loads(
        (data_dir / "research" / "p0_50etf_option_data_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if audit.get("status") != "DATA_QUALIFIED":
        raise RuntimeError("50ETF option data has not passed the frozen audit")
    master = pl.read_parquet(root / "contracts.parquet")
    fund = pl.read_parquet(root / "underlying.parquet").filter(
        pl.col("date") <= DEV_END
    )
    options = pl.read_parquet(str(root / "daily" / "date=*" / "part.parquet")).filter(
        pl.col("date") <= DEV_END
    )
    sessions = build_sessions(master, fund, options)
    fund_dates = fund.sort("date")["date"].to_list()
    accounts = {
        str(int(capital)): simulate_account(sessions, fund_dates, capital)
        for capital in INITIAL_CAPITALS
    }
    primary = accounts[str(int(PRIMARY_CAPITAL))]
    gate = evaluate_gate(primary)
    payload = {
        "schema_version": "p0-50etf-option-day-night-reversal-daily-proxy-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": DEV_START, "end": DEV_END},
        "price_boundary": "DAILY_OPEN_PROXY_NOT_0935",
        "eligible_capital_floor": OPTION_ACCOUNT_MIN_ASSETS,
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "session_count": len(sessions),
        "primary_capital": PRIMARY_CAPITAL,
        "gate": gate,
        "decision": (
            "COLLECT_5MIN_DATA" if all(gate.values()) else "TERMINATE_DAILY_PROXY"
        ),
        "strict_strategy_count_increment": 0,
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
            "/app/data/research/p0_50etf_option_day_night_reversal_daily_proxy.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
