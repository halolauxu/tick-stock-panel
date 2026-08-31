"""Run the frozen 2015-2020 50ETF iron-butterfly development accounts."""
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
STANDARD_MULTIPLIER = 10_000.0
LOWER_WING_RATIO = 0.93
UPPER_WING_RATIO = 1.07
SIGNAL_DAYS_BEFORE_EXPIRY = 21
EXIT_DAYS_BEFORE_EXPIRY = 5
MIN_VOLUME = 100.0
MIN_OPEN_INTEREST = 500.0
FEE_PER_LEG_SIDE = 5.0
RISK_FRACTION = 0.10
CASH_FRACTION = 0.50


def _quote_lookup(options: pl.DataFrame) -> dict[tuple[str, date], dict[str, Any]]:
    return {
        (str(row["contract"]), row["date"]): row
        for row in options.iter_rows(named=True)
    }


def _rejected_cycle(
    maturity: date,
    reason: str,
    signal_date: date | None = None,
    entry_date: date | None = None,
    exit_date: date | None = None,
) -> dict[str, Any]:
    return {
        "maturity_date": maturity,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "status": "REJECTED",
        "reason": reason,
        "legs": None,
    }


def _one_contract(frame: pl.DataFrame, call_put: str, strike: float) -> dict[str, Any] | None:
    rows = (
        frame.filter(
            (pl.col("call_put") == call_put)
            & (pl.col("exercise_price") == strike)
        )
        .sort("contract")
        .to_dicts()
    )
    return rows[0] if rows else None


def build_cycles(
    master: pl.DataFrame, fund: pl.DataFrame, options: pl.DataFrame
) -> list[dict[str, Any]]:
    """Build monthly choices using only the signal close and prior-day liquidity."""
    fund = fund.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both")).sort(
        "date"
    )
    fund_dates = fund["date"].to_list()
    close_by_date = dict(zip(fund_dates, fund["close"].to_list(), strict=True))
    quotes = _quote_lookup(options)
    standard = master.filter(
        (pl.col("opt_multiplier") == STANDARD_MULTIPLIER)
        & pl.col("maturity_date").is_between(DEV_START, DEV_END, closed="both")
    )
    maturities = sorted(standard["maturity_date"].unique().to_list())
    cycles: list[dict[str, Any]] = []
    for maturity in maturities:
        before_expiry = [trade_date for trade_date in fund_dates if trade_date < maturity]
        if len(before_expiry) < SIGNAL_DAYS_BEFORE_EXPIRY + 1:
            cycles.append(_rejected_cycle(maturity, "INSUFFICIENT_TRADING_HISTORY"))
            continue
        signal_date = before_expiry[-SIGNAL_DAYS_BEFORE_EXPIRY]
        prior_date = before_expiry[-SIGNAL_DAYS_BEFORE_EXPIRY - 1]
        entry_date = before_expiry[-SIGNAL_DAYS_BEFORE_EXPIRY + 1]
        exit_date = before_expiry[-EXIT_DAYS_BEFORE_EXPIRY]
        underlying_close = float(close_by_date[signal_date])
        chain = standard.filter(
            (pl.col("maturity_date") == maturity)
            & (pl.col("list_date") <= signal_date)
            & (pl.col("delist_date") >= exit_date)
        )
        call_strikes = set(chain.filter(pl.col("call_put") == "C")["exercise_price"].to_list())
        put_strikes = set(chain.filter(pl.col("call_put") == "P")["exercise_price"].to_list())
        paired_strikes = sorted(call_strikes.intersection(put_strikes))
        if not paired_strikes:
            cycles.append(
                _rejected_cycle(
                    maturity, "NO_PAIRED_ATM_STRIKE", signal_date, entry_date, exit_date
                )
            )
            continue
        atm_strike = min(paired_strikes, key=lambda strike: (abs(strike - underlying_close), strike))
        lower = [strike for strike in put_strikes if strike <= underlying_close * LOWER_WING_RATIO]
        upper = [strike for strike in call_strikes if strike >= underlying_close * UPPER_WING_RATIO]
        if not lower or not upper:
            cycles.append(
                _rejected_cycle(
                    maturity, "PROTECTIVE_WING_UNAVAILABLE", signal_date, entry_date, exit_date
                )
            )
            continue
        lower_strike = max(lower)
        upper_strike = min(upper)
        selected = {
            "short_call": _one_contract(chain, "C", atm_strike),
            "short_put": _one_contract(chain, "P", atm_strike),
            "long_put": _one_contract(chain, "P", lower_strike),
            "long_call": _one_contract(chain, "C", upper_strike),
        }
        if any(leg is None for leg in selected.values()):
            cycles.append(
                _rejected_cycle(
                    maturity, "FOUR_LEG_CHAIN_INCOMPLETE", signal_date, entry_date, exit_date
                )
            )
            continue
        liquid = True
        for leg in selected.values():
            assert leg is not None
            quote = quotes.get((leg["contract"], prior_date))
            if (
                quote is None
                or quote.get("volume") is None
                or quote.get("open_interest") is None
                or float(quote["volume"]) < MIN_VOLUME
                or float(quote["open_interest"]) < MIN_OPEN_INTEREST
            ):
                liquid = False
                break
        if not liquid:
            cycles.append(
                _rejected_cycle(
                    maturity, "PRIOR_DAY_LIQUIDITY_FAILED", signal_date, entry_date, exit_date
                )
            )
            continue
        legs = {
            name: {
                "contract": str(leg["contract"]),
                "strike": float(leg["exercise_price"]),
                "tick": float(leg["min_price_chg"]),
                "multiplier": float(leg["opt_multiplier"]),
            }
            for name, leg in selected.items()
            if leg is not None
        }
        cycles.append(
            {
                "maturity_date": maturity,
                "signal_date": signal_date,
                "liquidity_date": prior_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "underlying_close": underlying_close,
                "status": "READY",
                "reason": None,
                "legs": legs,
            }
        )
    return cycles


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _open_prices(
    cycle: dict[str, Any], quotes: dict[tuple[str, date], dict[str, Any]], trade_date: date
) -> dict[str, float] | None:
    prices: dict[str, float] = {}
    for name, leg in cycle["legs"].items():
        quote = quotes.get((leg["contract"], trade_date))
        if quote is None or not _positive(quote.get("open")):
            return None
        prices[name] = float(quote["open"])
    return prices


def _net_debit(prices: dict[str, float]) -> float:
    return prices["short_call"] + prices["short_put"] - prices["long_put"] - prices["long_call"]


def _year_returns(
    fund_dates: list[date], equity_curve: dict[date, float], initial: float
) -> dict[str, float]:
    output: dict[str, float] = {}
    previous = initial
    for year in range(DEV_START.year, DEV_END.year + 1):
        dates = [trade_date for trade_date in fund_dates if trade_date.year == year]
        ending = equity_curve[dates[-1]] if dates else previous
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
    cycles: list[dict[str, Any]],
    options: pl.DataFrame,
    fund_dates: list[date],
    initial_capital: float,
) -> dict[str, Any]:
    quotes = _quote_lookup(options)
    realized_equity = initial_capital
    records: list[dict[str, Any]] = []
    mark_overrides: dict[date, float] = {}
    realized_after: dict[date, float] = {}
    for cycle in cycles:
        record = {**cycle, "equity_before": realized_equity}
        if cycle["status"] != "READY":
            record.update(
                {"sets": 0, "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0, "equity_after": realized_equity}
            )
            records.append(record)
            continue
        entry_raw = _open_prices(cycle, quotes, cycle["entry_date"])
        exit_raw = _open_prices(cycle, quotes, cycle["exit_date"])
        if entry_raw is None or exit_raw is None:
            record.update(
                {
                    "status": "REJECTED",
                    "reason": "ENTRY_OR_EXIT_OPEN_MISSING",
                    "sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": realized_equity,
                }
            )
            records.append(record)
            continue
        legs = cycle["legs"]
        entry_adverse = {
            "short_call": max(0.0, entry_raw["short_call"] - legs["short_call"]["tick"]),
            "short_put": max(0.0, entry_raw["short_put"] - legs["short_put"]["tick"]),
            "long_put": entry_raw["long_put"] + legs["long_put"]["tick"],
            "long_call": entry_raw["long_call"] + legs["long_call"]["tick"],
        }
        exit_adverse = {
            "short_call": exit_raw["short_call"] + legs["short_call"]["tick"],
            "short_put": exit_raw["short_put"] + legs["short_put"]["tick"],
            "long_put": max(0.0, exit_raw["long_put"] - legs["long_put"]["tick"]),
            "long_call": max(0.0, exit_raw["long_call"] - legs["long_call"]["tick"]),
        }
        entry_credit = _net_debit(entry_adverse)
        if entry_credit <= 0:
            record.update(
                {
                    "status": "REJECTED",
                    "reason": "NONPOSITIVE_ENTRY_CREDIT",
                    "sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": realized_equity,
                }
            )
            records.append(record)
            continue
        multiplier = float(legs["short_call"]["multiplier"])
        width = max(
            legs["short_call"]["strike"] - legs["long_put"]["strike"],
            legs["long_call"]["strike"] - legs["short_call"]["strike"],
        )
        round_trip_fee = 8 * FEE_PER_LEG_SIDE
        max_loss = width * multiplier - entry_credit * multiplier + round_trip_fee
        risk_sets = math.floor(realized_equity * RISK_FRACTION / max_loss) if max_loss > 0 else 0
        cash_sets = math.floor(realized_equity * CASH_FRACTION / max_loss) if max_loss > 0 else 0
        sets = max(0, min(risk_sets, cash_sets))
        if sets <= 0:
            record.update(
                {
                    "status": "REJECTED",
                    "reason": "CAPITAL_INSUFFICIENT",
                    "sets": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "equity_after": realized_equity,
                }
            )
            records.append(record)
            continue
        entry_fee = 4 * FEE_PER_LEG_SIDE * sets
        exit_fee = 4 * FEE_PER_LEG_SIDE * sets
        entry_cash = entry_credit * multiplier * sets - entry_fee
        last_known = dict(entry_raw)
        active_dates = [
            trade_date
            for trade_date in fund_dates
            if cycle["entry_date"] <= trade_date < cycle["exit_date"]
        ]
        for trade_date in active_dates:
            marks: dict[str, float] = {}
            for name, leg in legs.items():
                quote = quotes.get((leg["contract"], trade_date))
                candidate = None
                if quote is not None:
                    if _positive(quote.get("close")):
                        candidate = float(quote["close"])
                    elif _positive(quote.get("settle")):
                        candidate = float(quote["settle"])
                if candidate is not None:
                    last_known[name] = candidate
                marks[name] = last_known[name]
            mark_overrides[trade_date] = (
                record["equity_before"] + entry_cash - _net_debit(marks) * multiplier * sets
            )
        raw_gross = (_net_debit(entry_raw) - _net_debit(exit_raw)) * multiplier * sets
        exit_debit = _net_debit(exit_adverse) * multiplier * sets
        net_pnl = entry_cash - exit_debit - exit_fee
        cost = raw_gross - net_pnl
        realized_equity += net_pnl
        realized_after[cycle["exit_date"]] = realized_equity
        mark_overrides[cycle["exit_date"]] = realized_equity
        record.update(
            {
                "status": "FILLED",
                "reason": None,
                "sets": sets,
                "entry_credit_per_unit": entry_credit,
                "max_loss_per_set": max_loss,
                "gross_pnl": raw_gross,
                "cost": cost,
                "net_pnl": net_pnl,
                "equity_after": realized_equity,
            }
        )
        records.append(record)
    equity_curve: dict[date, float] = {}
    current = initial_capital
    for trade_date in fund_dates:
        if trade_date in realized_after:
            current = realized_after[trade_date]
        equity_curve[trade_date] = mark_overrides.get(trade_date, current)
    filled = [record for record in records if record["status"] == "FILLED"]
    net_pnl_total = sum(float(record["net_pnl"]) for record in records)
    ledger_error = realized_equity - initial_capital - net_pnl_total
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (
        (realized_equity / initial_capital) ** (1.0 / years) - 1.0
        if realized_equity > 0
        else -1.0
    )
    yearly = _year_returns(fund_dates, equity_curve, initial_capital)
    gains = [float(record["net_pnl"]) for record in filled if record["net_pnl"] > 0]
    losses = [float(record["net_pnl"]) for record in filled if record["net_pnl"] < 0]
    return {
        "initial_capital": initial_capital,
        "ending_equity": realized_equity,
        "total_return": realized_equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown([initial_capital, *equity_curve.values()]),
        "cycles": len(records),
        "trades": len(filled),
        "execution_rate": len(filled) / len(records) if records else 0.0,
        "win_rate": sum(record["net_pnl"] > 0 for record in filled) / len(filled) if filled else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "positive_years": sum(value > 0 for value in yearly.values()),
        "year_returns": yearly,
        "total_cost": sum(float(record["cost"]) for record in filled),
        "ledger_error": ledger_error,
        "reject_reasons": {
            reason: sum(record["reason"] == reason for record in records)
            for reason in sorted({record["reason"] for record in records if record["reason"]})
        },
        "records": records,
    }


def evaluate_gate(account: dict[str, Any]) -> dict[str, bool]:
    return {
        "annualized_at_least_50pct": account["annualized_return"] >= 0.50,
        "max_drawdown_no_worse_than_25pct": account["max_drawdown"] >= -0.25,
        "at_least_30_iron_butterflies": account["trades"] >= 30,
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
        (data_dir / "research" / "p0_50etf_option_data_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "DATA_QUALIFIED":
        raise RuntimeError("50ETF option data has not passed the frozen audit")
    master = pl.read_parquet(root / "contracts.parquet")
    fund = pl.read_parquet(root / "underlying.parquet").filter(pl.col("date") <= DEV_END)
    options = pl.read_parquet(str(root / "daily" / "date=*" / "part.parquet")).filter(
        pl.col("date") <= DEV_END
    )
    cycles = build_cycles(master, fund, options)
    fund_dates = fund.sort("date")["date"].to_list()
    accounts = {
        str(int(capital)): simulate_account(cycles, options, fund_dates, capital)
        for capital in INITIAL_CAPITALS
    }
    primary = accounts[str(int(INITIAL_CAPITALS[0]))]
    gate = evaluate_gate(primary)
    payload = {
        "schema_version": "p0-50etf-iron-butterfly-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": DEV_START, "end": DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "cycle_count": len(cycles),
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
        default=Path("/app/data/research/p0_50etf_iron_butterfly_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
