"""Run the frozen 2015-2020 50ETF directional call-spread development study."""
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
TREND_WINDOW = 60
MOMENTUM_WINDOW = 20
MIN_MOMENTUM = 0.03
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


def _rejected(
    maturity: date,
    signal_date: date,
    entry_date: date,
    exit_date: date,
    reason: str,
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


def build_cycles(
    master: pl.DataFrame, fund: pl.DataFrame, options: pl.DataFrame
) -> list[dict[str, Any]]:
    fund = (
        fund.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .sort("date")
        .with_columns(
            pl.col("close")
            .shift(1)
            .rolling_mean(window_size=TREND_WINDOW, min_samples=TREND_WINDOW)
            .alias("prior_ma60"),
            (pl.col("close") / pl.col("close").shift(MOMENTUM_WINDOW) - 1.0).alias(
                "momentum20"
            ),
        )
    )
    fund_by_date = {row["date"]: row for row in fund.iter_rows(named=True)}
    dates = fund["date"].to_list()
    quotes = _quote_lookup(options)
    standard = master.filter(
        (pl.col("opt_multiplier") == STANDARD_MULTIPLIER)
        & pl.col("maturity_date").is_between(DEV_START, DEV_END, closed="both")
    )
    cycles: list[dict[str, Any]] = []
    for maturity in sorted(standard["maturity_date"].unique().to_list()):
        before = [trade_date for trade_date in dates if trade_date < maturity]
        if len(before) < SIGNAL_DAYS_BEFORE_EXPIRY + 1:
            continue
        signal_date = before[-SIGNAL_DAYS_BEFORE_EXPIRY]
        prior_date = before[-SIGNAL_DAYS_BEFORE_EXPIRY - 1]
        entry_date = before[-SIGNAL_DAYS_BEFORE_EXPIRY + 1]
        exit_date = before[-EXIT_DAYS_BEFORE_EXPIRY]
        signal = fund_by_date[signal_date]
        if (
            signal.get("prior_ma60") is None
            or signal.get("momentum20") is None
            or float(signal["close"]) <= float(signal["prior_ma60"])
            or float(signal["momentum20"]) < MIN_MOMENTUM
        ):
            continue
        chain = standard.filter(
            (pl.col("maturity_date") == maturity)
            & (pl.col("call_put") == "C")
            & (pl.col("list_date") <= signal_date)
            & (pl.col("delist_date") >= exit_date)
        ).sort(["exercise_price", "contract"])
        strikes = sorted(set(chain["exercise_price"].to_list()))
        if len(strikes) < 2:
            cycles.append(
                _rejected(maturity, signal_date, entry_date, exit_date, "CALL_CHAIN_INCOMPLETE")
            )
            continue
        underlying = float(signal["close"])
        lower_strike = min(strikes, key=lambda strike: (abs(strike - underlying), strike))
        upper = [strike for strike in strikes if strike > lower_strike]
        if not upper:
            cycles.append(
                _rejected(maturity, signal_date, entry_date, exit_date, "UPPER_CALL_UNAVAILABLE")
            )
            continue
        upper_strike = min(upper)
        lower_rows = chain.filter(pl.col("exercise_price") == lower_strike).to_dicts()
        upper_rows = chain.filter(pl.col("exercise_price") == upper_strike).to_dicts()
        lower_leg = lower_rows[0] if lower_rows else None
        upper_leg = upper_rows[0] if upper_rows else None
        if lower_leg is None or upper_leg is None:
            cycles.append(
                _rejected(maturity, signal_date, entry_date, exit_date, "TWO_LEG_CHAIN_INCOMPLETE")
            )
            continue
        liquid = True
        for leg in (lower_leg, upper_leg):
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
                _rejected(maturity, signal_date, entry_date, exit_date, "PRIOR_DAY_LIQUIDITY_FAILED")
            )
            continue
        cycles.append(
            {
                "maturity_date": maturity,
                "signal_date": signal_date,
                "liquidity_date": prior_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "underlying_close": underlying,
                "prior_ma60": float(signal["prior_ma60"]),
                "momentum20": float(signal["momentum20"]),
                "status": "READY",
                "reason": None,
                "legs": {
                    "long_call": {
                        "contract": str(lower_leg["contract"]),
                        "strike": float(lower_leg["exercise_price"]),
                        "tick": float(lower_leg["min_price_chg"]),
                        "multiplier": float(lower_leg["opt_multiplier"]),
                    },
                    "short_call": {
                        "contract": str(upper_leg["contract"]),
                        "strike": float(upper_leg["exercise_price"]),
                        "tick": float(upper_leg["min_price_chg"]),
                        "multiplier": float(upper_leg["opt_multiplier"]),
                    },
                },
            }
        )
    return cycles


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _open_prices(
    cycle: dict[str, Any], quotes: dict[tuple[str, date], dict[str, Any]], trade_date: date
) -> dict[str, float] | None:
    prices = {}
    for name, leg in cycle["legs"].items():
        quote = quotes.get((leg["contract"], trade_date))
        if quote is None or not _positive(quote.get("open")):
            return None
        prices[name] = float(quote["open"])
    return prices


def _spread_value(prices: dict[str, float]) -> float:
    return prices["long_call"] - prices["short_call"]


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _year_returns(
    dates: list[date], equity_curve: dict[date, float], initial: float
) -> dict[str, float]:
    output = {}
    previous = initial
    for year in range(DEV_START.year, DEV_END.year + 1):
        year_dates = [trade_date for trade_date in dates if trade_date.year == year]
        ending = equity_curve[year_dates[-1]] if year_dates else previous
        output[str(year)] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return output


def simulate_account(
    cycles: list[dict[str, Any]],
    options: pl.DataFrame,
    fund_dates: list[date],
    initial_capital: float,
) -> dict[str, Any]:
    quotes = _quote_lookup(options)
    equity = initial_capital
    records: list[dict[str, Any]] = []
    marks: dict[date, float] = {}
    realized: dict[date, float] = {}
    for cycle in cycles:
        record = {**cycle, "equity_before": equity}
        if cycle["status"] != "READY":
            record.update(
                {"spreads": 0, "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0, "equity_after": equity}
            )
            records.append(record)
            continue
        entry_raw = _open_prices(cycle, quotes, cycle["entry_date"])
        exit_raw = _open_prices(cycle, quotes, cycle["exit_date"])
        if entry_raw is None or exit_raw is None:
            record.update(
                {
                    "status": "REJECTED", "reason": "ENTRY_OR_EXIT_OPEN_MISSING",
                    "spreads": 0, "gross_pnl": 0.0, "cost": 0.0,
                    "net_pnl": 0.0, "equity_after": equity,
                }
            )
            records.append(record)
            continue
        legs = cycle["legs"]
        entry = {
            "long_call": entry_raw["long_call"] + legs["long_call"]["tick"],
            "short_call": max(0.0, entry_raw["short_call"] - legs["short_call"]["tick"]),
        }
        exit_prices = {
            "long_call": max(0.0, exit_raw["long_call"] - legs["long_call"]["tick"]),
            "short_call": exit_raw["short_call"] + legs["short_call"]["tick"],
        }
        debit = _spread_value(entry)
        width = legs["short_call"]["strike"] - legs["long_call"]["strike"]
        if debit <= 0 or debit >= width:
            record.update(
                {
                    "status": "REJECTED", "reason": "INVALID_ENTRY_DEBIT",
                    "spreads": 0, "gross_pnl": 0.0, "cost": 0.0,
                    "net_pnl": 0.0, "equity_after": equity,
                }
            )
            records.append(record)
            continue
        multiplier = legs["long_call"]["multiplier"]
        round_trip_fee = 4 * FEE_PER_LEG_SIDE
        max_loss = debit * multiplier + round_trip_fee
        risk_spreads = math.floor(equity * RISK_FRACTION / max_loss)
        cash_spreads = math.floor(equity * CASH_FRACTION / max_loss)
        spreads = max(0, min(risk_spreads, cash_spreads))
        if spreads <= 0:
            record.update(
                {
                    "status": "REJECTED", "reason": "CAPITAL_INSUFFICIENT",
                    "spreads": 0, "gross_pnl": 0.0, "cost": 0.0,
                    "net_pnl": 0.0, "equity_after": equity,
                }
            )
            records.append(record)
            continue
        entry_fee = 2 * FEE_PER_LEG_SIDE * spreads
        exit_fee = 2 * FEE_PER_LEG_SIDE * spreads
        entry_cash = debit * multiplier * spreads + entry_fee
        last_known = dict(entry_raw)
        active_dates = [
            trade_date
            for trade_date in fund_dates
            if cycle["entry_date"] <= trade_date < cycle["exit_date"]
        ]
        for trade_date in active_dates:
            prices = {}
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
                prices[name] = last_known[name]
            marks[trade_date] = (
                record["equity_before"] - entry_cash + _spread_value(prices) * multiplier * spreads
            )
        raw_gross = (
            _spread_value(exit_raw) - _spread_value(entry_raw)
        ) * multiplier * spreads
        exit_credit = _spread_value(exit_prices) * multiplier * spreads
        net_pnl = exit_credit - entry_cash - exit_fee
        cost = raw_gross - net_pnl
        equity += net_pnl
        realized[cycle["exit_date"]] = equity
        marks[cycle["exit_date"]] = equity
        record.update(
            {
                "status": "FILLED", "reason": None, "spreads": spreads,
                "entry_debit_per_unit": debit, "max_loss_per_spread": max_loss,
                "gross_pnl": raw_gross, "cost": cost, "net_pnl": net_pnl,
                "equity_after": equity,
            }
        )
        records.append(record)
    curve = {}
    current = initial_capital
    for trade_date in fund_dates:
        if trade_date in realized:
            current = realized[trade_date]
        curve[trade_date] = marks.get(trade_date, current)
    filled = [record for record in records if record["status"] == "FILLED"]
    yearly = _year_returns(fund_dates, curve, initial_capital)
    gains = [record["net_pnl"] for record in filled if record["net_pnl"] > 0]
    losses = [record["net_pnl"] for record in filled if record["net_pnl"] < 0]
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (equity / initial_capital) ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    ledger_error = equity - initial_capital - sum(record["net_pnl"] for record in records)
    return {
        "initial_capital": initial_capital,
        "ending_equity": equity,
        "total_return": equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown([initial_capital, *curve.values()]),
        "signals": len(records),
        "trades": len(filled),
        "execution_rate": len(filled) / len(records) if records else 0.0,
        "win_rate": sum(record["net_pnl"] > 0 for record in filled) / len(filled) if filled else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "positive_years": sum(value > 0 for value in yearly.values()),
        "year_returns": yearly,
        "total_cost": sum(record["cost"] for record in filled),
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
        "at_least_20_call_spreads": account["trades"] >= 20,
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
        "schema_version": "p0-50etf-call-spread-momentum-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": DEV_START, "end": DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "signal_count": len(cycles),
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
        default=Path("/app/data/research/p0_50etf_call_spread_momentum_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
