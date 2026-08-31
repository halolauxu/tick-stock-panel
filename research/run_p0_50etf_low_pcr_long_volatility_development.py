"""Run the frozen development-only 50ETF low-PCR long-volatility study."""

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
PCR_WINDOW = 20
PCR_THRESHOLD = 2.0
MIN_DAYS_TO_EXPIRY = 5
MIN_VOLUME = 100.0
MIN_OPEN_INTEREST = 500.0
FEE_PER_LEG_SIDE = 5.0
RISK_FRACTION = 0.05
CASH_FRACTION = 0.50


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _quote_lookup(options: pl.DataFrame) -> dict[tuple[str, date], dict[str, Any]]:
    return {
        (str(row["contract"]), row["date"]): row
        for row in options.iter_rows(named=True)
    }


def build_pcr_signals(
    master: pl.DataFrame, fund: pl.DataFrame, options: pl.DataFrame
) -> pl.DataFrame:
    """Calculate close-known OI PCR and a causal prior-20-day z-score."""
    standard = master.filter(pl.col("opt_multiplier") == STANDARD_MULTIPLIER).select(
        "contract", "call_put"
    )
    daily = (
        options.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .join(standard, on="contract", how="inner", validate="m:1")
        .group_by("date")
        .agg(
            pl.when(pl.col("call_put") == "P")
            .then(pl.col("open_interest"))
            .otherwise(0.0)
            .sum()
            .alias("put_open_interest"),
            pl.when(pl.col("call_put") == "C")
            .then(pl.col("open_interest"))
            .otherwise(0.0)
            .sum()
            .alias("call_open_interest"),
        )
        .filter((pl.col("put_open_interest") > 0) & (pl.col("call_open_interest") > 0))
        .with_columns(
            (pl.col("put_open_interest") / pl.col("call_open_interest")).alias("pcr_oi")
        )
        .sort("date")
        .with_columns(
            pl.col("pcr_oi")
            .shift(1)
            .rolling_mean(window_size=PCR_WINDOW, min_samples=PCR_WINDOW)
            .alias("pcr_mean_20"),
            pl.col("pcr_oi")
            .shift(1)
            .rolling_std(window_size=PCR_WINDOW, min_samples=PCR_WINDOW, ddof=1)
            .alias("pcr_std_20"),
        )
        .with_columns(
            ((pl.col("pcr_oi") - pl.col("pcr_mean_20")) / pl.col("pcr_std_20")).alias(
                "pcr_z"
            )
        )
    )
    return (
        daily.join(
            fund.select("date", pl.col("close").alias("underlying_close")),
            on="date",
            how="inner",
            validate="1:1",
        )
        .filter(pl.col("pcr_z").is_finite())
        .with_columns(
            pl.when(pl.col("pcr_z") < -PCR_THRESHOLD)
            .then(pl.lit("LOW"))
            .when(pl.col("pcr_z") > PCR_THRESHOLD)
            .then(pl.lit("HIGH"))
            .otherwise(pl.lit("NEUTRAL"))
            .alias("regime")
        )
        .sort("date")
    )


def _rejected_trade(signal: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **signal,
        "signal_date": signal["date"],
        "entry_date": None,
        "maturity_date": None,
        "status": "REJECTED",
        "reason": reason,
        "legs": None,
    }


def _one_contract(
    chain: pl.DataFrame, call_put: str, strike: float
) -> dict[str, Any] | None:
    rows = (
        chain.filter(
            (pl.col("call_put") == call_put) & (pl.col("exercise_price") == strike)
        )
        .sort("contract")
        .to_dicts()
    )
    return rows[0] if rows else None


def build_trades(
    signals: pl.DataFrame,
    master: pl.DataFrame,
    fund: pl.DataFrame,
    options: pl.DataFrame,
    *,
    regime: str,
) -> list[dict[str, Any]]:
    """Select signal-close ATM pairs without using execution-day information."""
    standard = master.filter(pl.col("opt_multiplier") == STANDARD_MULTIPLIER)
    fund_dates = (
        fund.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .sort("date")["date"]
        .to_list()
    )
    next_date = {
        current: following for current, following in zip(fund_dates, fund_dates[1:])
    }
    quotes = _quote_lookup(options)
    records: list[dict[str, Any]] = []
    for signal in signals.filter(pl.col("regime") == regime).iter_rows(named=True):
        signal_date = signal["date"]
        entry_date = next_date.get(signal_date)
        if entry_date is None:
            records.append(_rejected_trade(signal, "NO_NEXT_TRADING_DAY"))
            continue
        maturities = sorted(
            maturity
            for maturity in standard.filter(
                (pl.col("list_date") <= signal_date)
                & (pl.col("delist_date") >= entry_date)
                & (pl.col("maturity_date") >= entry_date)
            )["maturity_date"]
            .unique()
            .to_list()
            if sum(signal_date < value <= maturity for value in fund_dates)
            >= MIN_DAYS_TO_EXPIRY
        )
        if not maturities:
            records.append(_rejected_trade(signal, "NO_ELIGIBLE_MATURITY"))
            continue
        maturity = maturities[0]
        chain = standard.filter(
            (pl.col("maturity_date") == maturity)
            & (pl.col("list_date") <= signal_date)
            & (pl.col("delist_date") >= entry_date)
        )
        call_strikes = set(
            chain.filter(pl.col("call_put") == "C")["exercise_price"].to_list()
        )
        put_strikes = set(
            chain.filter(pl.col("call_put") == "P")["exercise_price"].to_list()
        )
        paired = sorted(call_strikes.intersection(put_strikes))
        if not paired:
            records.append(_rejected_trade(signal, "NO_PAIRED_ATM_STRIKE"))
            continue
        underlying_close = float(signal["underlying_close"])
        strike = min(paired, key=lambda value: (abs(value - underlying_close), value))
        selected = {
            "call": _one_contract(chain, "C", strike),
            "put": _one_contract(chain, "P", strike),
        }
        if any(leg is None for leg in selected.values()):
            records.append(_rejected_trade(signal, "ATM_PAIR_INCOMPLETE"))
            continue
        liquid = True
        for leg in selected.values():
            assert leg is not None
            quote = quotes.get((str(leg["contract"]), signal_date))
            if (
                quote is None
                or not _positive(quote.get("volume"))
                or not _positive(quote.get("open_interest"))
                or float(quote["volume"]) < MIN_VOLUME
                or float(quote["open_interest"]) < MIN_OPEN_INTEREST
            ):
                liquid = False
                break
        if not liquid:
            records.append(_rejected_trade(signal, "SIGNAL_DAY_LIQUIDITY_FAILED"))
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
        records.append(
            {
                **signal,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "maturity_date": maturity,
                "status": "READY",
                "reason": None,
                "legs": legs,
            }
        )
    return records


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _year_returns(
    fund_dates: list[date], equity_curve: dict[date, float], initial: float
) -> dict[str, float]:
    output: dict[str, float] = {}
    previous = initial
    for year in range(DEV_START.year, DEV_END.year + 1):
        dates = [value for value in fund_dates if value.year == year]
        ending = equity_curve[dates[-1]] if dates else previous
        output[str(year)] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return output


def simulate_account(
    trades: list[dict[str, Any]],
    options: pl.DataFrame,
    fund_dates: list[date],
    initial_capital: float,
) -> dict[str, Any]:
    quotes = _quote_lookup(options)
    equity = initial_capital
    realized_after: dict[date, float] = {}
    records: list[dict[str, Any]] = []
    max_ledger_error = 0.0
    for trade in trades:
        record = {**trade, "equity_before": equity}
        if trade["status"] != "READY":
            record.update(
                contracts=0,
                gross_pnl=0.0,
                cost=0.0,
                net_pnl=0.0,
                equity_after=equity,
            )
            records.append(record)
            continue
        entry_raw: dict[str, float] = {}
        exit_raw: dict[str, float] = {}
        valid = True
        for name, leg in trade["legs"].items():
            quote = quotes.get((leg["contract"], trade["entry_date"]))
            if (
                quote is None
                or not _positive(quote.get("open"))
                or not _positive(quote.get("close"))
            ):
                valid = False
                break
            entry_raw[name] = float(quote["open"])
            exit_raw[name] = float(quote["close"])
        if not valid:
            record.update(
                status="REJECTED",
                reason="ENTRY_OPEN_OR_CLOSE_MISSING",
                contracts=0,
                gross_pnl=0.0,
                cost=0.0,
                net_pnl=0.0,
                equity_after=equity,
            )
            records.append(record)
            continue
        entry_debit = sum(
            entry_raw[name] + trade["legs"][name]["tick"] for name in trade["legs"]
        )
        exit_credit = sum(
            max(0.0, exit_raw[name] - trade["legs"][name]["tick"])
            for name in trade["legs"]
        )
        multiplier = float(trade["legs"]["call"]["multiplier"])
        round_trip_fee = 4 * FEE_PER_LEG_SIDE
        entry_fee = 2 * FEE_PER_LEG_SIDE
        worst_loss = entry_debit * multiplier + round_trip_fee
        entry_cash = entry_debit * multiplier + entry_fee
        risk_contracts = math.floor(equity * RISK_FRACTION / worst_loss)
        cash_contracts = math.floor(equity * CASH_FRACTION / entry_cash)
        contracts = max(0, min(risk_contracts, cash_contracts))
        if contracts <= 0:
            record.update(
                status="REJECTED",
                reason="CAPITAL_INSUFFICIENT",
                contracts=0,
                gross_pnl=0.0,
                cost=0.0,
                net_pnl=0.0,
                equity_after=equity,
            )
            records.append(record)
            continue
        raw_gross = (
            (sum(exit_raw.values()) - sum(entry_raw.values())) * multiplier * contracts
        )
        net_pnl = (
            (exit_credit - entry_debit) * multiplier - round_trip_fee
        ) * contracts
        cost = raw_gross - net_pnl
        before = equity
        equity += net_pnl
        max_ledger_error = max(max_ledger_error, abs(equity - before - net_pnl))
        realized_after[trade["entry_date"]] = equity
        record.update(
            status="FILLED",
            reason=None,
            contracts=contracts,
            entry_debit_per_unit=entry_debit,
            exit_credit_per_unit=exit_credit,
            worst_loss_per_contract=worst_loss,
            gross_pnl=raw_gross,
            cost=cost,
            net_pnl=net_pnl,
            equity_after=equity,
        )
        records.append(record)

    curve: dict[date, float] = {}
    current = initial_capital
    for trade_date in fund_dates:
        current = realized_after.get(trade_date, current)
        curve[trade_date] = current
    filled = [record for record in records if record["status"] == "FILLED"]
    positive_pnl = [
        float(record["net_pnl"]) for record in filled if record["net_pnl"] > 0
    ]
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (
        (equity / initial_capital) ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    )
    yearly = _year_returns(fund_dates, curve, initial_capital)
    return {
        "initial_capital": initial_capital,
        "ending_equity": equity,
        "total_return": equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown([initial_capital, *curve.values()]),
        "signals": len(records),
        "trades": len(filled),
        "execution_rate": len(filled) / len(records) if records else 0.0,
        "win_rate": (
            sum(record["net_pnl"] > 0 for record in filled) / len(filled)
            if filled
            else 0.0
        ),
        "positive_years": sum(value > 0 for value in yearly.values()),
        "year_returns": yearly,
        "total_gross_pnl": sum(float(record["gross_pnl"]) for record in filled),
        "total_cost": sum(float(record["cost"]) for record in filled),
        "largest_positive_trade_share": (
            max(positive_pnl) / sum(positive_pnl) if positive_pnl else None
        ),
        "max_ledger_error": max_ledger_error,
        "reject_reasons": {
            reason: sum(record["reason"] == reason for record in records)
            for reason in sorted(
                {record["reason"] for record in records if record["reason"]}
            )
        },
        "records": records,
    }


def benchmark_metrics(fund: pl.DataFrame) -> dict[str, float]:
    values = (
        fund.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .sort("date")["close"]
        .to_list()
    )
    years = (DEV_END - DEV_START).days / 365.25
    total = float(values[-1]) / float(values[0]) - 1.0
    annualized = (1.0 + total) ** (1.0 / years) - 1.0
    return {"total_return": total, "annualized_return": annualized}


def evaluate_gate(
    candidate: dict[str, Any], control: dict[str, Any], benchmark: dict[str, float]
) -> dict[str, Any]:
    annualized = candidate["annualized_return"]
    concentration = candidate["largest_positive_trade_share"]
    checks = {
        "annualized_at_least_50pct": annualized >= 0.50,
        "benchmark_excess_at_least_20pp": (
            annualized - benchmark["annualized_return"] >= 0.20
        ),
        "high_pcr_control_increment_at_least_10pp": (
            annualized - control["annualized_return"] >= 0.10
        ),
        "max_drawdown_no_worse_than_25pct": candidate["max_drawdown"] >= -0.25,
        "at_least_four_positive_years": candidate["positive_years"] >= 4,
        "at_least_30_straddles": candidate["trades"] >= 30,
        "execution_rate_at_least_90pct": candidate["execution_rate"] >= 0.90,
        "largest_positive_trade_share_at_most_25pct": (
            concentration is not None and concentration <= 0.25
        ),
        "ledger_balanced": candidate["max_ledger_error"] <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "counts_toward_50pct_goal": False,
        "next_step": (
            "freeze_independent_validation"
            if all(checks.values())
            else "terminate_low_pcr_long_volatility"
        ),
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
    signals = build_pcr_signals(master, fund, options)
    low_trades = build_trades(signals, master, fund, options, regime="LOW")
    high_trades = build_trades(signals, master, fund, options, regime="HIGH")
    fund_dates = (
        fund.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .sort("date")["date"]
        .to_list()
    )
    accounts: dict[str, Any] = {}
    for capital in INITIAL_CAPITALS:
        accounts[str(int(capital))] = {
            "low_pcr_long_straddle": simulate_account(
                low_trades, options, fund_dates, capital
            ),
            "high_pcr_control": simulate_account(
                high_trades, options, fund_dates, capital
            ),
        }
    benchmark = benchmark_metrics(fund)
    primary = accounts["200000"]
    decision = evaluate_gate(
        primary["low_pcr_long_straddle"], primary["high_pcr_control"], benchmark
    )
    payload = {
        "schema_version": "p0-50etf-low-pcr-long-volatility-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "start": DEV_START,
            "end": DEV_END,
            "validation_read": False,
            "pressure_read": False,
        },
        "assumptions": {
            "pcr": "sum put open interest / sum call open interest",
            "window": PCR_WINDOW,
            "threshold_standard_deviations": PCR_THRESHOLD,
            "minimum_days_to_expiry": MIN_DAYS_TO_EXPIRY,
            "risk_fraction": RISK_FRACTION,
            "cash_fraction": CASH_FRACTION,
        },
        "data": {
            "pcr_days": signals.height,
            "low_pcr_signals": len(low_trades),
            "high_pcr_signals": len(high_trades),
        },
        "benchmark": benchmark,
        "accounts": accounts,
        "decision": decision,
        "strict_qualified_count": 0,
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
            capital: {
                variant: {
                    field: value
                    for field, value in result.items()
                    if field != "records"
                }
                for variant, result in variants.items()
            }
            for capital, variants in accounts.items()
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
            "/app/data/research/p0_50etf_low_pcr_long_volatility_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
