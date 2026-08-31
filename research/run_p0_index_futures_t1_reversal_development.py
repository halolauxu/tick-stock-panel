"""Run the frozen 2015-2020 index-futures T+1 reversal development account."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

DEV_START = date(2015, 1, 1)
DEV_END = date(2020, 12, 31)
INDEX_TO_SERIES = {
    "000300.SH": "IF.CFX",
    "000016.SH": "IH.CFX",
    "000905.SH": "IC.CFX",
}
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
ROLLING_DAYS = 60
TURNOVER_QUANTILE = 0.75
MARGIN_RATE = 0.15
CASH_BUFFER = 0.15
MAX_NOTIONAL_MULTIPLE = 5.5
MAX_VOLUME_PARTICIPATION = 0.001
ROUND_TRIP_COST_RATE = 0.0005


def build_signals(indices: pl.DataFrame) -> pl.DataFrame:
    prepared = (
        indices.filter(pl.col("date").is_between(DEV_START, DEV_END, closed="both"))
        .sort(["instrument", "date"])
        .with_columns(
            (pl.col("close") / pl.col("pre_close") - 1.0).alias("index_return"),
            pl.col("amount")
            .shift(1)
            .rolling_quantile(
                quantile=TURNOVER_QUANTILE,
                window_size=ROLLING_DAYS,
                min_samples=ROLLING_DAYS,
            )
            .over("instrument")
            .alias("prior_amount_q75"),
            pl.col("amount")
            .shift(1)
            .rolling_median(window_size=ROLLING_DAYS, min_samples=ROLLING_DAYS)
            .over("instrument")
            .alias("prior_amount_median"),
        )
        .filter(
            (pl.col("index_return") < 0)
            & (pl.col("amount") > pl.col("prior_amount_q75"))
            & (pl.col("prior_amount_median") > 0)
        )
        .with_columns(
            (
                -pl.col("index_return")
                * pl.col("amount")
                / pl.col("prior_amount_median")
            ).alias("signal_score"),
            pl.col("instrument")
            .replace_strict(INDEX_TO_SERIES)
            .alias("future_series"),
        )
        .rename({"date": "signal_date", "instrument": "index_code"})
    )
    return (
        prepared.sort(["signal_date", "signal_score"], descending=[False, True])
        .unique("signal_date", keep="first")
        .sort("signal_date")
    )


def attach_execution_market(
    signals: pl.DataFrame,
    futures: pl.DataFrame,
    mapping: pl.DataFrame,
    contracts: pl.DataFrame,
) -> pl.DataFrame:
    execution = (
        futures.sort(["instrument", "date"])
        .with_columns(pl.col("date").shift(1).over("instrument").alias("signal_date"))
        .rename(
            {
                "instrument": "future_series",
                "date": "execution_date",
                "open": "future_open",
                "high": "future_high",
                "low": "future_low",
                "close": "future_close",
                "volume": "future_volume",
            }
        )
        .select(
            "future_series",
            "signal_date",
            "execution_date",
            "future_open",
            "future_high",
            "future_low",
            "future_close",
            "future_volume",
        )
    )
    mapped = mapping.rename({"instrument": "future_series", "date": "execution_date"})
    multiplier = contracts.select("contract", "contract_multiplier").unique("contract")
    return (
        signals.join(execution, on=["future_series", "signal_date"], how="left")
        .join(mapped, on=["future_series", "execution_date"], how="left")
        .join(multiplier, on="contract", how="left")
        .sort("signal_date")
    )


def _max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0]
    worst = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _year_returns(
    initial: float, records: list[dict[str, Any]]
) -> dict[str, float]:
    output: dict[str, float] = {}
    previous = initial
    for year in range(DEV_START.year, DEV_END.year + 1):
        values = [
            float(row["equity_after"])
            for row in records
            if row.get("execution_date") is not None
            and row["execution_date"].year == year
        ]
        ending = values[-1] if values else previous
        output[str(year)] = ending / previous - 1.0 if previous > 0 else -1.0
        previous = ending
    return output


def simulate_account(events: pl.DataFrame, initial_capital: float) -> dict[str, Any]:
    equity = initial_capital
    records: list[dict[str, Any]] = []
    equity_values = [equity]
    by_series: dict[str, dict[str, float]] = defaultdict(
        lambda: {"trades": 0.0, "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0}
    )
    for row in events.to_dicts():
        before = equity
        status = "FILLED"
        reason = None
        required = (
            "execution_date",
            "future_open",
            "future_high",
            "future_low",
            "future_close",
            "future_volume",
            "contract",
            "contract_multiplier",
        )
        if any(row.get(column) is None for column in required):
            status = "REJECTED"
            reason = "MARKET_OR_CONTRACT_DATA_MISSING"
        elif row["execution_date"] > DEV_END:
            status = "REJECTED"
            reason = "OUTSIDE_DEVELOPMENT_WINDOW"
        elif (
            row["future_open"] <= 0
            or row["future_close"] <= 0
            or row["future_volume"] <= 0
            or row["future_high"] <= row["future_low"]
        ):
            status = "REJECTED"
            reason = "NO_EXECUTABLE_INTRADAY_MARKET"
        contracts_count = 0
        gross_pnl = 0.0
        cost = 0.0
        net_pnl = 0.0
        notional = 0.0
        if status == "FILLED":
            notional = float(row["future_open"] * row["contract_multiplier"])
            max_margin = math.floor(
                equity * (1.0 - CASH_BUFFER) / (notional * MARGIN_RATE)
            )
            max_leverage = math.floor(equity * MAX_NOTIONAL_MULTIPLE / notional)
            max_volume = math.floor(row["future_volume"] * MAX_VOLUME_PARTICIPATION)
            contracts_count = max(0, min(max_margin, max_leverage, max_volume))
            if contracts_count <= 0:
                status = "REJECTED"
                reason = "CAPITAL_OR_VOLUME_INSUFFICIENT"
            else:
                gross_pnl = (
                    (row["future_close"] - row["future_open"])
                    * row["contract_multiplier"]
                    * contracts_count
                )
                cost = notional * contracts_count * ROUND_TRIP_COST_RATE
                net_pnl = gross_pnl - cost
                equity += net_pnl
                series_metrics = by_series[row["future_series"]]
                series_metrics["trades"] += 1
                series_metrics["gross_pnl"] += gross_pnl
                series_metrics["cost"] += cost
                series_metrics["net_pnl"] += net_pnl
        records.append(
            {
                "signal_date": row["signal_date"],
                "execution_date": row.get("execution_date"),
                "future_series": row["future_series"],
                "contract": row.get("contract"),
                "status": status,
                "reason": reason,
                "contracts": contracts_count,
                "notional_per_contract": notional,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "equity_before": before,
                "equity_after": equity,
            }
        )
        equity_values.append(equity)
        if equity <= 0:
            break
    filled = [row for row in records if row["status"] == "FILLED"]
    years = (DEV_END - DEV_START).days / 365.25
    annualized = (equity / initial_capital) ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    yearly = _year_returns(initial_capital, records)
    ledger_error = equity - initial_capital - sum(row["net_pnl"] for row in records)
    losses = [row["net_pnl"] for row in filled if row["net_pnl"] < 0]
    gains = [row["net_pnl"] for row in filled if row["net_pnl"] > 0]
    return {
        "initial_capital": initial_capital,
        "ending_equity": equity,
        "total_return": equity / initial_capital - 1.0,
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown(equity_values),
        "signals": len(records),
        "trades": len(filled),
        "execution_rate": len(filled) / len(records) if records else 0.0,
        "win_rate": sum(row["net_pnl"] > 0 for row in filled) / len(filled) if filled else 0.0,
        "profit_factor": sum(gains) / -sum(losses) if losses else None,
        "total_cost": sum(row["cost"] for row in filled),
        "ledger_error": ledger_error,
        "positive_years": sum(value > 0 for value in yearly.values()),
        "year_returns": yearly,
        "by_series": dict(by_series),
        "reject_reasons": {
            reason: sum(row["reason"] == reason for row in records)
            for reason in sorted({row["reason"] for row in records if row["reason"]})
        },
        "records": records,
    }


def evaluate_gate(account: dict[str, Any]) -> dict[str, bool]:
    return {
        "annualized_at_least_50pct": account["annualized_return"] >= 0.50,
        "max_drawdown_no_worse_than_25pct": account["max_drawdown"] >= -0.25,
        "at_least_60_trades": account["trades"] >= 60,
        "at_least_4_positive_years": account["positive_years"] >= 4,
        "execution_rate_at_least_90pct": account["execution_rate"] >= 0.90,
        "ledger_balanced": abs(account["ledger_error"]) <= 0.01,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "index_futures_t1_reversal"
    indices = (
        pl.scan_parquet(root / "underlying_indices.parquet")
        .filter(pl.col("date") <= DEV_END)
        .collect()
    )
    futures = (
        pl.scan_parquet(root / "continuous_futures.parquet")
        .filter(pl.col("date") <= DEV_END)
        .collect()
    )
    mapping = (
        pl.scan_parquet(root / "main_mapping.parquet")
        .filter(pl.col("date") <= DEV_END)
        .collect()
    )
    contracts = pl.read_parquet(root / "contracts.parquet")
    signals = build_signals(indices)
    events = attach_execution_market(signals, futures, mapping, contracts)
    accounts = {
        str(int(capital)): simulate_account(events, capital)
        for capital in INITIAL_CAPITALS
    }
    primary = accounts[str(int(INITIAL_CAPITALS[0]))]
    gate = evaluate_gate(primary)
    payload = {
        "schema_version": "p0-index-futures-t1-reversal-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": DEV_START, "end": DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "signal_definition": {
            "return_direction": "negative",
            "prior_turnover_window": ROLLING_DAYS,
            "prior_turnover_quantile": TURNOVER_QUANTILE,
            "same_day_choice": "maximum negative_return_times_amount_over_prior_median",
        },
        "execution": {
            "entry": "next_session_main_future_open",
            "exit": "same_session_main_future_close",
            "margin_rate": MARGIN_RATE,
            "cash_buffer": CASH_BUFFER,
            "max_notional_multiple": MAX_NOTIONAL_MULTIPLE,
            "max_volume_participation": MAX_VOLUME_PARTICIPATION,
            "round_trip_cost_rate": ROUND_TRIP_COST_RATE,
        },
        "signal_count": signals.height,
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
        default=Path("/app/data/research/p0_index_futures_t1_reversal_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
