"""Run the frozen executable Chinese commodity-futures trend study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
BASE_CASH = 200_000.0
CAPITAL_LEVELS = {
    "cny_200k": BASE_CASH,
    "cny_300k": 300_000.0,
    "cny_500k": 500_000.0,
    "cny_1m": 1_000_000.0,
}
MOMENTUM_DAYS = 252
VOLATILITY_DAYS = 20
VOLATILITY_FLOOR = 0.005
GROSS_LEVERAGE = 3.0
MARGIN_RATE = 0.15
MAX_MARGIN_FRACTION = 0.80
COMMISSION_PCT = 0.0002
SLIPPAGE_PCT = 0.0005
TURNOVER_COST = COMMISSION_PCT + SLIPPAGE_PCT
DAILY_PARTICIPATION = 0.01


def prepare_signal_panel(
    continuous: pl.DataFrame,
    mapping: pl.DataFrame,
    contract_daily: pl.DataFrame,
) -> pl.DataFrame:
    dates = continuous.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    quotes = contract_daily.select(
        "contract",
        "date",
        pl.col("open").alias("contract_open"),
        pl.col("settle").alias("contract_settle"),
    )
    return (
        continuous.select("series", "date")
        .join(mapping, on=["series", "date"], how="inner")
        .join(quotes, on=["contract", "date"], how="inner")
        .join(dates, on="date", how="left")
        .sort(["series", "date"])
        .with_columns(
            pl.col("contract").shift(1).over("series").alias("_prev_contract"),
            pl.col("contract_settle")
            .shift(1)
            .over("series")
            .alias("_prev_settle"),
            pl.col("_global_index").shift(1).over("series").alias("_prev_index"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") != pl.col("_prev_index") + 1)
            .then(None)
            .when(pl.col("contract") == pl.col("_prev_contract"))
            .then(pl.col("contract_settle") / pl.col("_prev_settle") - 1.0)
            .otherwise(
                pl.col("contract_settle") / pl.col("contract_open") - 1.0
            )
            .alias("corrected_return"),
            (pl.col("contract") != pl.col("_prev_contract"))
            .fill_null(False)
            .alias("roll_changed"),
        )
        .with_columns(
            (1.0 + pl.col("corrected_return").fill_null(0.0))
            .cum_prod()
            .over("series")
            .alias("return_index"),
            pl.col("corrected_return")
            .rolling_std(window_size=VOLATILITY_DAYS, min_samples=VOLATILITY_DAYS)
            .over("series")
            .alias("volatility_20d"),
        )
        .with_columns(
            pl.col("return_index")
            .shift(MOMENTUM_DAYS)
            .over("series")
            .alias("_index_252d"),
            pl.col("_global_index")
            .shift(MOMENTUM_DAYS)
            .over("series")
            .alias("_global_252d"),
        )
        .with_columns(
            pl.when(
                pl.col("_global_index") == pl.col("_global_252d") + MOMENTUM_DAYS
            )
            .then(pl.col("return_index") / pl.col("_index_252d") - 1.0)
            .otherwise(None)
            .alias("momentum_252d")
        )
    )


def monthly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
        )
        .group_by("month", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )


def build_signals(
    panel: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    eligible = (
        panel.join(
            schedule, left_on="date", right_on="signal_date", how="inner"
        )
        .filter(
            pl.col("momentum_252d").is_not_null()
            & pl.col("volatility_20d").is_not_null()
            & (pl.col("volatility_20d") > 0)
        )
        .with_columns(
            pl.when(pl.col("momentum_252d") > 0)
            .then(1.0)
            .otherwise(-1.0)
            .alias("direction"),
            (1.0 / pl.col("volatility_20d").clip(lower_bound=VOLATILITY_FLOOR))
            .alias("inverse_volatility"),
        )
    )
    return (
        eligible.with_columns(
            pl.col("inverse_volatility").sum().over("date").alias("inverse_sum")
        )
        .with_columns(
            (
                pl.col("direction")
                * GROSS_LEVERAGE
                * pl.col("inverse_volatility")
                / pl.col("inverse_sum")
            ).alias("target_weight")
        )
        .select(
            "date",
            "entry_date",
            "series",
            "momentum_252d",
            "volatility_20d",
            "target_weight",
        )
        .sort(["entry_date", "series"])
    )


def target_quantity(equity: float, weight: float, open_price: float, unit: float) -> int:
    if equity <= 0 or open_price <= 0 or unit <= 0 or weight == 0:
        return 0
    quantity = math.floor(abs(equity * weight) / (open_price * unit))
    return quantity if weight > 0 else -quantity


def _partition(frame: pl.DataFrame, column: str) -> dict[date, list[dict[str, Any]]]:
    output: dict[date, list[dict[str, Any]]] = {}
    for key, group in frame.partition_by(column, as_dict=True).items():
        day = key[0] if isinstance(key, tuple) else key
        output[day] = group.to_dicts()
    return output


def simulate_account(
    signals: pl.DataFrame,
    mapping: pl.DataFrame,
    contract_daily: pl.DataFrame,
    contracts: pl.DataFrame,
    all_dates: list[date],
    initial_cash: float,
) -> dict[str, Any]:
    signal_by_date = _partition(signals, "entry_date")
    mapping_by_date = _partition(mapping, "date")
    quote_by_key = {
        (row["date"], row["contract"]): row
        for row in contract_daily.to_dicts()
    }
    price_limit_locks: dict[tuple[date, str], str] = {}
    lock_columns = {"open", "high", "low", "close", "settle"}
    if lock_columns.issubset(contract_daily.columns):
        lock_rows = (
            contract_daily.sort(["contract", "date"])
            .with_columns(
                pl.col("settle")
                .shift(1)
                .over("contract")
                .alias("previous_settle")
            )
            .filter(
                pl.col("previous_settle").is_not_null()
                & (pl.col("open") == pl.col("high"))
                & (pl.col("open") == pl.col("low"))
                & (pl.col("open") == pl.col("close"))
            )
            .select("date", "contract", "open", "previous_settle")
            .to_dicts()
        )
        for row in lock_rows:
            if row["open"] > row["previous_settle"]:
                price_limit_locks[(row["date"], row["contract"])] = "UP"
            elif row["open"] < row["previous_settle"]:
                price_limit_locks[(row["date"], row["contract"])] = "DOWN"
    unit_by_contract = {
        row["contract"]: float(row["per_unit"])
        for row in contracts.select("contract", "per_unit").to_dicts()
    }
    positions: dict[str, dict[str, Any]] = {}
    equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    missing_quotes = 0
    margin_breaches = 0

    def execute(
        day: date,
        series: str,
        contract: str,
        delta: int,
        price: float,
        volume: float,
        unit: float,
        reason: str,
    ) -> tuple[bool, float]:
        if delta == 0:
            return True, 0.0
        capacity = max(0, math.floor(volume * DAILY_PARTICIPATION))
        lock = price_limit_locks.get((day, contract))
        if delta > 0 and lock == "UP":
            status = "REJECTED_LIMIT_UP_LOCK"
        elif delta < 0 and lock == "DOWN":
            status = "REJECTED_LIMIT_DOWN_LOCK"
        elif abs(delta) > capacity:
            status = "REJECTED_CAPACITY"
        else:
            status = "FILLED"
        filled = status == "FILLED"
        notional = abs(delta) * price * unit
        orders.append(
            {
                "date": day,
                "series": series,
                "contract": contract,
                "delta": delta,
                "reason": reason,
                "status": status,
                "notional": notional,
                "capacity_contracts": capacity,
            }
        )
        return filled, notional * TURNOVER_COST if filled else 0.0

    for day in all_dates:
        starting_equity = equity
        day_pnl = 0.0
        day_cost = 0.0
        mappings = {
            row["series"]: row["contract"]
            for row in mapping_by_date.get(day, [])
        }
        roll_targets: dict[str, float] = {}
        blocked_series: set[str] = set()
        for series in list(positions):
            position = positions[series]
            old_contract = position["contract"]
            old_quote = quote_by_key.get((day, old_contract))
            if old_quote is None or not old_quote.get("open"):
                missing_quotes += 1
                continue
            day_pnl += (
                position["quantity"]
                * position["unit"]
                * (float(old_quote["open"]) - position["last_settle"])
            )
            new_contract = mappings.get(series)
            if new_contract is None or new_contract == old_contract:
                continue
            roll_notional = (
                position["quantity"]
                * float(old_quote["open"])
                * position["unit"]
            )
            filled, cost = execute(
                day,
                series,
                old_contract,
                -position["quantity"],
                float(old_quote["open"]),
                float(old_quote["volume"]),
                position["unit"],
                "roll_close",
            )
            day_cost += cost
            if filled:
                roll_targets[series] = roll_notional
                del positions[series]
            else:
                blocked_series.add(series)

        pre_open_equity = equity + day_pnl - day_cost
        requested_weights = {
            row["series"]: float(row["target_weight"])
            for row in signal_by_date.get(day, [])
        }
        targets: dict[str, int] = {}
        target_reasons: dict[str, str] = {}
        if day in signal_by_date:
            for series, contract in mappings.items():
                if series in blocked_series:
                    continue
                quote = quote_by_key.get((day, contract))
                unit = unit_by_contract.get(contract)
                if quote is None or unit is None:
                    missing_quotes += 1
                    continue
                targets[series] = target_quantity(
                    pre_open_equity,
                    requested_weights.get(series, 0.0),
                    float(quote["open"]),
                    unit,
                )
                target_reasons[series] = "monthly_rebalance"
        else:
            for series, notional in roll_targets.items():
                contract = mappings.get(series)
                quote = quote_by_key.get((day, contract)) if contract else None
                unit = unit_by_contract.get(contract) if contract else None
                if quote is None or unit is None:
                    missing_quotes += 1
                    continue
                quantity = math.floor(abs(notional) / (float(quote["open"]) * unit))
                targets[series] = quantity if notional > 0 else -quantity
                target_reasons[series] = "roll_open"

        for series, target in targets.items():
            contract = mappings[series]
            quote = quote_by_key[(day, contract)]
            unit = unit_by_contract[contract]
            current = positions.get(series)
            current_quantity = (
                int(current["quantity"])
                if current is not None and current["contract"] == contract
                else 0
            )
            delta = target - current_quantity
            filled, cost = execute(
                day,
                series,
                contract,
                delta,
                float(quote["open"]),
                float(quote["volume"]),
                unit,
                target_reasons[series],
            )
            day_cost += cost
            if not filled:
                continue
            if target == 0:
                positions.pop(series, None)
            else:
                positions[series] = {
                    "contract": contract,
                    "quantity": target,
                    "unit": unit,
                    "last_settle": float(quote["open"]),
                }

        for series, position in list(positions.items()):
            quote = quote_by_key.get((day, position["contract"]))
            if quote is None or not quote.get("settle"):
                missing_quotes += 1
                continue
            day_pnl += (
                position["quantity"]
                * position["unit"]
                * (float(quote["settle"]) - float(quote["open"]))
            )
            position["last_settle"] = float(quote["settle"])

        equity += day_pnl - day_cost
        gross_notional = 0.0
        for position in positions.values():
            gross_notional += (
                abs(position["quantity"])
                * position["unit"]
                * position["last_settle"]
            )
        margin = gross_notional * MARGIN_RATE
        if equity <= 0 or margin > max(0.0, equity) * MAX_MARGIN_FRACTION:
            margin_breaches += 1
            forced_cost = gross_notional * TURNOVER_COST
            equity -= forced_cost
            day_cost += forced_cost
            positions.clear()
            margin = 0.0
            gross_notional = 0.0
        daily_rows.append(
            {
                "date": day,
                "equity": equity,
                "daily_return": equity / starting_equity - 1.0
                if starting_equity > 0
                else -1.0,
                "pnl": day_pnl,
                "cost": day_cost,
                "gross_notional": gross_notional,
                "margin": margin,
                "position_count": len(positions),
            }
        )

    final_day = all_dates[-1]
    final_cost = 0.0
    for series, position in list(positions.items()):
        quote = quote_by_key.get((final_day, position["contract"]))
        if quote is None or not quote.get("settle"):
            missing_quotes += 1
            continue
        filled, cost = execute(
            final_day,
            series,
            position["contract"],
            -position["quantity"],
            float(quote["settle"]),
            float(quote["volume"]),
            position["unit"],
            "final_liquidation",
        )
        final_cost += cost
        if filled:
            del positions[series]
    if final_cost:
        equity -= final_cost
        previous_equity = daily_rows[-2]["equity"] if len(daily_rows) > 1 else initial_cash
        daily_rows[-1]["equity"] = equity
        daily_rows[-1]["cost"] += final_cost
        daily_rows[-1]["daily_return"] = equity / previous_equity - 1.0
    final_gross = sum(
        abs(position["quantity"])
        * position["unit"]
        * position["last_settle"]
        for position in positions.values()
    )
    daily_rows[-1]["gross_notional"] = final_gross
    daily_rows[-1]["margin"] = final_gross * MARGIN_RATE
    daily_rows[-1]["position_count"] = len(positions)

    daily = pl.DataFrame(daily_rows, infer_schema_length=None)
    filled = sum(row["status"] == "FILLED" for row in orders)
    rejection_reasons = Counter(
        row["status"] for row in orders if row["status"] != "FILLED"
    )
    return {
        "daily": daily,
        "orders": orders,
        "execution": {
            "orders": len(orders),
            "filled": filled,
            "execution_rate": filled / len(orders) if orders else 1.0,
            "rejections": dict(sorted(rejection_reasons.items())),
        },
        "integrity": {
            "missing_settlement_quotes": missing_quotes,
            "margin_breaches": margin_breaches,
            "ending_positions": len(positions),
        },
        "ending_equity": equity,
    }


def compound(returns: list[float]) -> float | None:
    if not returns:
        return None
    return math.prod(1.0 + value for value in returns) - 1.0


def annualized(returns: list[float]) -> float | None:
    total = compound(returns)
    if total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(returns)) - 1.0


def max_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def summarize_account(result: dict[str, Any]) -> dict[str, Any]:
    daily = result["daily"]
    returns = daily["daily_return"].to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = daily.filter(pl.col("date").dt.year() == year)[
            "daily_return"
        ].to_list()
        value = compound(values)
        positive_years += int(value is not None and value > 0)
        yearly.append({"year": year, "account_return": value})
    return {
        "metrics": {
            "annualized": annualized(returns),
            "total_return": compound(returns),
            "max_drawdown": max_drawdown(returns),
            "positive_years": positive_years,
            "yearly": yearly,
            "mean_gross_leverage": (
                daily["gross_notional"] / daily["equity"]
            ).mean(),
            "maximum_margin_fraction": (daily["margin"] / daily["equity"]).max(),
        },
        "execution": result["execution"],
        "integrity": result["integrity"],
        "ending_equity": result["ending_equity"],
        "total_cost": daily["cost"].sum(),
        "orders": result["orders"],
        "daily_equity": daily.to_dicts(),
    }


def benchmark_metrics(panel: pl.DataFrame) -> dict[str, Any]:
    daily = (
        panel.filter(pl.col("corrected_return").is_not_null())
        .group_by("date")
        .agg(pl.col("corrected_return").mean().alias("return"))
        .sort("date")
    )
    returns = daily["return"].to_list()
    return {
        "name": "twenty_commodity_unlevered_equal_weight_long",
        "annualized": annualized(returns),
        "total_return": compound(returns),
        "max_drawdown": max_drawdown(returns),
    }


def evaluate_gate(
    accounts: dict[str, dict[str, Any]], benchmark: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "all_frozen_capital_levels_present": set(accounts) == set(CAPITAL_LEVELS)
    }
    for name, result in accounts.items():
        metrics = result["metrics"]
        annual = metrics.get("annualized")
        excess = (
            annual - benchmark["annualized"]
            if annual is not None and benchmark.get("annualized") is not None
            else -math.inf
        )
        checks.update(
            {
                f"{name}_annualized_at_least_50pct": (annual or -math.inf) >= 0.50,
                f"{name}_excess_at_least_20pp": excess >= 0.20,
                f"{name}_drawdown_no_worse_than_35pct": (
                    metrics.get("max_drawdown") or -math.inf
                )
                >= -0.35,
                f"{name}_at_least_five_positive_years": metrics[
                    "positive_years"
                ]
                >= 5,
                f"{name}_execution_at_least_95pct": result["execution"][
                    "execution_rate"
                ]
                >= 0.95,
                f"{name}_no_missing_settlements": result["integrity"][
                    "missing_settlement_quotes"
                ]
                == 0,
                f"{name}_no_margin_breach": result["integrity"][
                    "margin_breaches"
                ]
                == 0,
                f"{name}_flat_at_end": result["integrity"]["ending_positions"]
                == 0,
            }
        )
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "cn_commodity_futures"
    continuous = pl.read_parquet(root / "continuous_daily.parquet")
    mapping = pl.read_parquet(root / "main_mapping.parquet")
    contract_daily = pl.read_parquet(root / "contract_daily.parquet")
    contracts = pl.read_parquet(root / "contracts.parquet")
    panel = prepare_signal_panel(continuous, mapping, contract_daily)
    schedule = monthly_schedule(panel)
    signals = build_signals(panel, schedule)
    all_dates = panel["date"].unique().sort().to_list()
    accounts = {
        name: summarize_account(
            simulate_account(
                signals,
                mapping,
                contract_daily,
                contracts,
                all_dates,
                cash,
            )
        )
        for name, cash in CAPITAL_LEVELS.items()
    }
    benchmark = benchmark_metrics(panel)
    decision = evaluate_gate(accounts, benchmark)
    payload = {
        "schema_version": "p0-cn-commodity-futures-trend-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "base_cash_cny": BASE_CASH,
            "capital_levels_cny": CAPITAL_LEVELS,
            "momentum_days": MOMENTUM_DAYS,
            "volatility_days": VOLATILITY_DAYS,
            "volatility_floor": VOLATILITY_FLOOR,
            "gross_leverage": GROSS_LEVERAGE,
            "margin_rate": MARGIN_RATE,
            "maximum_margin_fraction": MAX_MARGIN_FRACTION,
            "commission_pct": COMMISSION_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "daily_participation": DAILY_PARTICIPATION,
            "price_limit_mode": "conservative_full_day_one_price_lock_proxy",
        },
        "data": {
            "series": panel["series"].n_unique(),
            "trading_days": len(all_dates),
            "roll_events": panel.filter(pl.col("roll_changed")).height,
            "signal_rows": signals.height,
            "signal_months": signals["entry_date"].n_unique(),
        },
        "benchmark": benchmark,
        "accounts": accounts,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "data": payload["data"],
                "benchmark": benchmark,
                "accounts": {
                    name: {
                        "metrics": result["metrics"],
                        "execution": result["execution"],
                        "integrity": result["integrity"],
                        "ending_equity": result["ending_equity"],
                        "total_cost": result["total_cost"],
                    }
                    for name, result in accounts.items()
                },
                "decision": decision,
                "output": str(output),
                "sha256": digest,
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
        default=Path(
            "/app/data/research/p0_cn_commodity_futures_trend_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
