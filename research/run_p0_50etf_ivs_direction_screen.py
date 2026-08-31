"""Run the frozen 2016-2024 50ETF IVS direction-information screen."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

ANALYSIS_START = date(2016, 1, 1)
ANALYSIS_END = date(2024, 12, 31)
OOS_START = date(2019, 1, 1)
OOS_SPLIT = date(2022, 1, 1)
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
MIN_TRADING_DAYS_TO_EXPIRY = 5
MAX_RATE_STALENESS_DAYS = 7
MIN_IV = 0.0001
MAX_IV = 5.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
SLIPPAGE_RATE = 0.0005
LOT_SIZE = 100
SHIBOR_TENORS = (
    (1.0 / 365.0, "on"),
    (7.0 / 365.0, "1w"),
    (14.0 / 365.0, "2w"),
    (1.0 / 12.0, "1m"),
    (0.25, "3m"),
    (0.50, "6m"),
    (0.75, "9m"),
    (1.0, "1y"),
)


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def bs_price(
    *, spot: float, strike: float, time_years: float, rate: float, volatility: float, call_put: str
) -> float:
    if min(spot, strike, time_years, volatility) <= 0:
        raise ValueError("Black-Scholes inputs must be positive")
    root_time = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * time_years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    discount = math.exp(-rate * time_years)
    if call_put == "C":
        return spot * normal_cdf(d1) - strike * discount * normal_cdf(d2)
    if call_put == "P":
        return strike * discount * normal_cdf(-d2) - spot * normal_cdf(-d1)
    raise ValueError(f"Unsupported option side: {call_put}")


def implied_volatility(
    *, price: float, spot: float, strike: float, time_years: float, rate: float, call_put: str
) -> float | None:
    if not all(_positive(value) for value in (price, spot, strike, time_years)):
        return None
    discount_strike = strike * math.exp(-rate * time_years)
    if call_put == "C":
        lower_bound = max(0.0, spot - discount_strike)
        upper_bound = spot
    elif call_put == "P":
        lower_bound = max(0.0, discount_strike - spot)
        upper_bound = discount_strike
    else:
        return None
    tolerance = max(1e-10, upper_bound * 1e-10)
    if price < lower_bound - tolerance or price > upper_bound + tolerance:
        return None
    low_price = bs_price(
        spot=spot,
        strike=strike,
        time_years=time_years,
        rate=rate,
        volatility=MIN_IV,
        call_put=call_put,
    )
    high_price = bs_price(
        spot=spot,
        strike=strike,
        time_years=time_years,
        rate=rate,
        volatility=MAX_IV,
        call_put=call_put,
    )
    if price < low_price - tolerance or price > high_price + tolerance:
        return None
    lower = MIN_IV
    upper = MAX_IV
    for _ in range(80):
        middle = (lower + upper) / 2.0
        model = bs_price(
            spot=spot,
            strike=strike,
            time_years=time_years,
            rate=rate,
            volatility=middle,
            call_put=call_put,
        )
        if model < price:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _asof_shibor_lookup(shibor: pl.DataFrame) -> Callable[[date, float], float | None]:
    rows = shibor.sort("date").to_dicts()
    dates = [row["date"] for row in rows]

    def lookup(trade_date: date, maturity_years: float) -> float | None:
        index = bisect.bisect_right(dates, trade_date) - 1
        if index < 0 or (trade_date - dates[index]).days > MAX_RATE_STALENESS_DAYS:
            return None
        row = rows[index]
        _, column = min(SHIBOR_TENORS, key=lambda item: abs(item[0] - maturity_years))
        value = row.get(column)
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value) / 100.0

    return lookup


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def build_daily_ivs(
    master: pl.DataFrame,
    fund: pl.DataFrame,
    options: pl.DataFrame,
    shibor: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    fund = fund.filter(
        pl.col("date").is_between(ANALYSIS_START, ANALYSIS_END, closed="both")
    ).sort("date")
    fund_dates = fund["date"].to_list()
    fund_close = {
        row["date"]: float(row["close"])
        for row in fund.select("date", "close").iter_rows(named=True)
        if _positive(row["close"])
    }
    date_index = {trade_date: index for index, trade_date in enumerate(fund_dates)}
    rate_lookup = _asof_shibor_lookup(shibor)
    terms = master.select(
        "contract",
        "call_put",
        "exercise_price",
        "opt_multiplier",
        "maturity_date",
        "list_date",
        "delist_date",
    )
    joined = (
        options.filter(
            pl.col("date").is_between(ANALYSIS_START, ANALYSIS_END, closed="both")
        )
        .join(terms, on="contract", how="inner", validate="m:1")
        .filter(
            (pl.col("list_date") <= pl.col("date"))
            & (pl.col("delist_date") >= pl.col("date"))
            & (pl.col("exercise_price") > 0)
            & (pl.col("opt_multiplier") > 0)
            & (pl.col("close") > 0)
            & (pl.col("open_interest") > 0)
        )
        .sort(["date", "contract"])
    )
    counters: defaultdict[str, int] = defaultdict(int)
    daily_rows: list[dict[str, Any]] = []
    for frame in joined.partition_by("date", maintain_order=True):
        trade_date = frame["date"][0]
        spot = fund_close.get(trade_date)
        if spot is None:
            counters["missing_underlying"] += 1
            continue
        current_index = date_index[trade_date]
        grouped: defaultdict[tuple[date, float, float], dict[str, list[dict[str, Any]]]]
        grouped = defaultdict(lambda: {"C": [], "P": []})
        for row in frame.iter_rows(named=True):
            maturity = row["maturity_date"]
            maturity_index = bisect.bisect_right(fund_dates, maturity) - 1
            remaining_sessions = maturity_index - current_index
            if remaining_sessions < MIN_TRADING_DAYS_TO_EXPIRY:
                counters["too_close_to_expiry"] += 1
                continue
            key = (
                maturity,
                float(row["exercise_price"]),
                float(row["opt_multiplier"]),
            )
            side = str(row["call_put"])
            if side in {"C", "P"}:
                grouped[key][side].append(row)
        weighted_ivs = 0.0
        weighted_zero = 0.0
        weight_sum = 0.0
        valid_pairs = 0
        for (maturity, strike, _multiplier), sides in grouped.items():
            if not sides["C"] or not sides["P"]:
                counters["unpaired"] += 1
                continue
            selected = {
                side: sorted(
                    rows,
                    key=lambda row: (-float(row["open_interest"]), str(row["contract"])),
                )[0]
                for side, rows in sides.items()
            }
            time_years = max((maturity - trade_date).days / 365.25, 1.0 / 365.25)
            rate = rate_lookup(trade_date, time_years)
            if rate is None:
                counters["missing_rate"] += 1
                continue
            values: dict[str, float] = {}
            zero_values: dict[str, float] = {}
            valid = True
            for side, row in selected.items():
                price = float(row["close"])
                value = implied_volatility(
                    price=price,
                    spot=spot,
                    strike=strike,
                    time_years=time_years,
                    rate=rate,
                    call_put=side,
                )
                zero_value = implied_volatility(
                    price=price,
                    spot=spot,
                    strike=strike,
                    time_years=time_years,
                    rate=0.0,
                    call_put=side,
                )
                if value is None or zero_value is None:
                    counters["iv_not_invertible"] += 1
                    valid = False
                    break
                values[side] = value
                zero_values[side] = zero_value
            if not valid:
                continue
            weight = sum(float(row["open_interest"]) for row in selected.values())
            weighted_ivs += weight * (values["C"] - values["P"])
            weighted_zero += weight * (zero_values["C"] - zero_values["P"])
            weight_sum += weight
            valid_pairs += 1
        if valid_pairs > 0 and weight_sum > 0:
            daily_rows.append(
                {
                    "date": trade_date,
                    "ivs": weighted_ivs / weight_sum,
                    "ivs_zero_rate": weighted_zero / weight_sum,
                    "valid_pairs": valid_pairs,
                    "weight_sum": weight_sum,
                    "underlying_close": spot,
                }
            )
        else:
            counters["no_valid_pairs_day"] += 1
    daily = pl.DataFrame(daily_rows).sort("date") if daily_rows else pl.DataFrame()
    rate_correlation = (
        _correlation(daily["ivs"].to_list(), daily["ivs_zero_rate"].to_list())
        if not daily.is_empty()
        else None
    )
    coverage = daily.height / fund.height if fund.height else 0.0
    return daily, {
        "underlying_days": fund.height,
        "valid_ivs_days": daily.height,
        "valid_day_coverage": coverage,
        "shibor_zero_rate_correlation": rate_correlation,
        "rejection_counts": dict(sorted(counters.items())),
    }


def _period_observations(daily: pl.DataFrame, *, frequency: str) -> list[dict[str, Any]]:
    rows = daily.sort("date").to_dicts()
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        trade_date = row["date"]
        if frequency == "weekly":
            iso = trade_date.isocalendar()
            key = (iso.year, iso.week)
        elif frequency == "monthly":
            key = (trade_date.year, trade_date.month)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        grouped[key] = row
    ends = sorted(grouped.values(), key=lambda row: row["date"])
    output: list[dict[str, Any]] = []
    for current, target in zip(ends, ends[1:], strict=True):
        output.append(
            {
                "signal_date": current["date"],
                "target_date": target["date"],
                "ivs": float(current["ivs"]),
                "realized_log_return": math.log(
                    float(target["underlying_close"]) / float(current["underlying_close"])
                ),
            }
        )
    return output


def _fit_ols(rows: list[dict[str, Any]]) -> tuple[float, float]:
    x = [float(row["ivs"]) for row in rows]
    y = [float(row["realized_log_return"]) for row in rows]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    beta = (
        sum((xvalue - x_mean) * (yvalue - y_mean) for xvalue, yvalue in zip(x, y))
        / denominator
        if denominator > 0
        else 0.0
    )
    return y_mean - beta * x_mean, beta


def recursive_oos_forecasts(
    observations: list[dict[str, Any]], *, min_training: int
) -> list[dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    for current in observations:
        if current["signal_date"] < OOS_START:
            continue
        training = [
            row
            for row in observations
            if row["signal_date"] >= ANALYSIS_START
            and row["target_date"] <= current["signal_date"]
        ]
        if len(training) < min_training:
            continue
        alpha, beta = _fit_ols(training)
        history_mean = sum(float(row["realized_log_return"]) for row in training) / len(training)
        forecasts.append(
            {
                **current,
                "forecast": alpha + beta * float(current["ivs"]),
                "history_mean_forecast": history_mean,
                "alpha": alpha,
                "beta": beta,
                "training_observations": len(training),
                "training_last_target_date": max(row["target_date"] for row in training),
            }
        )
    return forecasts


def forecast_metrics(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    if not forecasts:
        return {
            "observations": 0,
            "oos_r2": None,
            "forecast_realized_correlation": None,
            "negative_beta_share": None,
            "low_minus_high_ivs_mean_return": None,
            "half_correlations": {},
        }
    actual = [float(row["realized_log_return"]) for row in forecasts]
    predicted = [float(row["forecast"]) for row in forecasts]
    benchmark = [float(row["history_mean_forecast"]) for row in forecasts]
    model_error = sum((realized - forecast) ** 2 for realized, forecast in zip(actual, predicted))
    benchmark_error = sum(
        (realized - forecast) ** 2 for realized, forecast in zip(actual, benchmark)
    )
    sorted_ivs = sorted(float(row["ivs"]) for row in forecasts)
    lower = sorted_ivs[len(sorted_ivs) // 3]
    upper = sorted_ivs[(2 * len(sorted_ivs)) // 3]
    low_returns = [
        float(row["realized_log_return"]) for row in forecasts if float(row["ivs"]) <= lower
    ]
    high_returns = [
        float(row["realized_log_return"]) for row in forecasts if float(row["ivs"]) >= upper
    ]
    halves = {
        "2019_2021": [row for row in forecasts if row["signal_date"] < OOS_SPLIT],
        "2022_2024": [row for row in forecasts if row["signal_date"] >= OOS_SPLIT],
    }
    half_correlations = {
        name: _correlation(
            [float(row["forecast"]) for row in rows],
            [float(row["realized_log_return"]) for row in rows],
        )
        for name, rows in halves.items()
    }
    return {
        "observations": len(forecasts),
        "oos_r2": 1.0 - model_error / benchmark_error if benchmark_error > 0 else None,
        "forecast_realized_correlation": _correlation(predicted, actual),
        "negative_beta_share": sum(float(row["beta"]) < 0 for row in forecasts)
        / len(forecasts),
        "low_minus_high_ivs_mean_return": (
            sum(low_returns) / len(low_returns) - sum(high_returns) / len(high_returns)
            if low_returns and high_returns
            else None
        ),
        "half_correlations": half_correlations,
    }


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _annualized_return(initial: float, ending: float, start: date, end: date) -> float:
    years = max((end - start).days / 365.25, 1.0 / 365.25)
    return (ending / initial) ** (1.0 / years) - 1.0 if initial > 0 and ending > 0 else -1.0


def simulate_weekly_timing(
    forecasts: list[dict[str, Any]],
    fund: pl.DataFrame,
    initial_capital: float,
    *,
    forecast_field: str = "forecast",
) -> dict[str, Any]:
    fund_rows = fund.sort("date").to_dicts()
    row_by_date = {row["date"]: row for row in fund_rows}
    fund_dates = [row["date"] for row in fund_rows]
    equity = initial_capital
    marks = [equity]
    records: list[dict[str, Any]] = []
    max_ledger_error = 0.0
    for signal in forecasts:
        before = equity
        forecast = float(signal[forecast_field])
        if forecast <= 0:
            records.append(
                {
                    "signal_date": signal["signal_date"],
                    "target_date": signal["target_date"],
                    "status": "CASH",
                    "forecast": forecast,
                    "shares": 0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            marks.append(equity)
            continue
        signal_index = bisect.bisect_right(fund_dates, signal["signal_date"])
        if signal_index >= len(fund_dates):
            continue
        entry_date = fund_dates[signal_index]
        target = row_by_date.get(signal["target_date"])
        entry = row_by_date.get(entry_date)
        if entry is None or target is None or not _positive(entry.get("open")) or not _positive(target.get("close")):
            records.append(
                {
                    "signal_date": signal["signal_date"],
                    "target_date": signal["target_date"],
                    "status": "REJECTED",
                    "reason": "ENTRY_OPEN_OR_TARGET_CLOSE_MISSING",
                    "forecast": forecast,
                    "shares": 0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            marks.append(equity)
            continue
        buy_price = float(entry["open"]) * (1.0 + SLIPPAGE_RATE)
        sell_price = float(target["close"]) * (1.0 - SLIPPAGE_RATE)
        shares = math.floor(equity / buy_price / LOT_SIZE) * LOT_SIZE
        while shares > 0:
            buy_notional = shares * buy_price
            buy_fee = max(MIN_COMMISSION, buy_notional * COMMISSION_RATE)
            if buy_notional + buy_fee <= equity + 1e-9:
                break
            shares -= LOT_SIZE
        if shares <= 0:
            records.append(
                {
                    "signal_date": signal["signal_date"],
                    "target_date": signal["target_date"],
                    "status": "REJECTED",
                    "reason": "INSUFFICIENT_CASH_FOR_ONE_LOT",
                    "forecast": forecast,
                    "shares": 0,
                    "net_pnl": 0.0,
                    "equity_after": equity,
                }
            )
            marks.append(equity)
            continue
        buy_notional = shares * buy_price
        sell_notional = shares * sell_price
        buy_fee = max(MIN_COMMISSION, buy_notional * COMMISSION_RATE)
        sell_fee = max(MIN_COMMISSION, sell_notional * COMMISSION_RATE)
        residual_cash = equity - buy_notional - buy_fee
        equity = residual_cash + sell_notional - sell_fee
        net_pnl = equity - before
        ledger_error = abs(
            equity
            - (
                before
                + shares * (sell_price - buy_price)
                - buy_fee
                - sell_fee
            )
        )
        max_ledger_error = max(max_ledger_error, ledger_error)
        records.append(
            {
                "signal_date": signal["signal_date"],
                "entry_date": entry_date,
                "target_date": signal["target_date"],
                "status": "TRADED",
                "forecast": forecast,
                "shares": shares,
                "entry_raw": float(entry["open"]),
                "exit_raw": float(target["close"]),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "fees": buy_fee + sell_fee,
                "net_pnl": net_pnl,
                "equity_after": equity,
            }
        )
        marks.append(equity)
    start = forecasts[0]["signal_date"] if forecasts else OOS_START
    end = forecasts[-1]["target_date"] if forecasts else ANALYSIS_END
    return {
        "initial_capital": initial_capital,
        "ending_equity": equity,
        "cumulative_return": equity / initial_capital - 1.0,
        "annualized_return": _annualized_return(initial_capital, equity, start, end),
        "max_drawdown": _max_drawdown(marks),
        "traded_weeks": sum(row["status"] == "TRADED" for row in records),
        "cash_weeks": sum(row["status"] == "CASH" for row in records),
        "rejected_weeks": sum(row["status"] == "REJECTED" for row in records),
        "max_ledger_error": max_ledger_error,
        "records": records,
    }


def simulate_buy_and_hold(
    forecasts: list[dict[str, Any]], fund: pl.DataFrame, initial_capital: float
) -> dict[str, Any]:
    synthetic = [{**row, "buy_hold_forecast": 1.0} for row in forecasts]
    # A continuous weekly rebalance is slightly more costly than literal buy-and-hold, so compute once.
    if not synthetic:
        return {
            "initial_capital": initial_capital,
            "ending_equity": initial_capital,
            "annualized_return": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
        }
    fund = fund.sort("date")
    dates = fund["date"].to_list()
    first_index = bisect.bisect_right(dates, synthetic[0]["signal_date"])
    entry_date = dates[first_index]
    exit_date = synthetic[-1]["target_date"]
    row_by_date = {row["date"]: row for row in fund.iter_rows(named=True)}
    entry_price = float(row_by_date[entry_date]["open"]) * (1.0 + SLIPPAGE_RATE)
    exit_price = float(row_by_date[exit_date]["close"]) * (1.0 - SLIPPAGE_RATE)
    shares = math.floor(initial_capital / entry_price / LOT_SIZE) * LOT_SIZE
    while shares > 0:
        buy_notional = shares * entry_price
        buy_fee = max(MIN_COMMISSION, buy_notional * COMMISSION_RATE)
        if buy_notional + buy_fee <= initial_capital + 1e-9:
            break
        shares -= LOT_SIZE
    buy_notional = shares * entry_price
    buy_fee = max(MIN_COMMISSION, buy_notional * COMMISSION_RATE) if shares else 0.0
    cash = initial_capital - buy_notional - buy_fee
    marks = [initial_capital]
    for signal in synthetic:
        close = float(row_by_date[signal["target_date"]]["close"])
        marks.append(cash + shares * close)
    sell_notional = shares * exit_price
    sell_fee = max(MIN_COMMISSION, sell_notional * COMMISSION_RATE) if shares else 0.0
    ending = cash + sell_notional - sell_fee
    return {
        "initial_capital": initial_capital,
        "ending_equity": ending,
        "cumulative_return": ending / initial_capital - 1.0,
        "annualized_return": _annualized_return(initial_capital, ending, entry_date, exit_date),
        "max_drawdown": _max_drawdown(marks),
        "shares": shares,
        "entry_date": entry_date,
        "exit_date": exit_date,
    }


def evaluate_gate(
    data_audit: dict[str, Any],
    weekly_metrics: dict[str, Any],
    monthly_metrics: dict[str, Any],
    timing: dict[str, Any],
    buy_hold: dict[str, Any],
) -> dict[str, Any]:
    halves = weekly_metrics.get("half_correlations") or {}
    checks = {
        "valid_ivs_day_coverage_at_least_95pct": float(
            data_audit.get("valid_day_coverage") or 0.0
        )
        >= 0.95,
        "rate_proxy_correlation_at_least_95pct": float(
            data_audit.get("shibor_zero_rate_correlation") or -1.0
        )
        >= 0.95,
        "weekly_oos_r2_positive": float(weekly_metrics.get("oos_r2") or -1.0) > 0,
        "monthly_oos_r2_positive": float(monthly_metrics.get("oos_r2") or -1.0) > 0,
        "negative_weekly_beta_share_at_least_90pct": float(
            weekly_metrics.get("negative_beta_share") or 0.0
        )
        >= 0.90,
        "both_oos_halves_positive_correlation": all(
            halves.get(name) is not None and float(halves[name]) > 0
            for name in ("2019_2021", "2022_2024")
        ),
        "low_ivs_outperforms_high_ivs": float(
            weekly_metrics.get("low_minus_high_ivs_mean_return") or -1.0
        )
        > 0,
        "timing_annualized_positive": float(timing.get("annualized_return") or -1.0) > 0,
        "timing_not_worse_than_buy_hold_by_more_than_2pp": float(
            timing.get("annualized_return") or -1.0
        )
        >= float(buy_hold.get("annualized_return") or 0.0) - 0.02,
        "ledger_error_at_most_one_cent": float(timing.get("max_ledger_error") or 0.0)
        <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "decision": "ADVANCE_TO_SEPARATE_AMPLIFIER_CONTRACT" if all(checks.values()) else "TERMINATE_IVS_DIRECTION_FAMILY",
        "strict_qualified_strategy_count_increment": 0,
        "checks": checks,
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "50etf_option_vrp"
    master = pl.read_parquet(root / "contracts.parquet")
    fund = pl.read_parquet(root / "underlying.parquet").filter(
        pl.col("date") <= ANALYSIS_END
    )
    options = pl.read_parquet(str(root / "daily" / "date=*" / "part.parquet")).filter(
        pl.col("date") <= ANALYSIS_END
    )
    shibor = pl.read_parquet(root / "shibor.parquet").filter(pl.col("date") <= ANALYSIS_END)
    daily, data_audit = build_daily_ivs(master, fund, options, shibor)
    weekly_observations = _period_observations(daily, frequency="weekly")
    monthly_observations = _period_observations(daily, frequency="monthly")
    weekly_forecasts = recursive_oos_forecasts(weekly_observations, min_training=100)
    monthly_forecasts = recursive_oos_forecasts(monthly_observations, min_training=24)
    weekly_metrics = forecast_metrics(weekly_forecasts)
    monthly_metrics = forecast_metrics(monthly_forecasts)
    accounts: dict[str, Any] = {}
    for capital in INITIAL_CAPITALS:
        timing = simulate_weekly_timing(weekly_forecasts, fund, capital)
        historical_mean = simulate_weekly_timing(
            weekly_forecasts, fund, capital, forecast_field="history_mean_forecast"
        )
        buy_hold = simulate_buy_and_hold(weekly_forecasts, fund, capital)
        accounts[str(int(capital))] = {
            "ivs_timing": timing,
            "historical_mean_timing": historical_mean,
            "buy_and_hold": buy_hold,
        }
    main = accounts["200000"]
    gate = evaluate_gate(
        data_audit,
        weekly_metrics,
        monthly_metrics,
        main["ivs_timing"],
        main["buy_and_hold"],
    )
    payload = {
        "schema_version": "p0-50etf-ivs-direction-screen-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
            "initial_estimation_end": date(2018, 12, 31),
            "oos_start": OOS_START,
        },
        "data_audit": data_audit,
        "weekly": {
            "metrics": weekly_metrics,
            "forecasts": weekly_forecasts,
        },
        "monthly": {
            "metrics": monthly_metrics,
            "forecasts": monthly_forecasts,
        },
        "accounts": accounts,
        "gate": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({**payload, "sha256": digest}, ensure_ascii=False, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_50etf_ivs_direction_screen.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
