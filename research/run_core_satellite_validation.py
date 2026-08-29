"""Exact validation of a semiannually rebalanced core-satellite strategy."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import date
from pathlib import Path

import numpy as np

import run_exact_online_year_benchmark as exact
import run_independent_alpha_study as study
import run_reversal_study as common
import run_winner_pool_study as winner_study
from app.backtest.matrix import matrix_feature
from app.strategy import config as strategy_config

START = date(2014, 1, 1)
END = date(2026, 8, 27)
RECENT_START = date(2025, 8, 27)
LOAD_START = date(2013, 1, 1)
RECENT_WINDOWS = {
    "three_months": date(2026, 5, 27),
    "six_months": date(2026, 2, 27),
    "one_year": RECENT_START,
}
WEIGHTS = {
    "online_n_day_low_reversal": 0.65,
    "secondary_ignition": 0.10,
    "accumulation_secondary_ignition": 0.25,
}
SLEEVES = {
    "secondary_ignition": winner_study.CANDIDATES["secondary_ignition"],
    "accumulation_secondary_ignition": winner_study.CANDIDATES[
        "accumulation_secondary_ignition"
    ],
}
FOLDS = tuple(
    (
        f"{year}{half}",
        date(year, 1 if half == "H1" else 7, 1),
        (
            date(year, 6, 30)
            if half == "H1"
            else min(date(year, 12, 31), END)
        ),
    )
    for year in range(2014, 2027)
    for half in ("H1", "H2")
    if date(year, 1 if half == "H1" else 7, 1) <= END
)


def _curve_stats(dates: list[str], equity: np.ndarray) -> dict:
    returns = np.diff(equity) / equity[:-1]
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(np.mean(returns) / volatility * math.sqrt(252))
        if volatility > 0
        else 0.0
    )
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "total_return": round(float(equity[-1] / equity[0] - 1.0), 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(float(np.min(drawdown)), 4),
        "equity_curve": [
            {"date": day, "value": round(float(value), 8)}
            for day, value in zip(dates, equity, strict=True)
        ],
    }


def _portfolio(results: dict) -> dict:
    curves = []
    for name, weight in WEIGHTS.items():
        curve = results[name]["equity_curve"]
        curves.append(
            (
                weight,
                {row["date"]: float(row["value"]) / 200_000.0 for row in curve},
            )
        )
    dates = sorted(set.intersection(*(set(curve) for _, curve in curves)))
    equity = np.array(
        [sum(weight * curve[day] for weight, curve in curves) for day in dates],
        dtype=np.float64,
    )
    return {"weights": WEIGHTS, **_curve_stats(dates, equity)}


def _cash_sweep_portfolio(results: dict) -> dict:
    curve_maps = {
        name: {row["date"]: row for row in result["equity_curve"]}
        for name, result in results.items()
        if name in WEIGHTS
    }
    dates = sorted(set.intersection(*(set(curve) for curve in curve_maps.values())))
    values = {
        name: np.asarray(
            [float(curve_maps[name][day]["value"]) for day in dates],
            dtype=np.float64,
        )
        for name in WEIGHTS
    }
    returns = {
        name: np.divide(
            np.diff(curve),
            curve[:-1],
            out=np.zeros(len(curve) - 1, dtype=np.float64),
            where=curve[:-1] != 0,
        )
        for name, curve in values.items()
    }
    exposures = {
        name: np.asarray(
            [float(curve_maps[name][day].get("exposure", 0.0)) for day in dates],
            dtype=np.float64,
        )
        for name in (
            "secondary_ignition",
            "accumulation_secondary_ignition",
        )
    }
    equity = np.ones(len(dates), dtype=np.float64)
    for index in range(1, len(dates)):
        secondary_reserved = max(
            exposures["secondary_ignition"][index - 1],
            exposures["secondary_ignition"][index],
        )
        accumulation_reserved = max(
            exposures["accumulation_secondary_ignition"][index - 1],
            exposures["accumulation_secondary_ignition"][index],
        )
        core_weight = (
            WEIGHTS["online_n_day_low_reversal"]
            + WEIGHTS["secondary_ignition"] * (1.0 - secondary_reserved)
            + WEIGHTS["accumulation_secondary_ignition"]
            * (1.0 - accumulation_reserved)
        )
        daily_return = (
            core_weight * returns["online_n_day_low_reversal"][index - 1]
            + WEIGHTS["secondary_ignition"]
            * returns["secondary_ignition"][index - 1]
            + WEIGHTS["accumulation_secondary_ignition"]
            * returns["accumulation_secondary_ignition"][index - 1]
        )
        equity[index] = equity[index - 1] * (1.0 + daily_return)
    return {
        "weights": WEIGHTS,
        "idle_satellite_cash": "swept_to_core",
        **_curve_stats(dates, equity),
    }


def _risk_on_by_date(market) -> dict[str, bool]:
    change = matrix_feature(market, "change_pct")
    ma20 = matrix_feature(market, "ma20")
    eligible = matrix_feature(market, "pit_eligible") > np.float32(0.5)
    symbols = np.asarray(market.symbols)
    main_board = np.asarray(
        [
            (symbol.endswith(".SH") and symbol.startswith("60"))
            or (
                symbol.endswith(".SZ")
                and symbol.startswith(("000", "001", "002", "003"))
            )
            for symbol in symbols
        ],
        dtype=bool,
    )
    valid_change = np.isfinite(change) & eligible & main_board[None, :]
    change_count = valid_change.sum(axis=1)
    breadth = np.divide(
        ((change > 0) & valid_change).sum(axis=1),
        change_count,
        out=np.zeros_like(change_count, dtype=np.float32),
        where=change_count > 0,
    )
    valid_ma = (
        np.isfinite(ma20)
        & np.isfinite(market.close)
        & eligible
        & main_board[None, :]
    )
    ma_count = valid_ma.sum(axis=1)
    above_ma20 = np.divide(
        ((market.close > ma20) & valid_ma).sum(axis=1),
        ma_count,
        out=np.zeros_like(ma_count, dtype=np.float32),
        where=ma_count > 0,
    )
    raw = (above_ma20 >= np.float32(0.50)) & (breadth >= np.float32(0.40))
    prior = np.zeros_like(raw, dtype=bool)
    prior[1:] = raw[:-1]
    return {
        label[:10]: bool(value)
        for label, value in zip(market.timestamp_labels, prior, strict=True)
    }


def _regime_cash_sweep_portfolio(
    results: dict, risk_on_by_date: dict[str, bool]
) -> dict:
    curve_maps = {
        name: {row["date"]: row for row in result["equity_curve"]}
        for name, result in results.items()
        if name in WEIGHTS
    }
    dates = sorted(set.intersection(*(set(curve) for curve in curve_maps.values())))
    values = {
        name: np.asarray(
            [float(curve_maps[name][day]["value"]) for day in dates],
            dtype=np.float64,
        )
        for name in WEIGHTS
    }
    returns = {
        name: np.divide(
            np.diff(curve),
            curve[:-1],
            out=np.zeros(len(curve) - 1, dtype=np.float64),
            where=curve[:-1] != 0,
        )
        for name, curve in values.items()
    }
    exposures = {
        name: np.asarray(
            [float(curve_maps[name][day].get("exposure", 0.0)) for day in dates],
            dtype=np.float64,
        )
        for name in (
            "secondary_ignition",
            "accumulation_secondary_ignition",
        )
    }
    equity = np.ones(len(dates), dtype=np.float64)
    for index in range(1, len(dates)):
        core_weight = WEIGHTS["online_n_day_low_reversal"]
        if risk_on_by_date.get(dates[index], False):
            secondary_reserved = max(
                exposures["secondary_ignition"][index - 1],
                exposures["secondary_ignition"][index],
            )
            accumulation_reserved = max(
                exposures["accumulation_secondary_ignition"][index - 1],
                exposures["accumulation_secondary_ignition"][index],
            )
            core_weight += WEIGHTS["secondary_ignition"] * (
                1.0 - secondary_reserved
            ) + WEIGHTS["accumulation_secondary_ignition"] * (
                1.0 - accumulation_reserved
            )
        daily_return = (
            core_weight * returns["online_n_day_low_reversal"][index - 1]
            + WEIGHTS["secondary_ignition"]
            * returns["secondary_ignition"][index - 1]
            + WEIGHTS["accumulation_secondary_ignition"]
            * returns["accumulation_secondary_ignition"][index - 1]
        )
        equity[index] = equity[index - 1] * (1.0 + daily_return)
    return {
        "weights": WEIGHTS,
        "idle_satellite_cash": "swept_to_core_only_in_prior_day_risk_on",
        "risk_on": "prior above_ma20>=0.50 and breadth>=0.40",
        **_curve_stats(dates, equity),
    }


def _online_config(
    data_dir: Path, start: date, end: date, enforce_t_plus_one: bool
):
    override = strategy_config.load_override(data_dir, "n_day_low_reversal")
    config = common._config(
        "n_day_low_reversal",
        start,
        end,
        overrides=override,
        max_positions=10,
        max_hold_days=15,
        stop_loss=-0.06,
    )
    config.enforce_t_plus_one = enforce_t_plus_one
    return config


def _run_one(service, config, prepared) -> dict:
    result = service.run(config, prepared=prepared, result_policy=exact.CURVE_POLICY)
    if result.error:
        dates = [
            label[:10]
            for label in prepared.market_data.timestamp_labels
            if config.start.isoformat() <= label[:10] <= config.end.isoformat()
        ]
        verified_no_signal = (
            "未产生买入信号" in result.error
            or result.error == "no data or no signals"
        ) and bool(dates)
        if not verified_no_signal:
            raise RuntimeError(f"{config.strategy_id}: {result.error}")
        return {
            "stats": {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "verified_no_signal": True,
            },
            "equity_curve": [
                {"date": day, "value": 200_000.0} for day in dates
            ],
        }
    return {
        "stats": {
            key: result.stats.get(key) for key in common.POLICY.required_stats
        },
        "equity_curve": result.equity_curve,
    }


def _run_period(
    service,
    data_dir,
    online_market,
    research_market,
    risk_on_by_date,
    enforce_t_plus_one,
    start,
    end,
):
    online_config = _online_config(
        data_dir, start, end, enforce_t_plus_one
    )
    names = list(SLEEVES)
    configs = [study._config(SLEEVES[name], start, end) for name in names]
    for config in configs:
        config.enforce_t_plus_one = enforce_t_plus_one
    online_prepared = common._prepared(service, [online_config], online_market)
    prepared_by_name, prepared_objects = common._prepared_groups(
        service, names, configs, research_market
    )
    try:
        results = {
            "online_n_day_low_reversal": _run_one(
                service, online_config, online_prepared
            )
        }
        results.update(
            {
                name: _run_one(service, config, prepared_by_name[name])
                for name, config in zip(names, configs, strict=True)
            }
        )
        portfolio = _portfolio(results)
        cash_sweep = _cash_sweep_portfolio(results)
        regime_sweep = _regime_cash_sweep_portfolio(results, risk_on_by_date)
        return {
            "individual": {
                name: row["stats"] for name, row in results.items()
            },
            "portfolio": {
                key: value
                for key, value in portfolio.items()
                if key != "equity_curve"
            },
            "portfolio_curve": portfolio["equity_curve"],
            "cash_sweep": {
                key: value
                for key, value in cash_sweep.items()
                if key != "equity_curve"
            },
            "cash_sweep_curve": cash_sweep["equity_curve"],
            "regime_sweep": {
                key: value
                for key, value in regime_sweep.items()
                if key != "equity_curve"
            },
            "regime_sweep_curve": regime_sweep["equity_curve"],
        }
    finally:
        online_prepared.compute_cache.close()
        for prepared in prepared_objects:
            prepared.compute_cache.close()


def _stitch(rows: list[dict], curve_key: str) -> dict:
    dates = []
    values = []
    level = 1.0
    for row in rows:
        curve = row[curve_key]
        base = float(curve[0]["value"])
        for point in curve:
            dates.append(point["date"])
            values.append(level * float(point["value"]) / base)
        level = values[-1]
    return _curve_stats(dates, np.asarray(values, dtype=np.float64))


def _fold_summary(rows: list[dict], result_key: str) -> dict:
    candidate = [float(row[result_key]["total_return"]) for row in rows]
    benchmark = [
        float(row["individual"]["online_n_day_low_reversal"]["total_return"])
        for row in rows
    ]
    positive = sum(value > 0 for value in candidate)
    benchmark_positive = sum(value > 0 for value in benchmark)
    beats = sum(
        left > right for left, right in zip(candidate, benchmark, strict=True)
    )
    median_excess = statistics.median(
        left - right for left, right in zip(candidate, benchmark, strict=True)
    )
    return {
        "folds": len(rows),
        "positive_folds": positive,
        "benchmark_positive_folds": benchmark_positive,
        "beats_online_new_low_folds": beats,
        "median_return": round(statistics.median(candidate), 6),
        "median_excess_vs_online_new_low": round(median_excess, 6),
        "passes_frozen_gate": (
            positive >= max(benchmark_positive, math.ceil(len(rows) * 0.50))
            and beats >= math.ceil(len(rows) * 0.55)
            and median_excess > 0
        ),
    }


def run(
    data_dir: Path,
    research_dir: Path,
    output: Path,
    enforce_t_plus_one: bool = False,
) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                LOAD_START,
                END,
                params={**study.BASELINE_PARAMS, "eligibility_mode": "none"},
                basic_filter_override=study.PIT_FILTER,
            )
        ],
    )
    online_market = common._attach_industry_context(loader.market_data, data_dir)
    research_market, pit_context = common._attach_point_in_time_universe(
        online_market, data_dir
    )
    risk_on = _risk_on_by_date(research_market)
    try:
        recent = {
            label: _run_period(
                service,
                data_dir,
                online_market,
                research_market,
                risk_on,
                enforce_t_plus_one,
                start,
                END,
            )
            for label, start in RECENT_WINDOWS.items()
        }
        long_fixed = _run_period(
            service,
            data_dir,
            online_market,
            research_market,
            risk_on,
            enforce_t_plus_one,
            START,
            END,
        )
        folds = []
        for label, start, end in FOLDS:
            row = _run_period(
                service,
                data_dir,
                online_market,
                research_market,
                risk_on,
                enforce_t_plus_one,
                start,
                end,
            )
            folds.append(
                {"label": label, "range": [start.isoformat(), end.isoformat()], **row}
            )
    finally:
        loader.compute_cache.close()

    fixed_rebalanced = _stitch(folds, "portfolio_curve")
    cash_sweep_rebalanced = _stitch(folds, "cash_sweep_curve")
    regime_sweep_rebalanced = _stitch(folds, "regime_sweep_curve")
    fold_summary = _fold_summary(folds, "regime_sweep")
    long_benchmark = long_fixed["individual"]["online_n_day_low_reversal"]
    recent_passes = all(
        float(row["regime_sweep"]["total_return"])
        > float(row["individual"]["online_n_day_low_reversal"]["total_return"])
        and float(row["regime_sweep"]["total_return"]) > 0
        and float(row["regime_sweep"]["sharpe"])
        > float(row["individual"]["online_n_day_low_reversal"]["sharpe"])
        and float(row["regime_sweep"]["max_drawdown"])
        > float(row["individual"]["online_n_day_low_reversal"]["max_drawdown"])
        for row in recent.values()
    )
    passes = (
        fold_summary["passes_frozen_gate"]
        and recent_passes
        and float(regime_sweep_rebalanced["total_return"])
        > float(long_benchmark["total_return"])
        and float(regime_sweep_rebalanced["sharpe"])
        > float(long_benchmark["sharpe"])
        and float(regime_sweep_rebalanced["max_drawdown"])
        > float(long_benchmark["max_drawdown"])
    )
    for row in folds:
        row.pop("portfolio_curve", None)
        row.pop("cash_sweep_curve", None)
        row.pop("regime_sweep_curve", None)
    fixed_rebalanced.pop("equity_curve", None)
    cash_sweep_rebalanced.pop("equity_curve", None)
    regime_sweep_rebalanced.pop("equity_curve", None)
    for row in recent.values():
        row.pop("portfolio_curve", None)
        row.pop("cash_sweep_curve", None)
        row.pop("regime_sweep_curve", None)
    long_fixed.pop("portfolio_curve", None)
    long_fixed.pop("cash_sweep_curve", None)
    long_fixed.pop("regime_sweep_curve", None)
    payload = {
        "phase": "core_satellite_exact_validation",
        "strategy": {
            "name": "新低核心_双点火卫星_现金回流",
            "weights": WEIGHTS,
            "rebalance": "semiannual",
            "idle_satellite_cash": "swept_to_core_only_in_prior_day_risk_on",
        },
        "execution": (
            "next open with A-share T+1 sell lock enabled"
            if enforce_t_plus_one
            else "online page contract with same-day T+1 sell lock disabled"
        ),
        "point_in_time_context": pit_context,
        "recent_windows": recent,
        "long_fixed_initial_weights": long_fixed,
        "long_fixed_allocation_rebalanced": fixed_rebalanced,
        "long_cash_sweep_rebalanced": cash_sweep_rebalanced,
        "long_regime_sweep_rebalanced": regime_sweep_rebalanced,
        "fold_summary": fold_summary,
        "folds": folds,
        "passes_all_gates": passes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "recent_windows": recent,
                "long_fixed_allocation_rebalanced": fixed_rebalanced,
                "long_cash_sweep_rebalanced": cash_sweep_rebalanced,
                "long_regime_sweep_rebalanced": regime_sweep_rebalanced,
                "long_benchmark": long_benchmark,
                "fold_summary": fold_summary,
                "passes_all_gates": passes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/core_satellite_validation.json"),
    )
    parser.add_argument("--enforce-t-plus-one", action="store_true")
    args = parser.parse_args()
    run(
        args.data_dir,
        args.research_dir,
        args.output,
        enforce_t_plus_one=args.enforce_t_plus_one,
    )


if __name__ == "__main__":
    main()
