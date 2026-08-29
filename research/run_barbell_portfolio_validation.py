"""Validate a fixed 50/50 portfolio of two independent A-share signal sleeves."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import date
from pathlib import Path

import numpy as np

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common
import run_winner_pool_study as winner_study
from app.backtest.strategy import BacktestResultPolicy

SLEEVES = (
    "accumulation_secondary_ignition",
    "breadth_oversold_repair",
)
CANDIDATES = {
    "point_in_time_new_low_reversal": {
        "strategy_id": "reversal_first_principles",
        "params": study.BASELINE_PARAMS,
        "execution": {
            "max_positions": 10,
            "max_hold_days": 15,
            "stop_loss": -0.06,
        },
    },
    "reversal_recovery_p15": winner_study.CANDIDATES["reversal_recovery_p15"],
    **{name: winner_study.CANDIDATES[name] for name in SLEEVES},
}
CURVE_POLICY = BacktestResultPolicy(
    required_stats=common.POLICY.required_stats,
    include_monte_carlo=False,
    include_curves=True,
    include_trades=True,
    include_per_symbol_stats=False,
    include_return_distribution=False,
    include_benchmark=False,
    include_strategy_info=False,
)


def _config(spec: dict, start: date, end: date):
    return common._config(
        spec["strategy_id"],
        start,
        end,
        params=spec["params"],
        basic_filter_override=study.PIT_FILTER,
        **spec["execution"],
    )


def _portfolio_stats(results: dict) -> dict:
    curves = []
    for name in SLEEVES:
        curve = results[name]["equity_curve"]
        if not curve:
            raise RuntimeError(f"{name} returned no equity curve")
        curves.append({row["date"]: float(row["value"]) / 200_000.0 for row in curve})
    dates = sorted(set(curves[0]) & set(curves[1]))
    equity = np.array(
        [0.5 * curves[0][day] + 0.5 * curves[1][day] for day in dates],
        dtype=np.float64,
    )
    returns = np.diff(equity) / equity[:-1]
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(np.mean(returns) / volatility * math.sqrt(252))
        if volatility > 0
        else 0.0
    )
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return {
        "allocation": {name: 0.5 for name in SLEEVES},
        "rebalance": "fixed initial sleeves; no cross-sleeve rebalancing",
        "total_return": round(float(equity[-1] - 1.0), 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(float(np.min(drawdown)), 4),
        "n_trades": sum(int(results[name]["stats"]["n_trades"]) for name in SLEEVES),
    }


def _run_period(service, market, start: date, end: date) -> dict:
    names = list(CANDIDATES)
    configs = [_config(CANDIDATES[name], start, end) for name in names]
    for config in configs:
        config.enforce_t_plus_one = True
    prepared_by_name, prepared_objects = common._prepared_groups(
        service, names, configs, market
    )
    try:
        results = {}
        for name, config in zip(names, configs, strict=True):
            result = service.run(
                config,
                prepared=prepared_by_name[name],
                result_policy=CURVE_POLICY,
            )
            if result.error:
                raise RuntimeError(f"{name}: {result.error}")
            results[name] = {
                "stats": {
                    key: result.stats.get(key) for key in common.POLICY.required_stats
                },
                "equity_curve": result.equity_curve,
                "trades": result.trades,
            }
        return {
            "individual": {
                name: results[name]["stats"] for name in names
            },
            "barbell_portfolio": _portfolio_stats(results),
            "accumulation_trades": results[
                "accumulation_secondary_ignition"
            ]["trades"],
        }
    finally:
        for prepared in prepared_objects:
            prepared.compute_cache.close()


def _fold_gate(fold_input: Path) -> dict:
    payload = json.loads(fold_input.read_text(encoding="utf-8"))
    returns = []
    benchmark = []
    rows = []
    for row in payload["folds"]:
        left = float(row["results"][SLEEVES[0]]["total_return"])
        right = float(row["results"][SLEEVES[1]]["total_return"])
        combined = 0.5 * left + 0.5 * right
        base = float(row["results"]["reversal_recovery_p15"]["total_return"])
        returns.append(combined)
        benchmark.append(base)
        rows.append(
            {
                "label": row["label"],
                "return": round(combined, 6),
                "reversal_p15": base,
                "excess": round(combined - base, 6),
            }
        )
    positive = sum(value > 0 for value in returns)
    beats = sum(
        left > right for left, right in zip(returns, benchmark, strict=True)
    )
    return {
        "folds": len(rows),
        "positive_folds": positive,
        "beats_reversal_p15_folds": beats,
        "median_return": round(statistics.median(returns), 6),
        "median_excess_vs_reversal_p15": round(
            statistics.median(
                left - right
                for left, right in zip(returns, benchmark, strict=True)
            ),
            6,
        ),
        "passes_frozen_gate": (
            positive >= math.ceil(len(rows) * 0.50)
            and beats >= math.ceil(len(rows) * 0.55)
        ),
        "rows": rows,
    }


def run(data_dir: Path, research_dir: Path, fold_input: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader, market, pit_context = winner_study._base_market(
        service, data_dir, winner_study.LONG_LOAD_START
    )
    try:
        periods = {
            "backward_oos_2014_2025": _run_period(
                service, market, date(2014, 1, 1), date(2025, 12, 31)
            ),
            "recent_year": _run_period(service, market, recent.START, recent.END),
            "design_year_2026": _run_period(
                service, market, date(2026, 1, 1), recent.END
            ),
            "long_full": _run_period(
                service, market, date(2014, 1, 1), recent.END
            ),
        }
    finally:
        loader.compute_cache.close()
    payload = {
        "phase": "fixed_barbell_portfolio_validation",
        "execution": "open_t_plus_one with A-share T+1 enforced",
        "point_in_time_context": pit_context,
        "hypothesis": (
            "equal capital to left-side breadth repair and right-side cooled "
            "accumulation re-ignition"
        ),
        "fold_gate": _fold_gate(fold_input),
        "periods": periods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--fold-input",
        type=Path,
        default=Path("/app/data/research/secondary_ignition_validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/barbell_portfolio_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.fold_input, args.output)


if __name__ == "__main__":
    main()
