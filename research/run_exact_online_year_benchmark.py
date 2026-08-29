"""Replay the exact online one-year benchmark against research candidates."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

import run_independent_alpha_study as study
import run_reversal_study as common
import run_winner_pool_study as winner_study
from app.backtest.strategy import BacktestResultPolicy
from app.strategy import config as strategy_config

START = date(2025, 8, 27)
END = date(2026, 8, 27)
LOAD_START = date(2024, 8, 27)
RESEARCH_CANDIDATES = {
    "reversal_recovery_p15": winner_study.CANDIDATES["reversal_recovery_p15"],
    "sentiment_anti_chase": winner_study.CANDIDATES["sentiment_anti_chase"],
    "secondary_ignition": winner_study.CANDIDATES["secondary_ignition"],
    "accumulation_secondary_ignition": winner_study.CANDIDATES[
        "accumulation_secondary_ignition"
    ],
    "breadth_oversold_repair": winner_study.CANDIDATES[
        "breadth_oversold_repair"
    ],
}
CURVE_POLICY = BacktestResultPolicy(
    required_stats=common.POLICY.required_stats,
    include_monte_carlo=False,
    include_curves=True,
    include_trades=False,
    include_per_symbol_stats=False,
    include_return_distribution=False,
    include_benchmark=False,
    include_strategy_info=False,
)


def _portfolio(results: dict, weights: dict[str, float]) -> dict:
    curves = []
    for name, weight in weights.items():
        curve = results[name]["equity_curve"]
        if not curve:
            raise RuntimeError(f"{name} returned no equity curve")
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
    returns = np.diff(equity) / equity[:-1]
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(np.mean(returns) / volatility * math.sqrt(252))
        if volatility > 0
        else 0.0
    )
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "weights": weights,
        "total_return": round(float(equity[-1] - 1.0), 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(float(np.min(drawdown)), 4),
    }


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    online_override = strategy_config.load_override(data_dir, "n_day_low_reversal")
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
    online_config = common._config(
        "n_day_low_reversal",
        START,
        END,
        overrides=online_override,
        max_positions=10,
        max_hold_days=15,
        stop_loss=-0.06,
    )
    names = list(RESEARCH_CANDIDATES)
    configs = [
        study._config(RESEARCH_CANDIDATES[name], START, END) for name in names
    ]
    # StrategyBacktest.tsx does not send enforce_t_plus_one. The API model keeps
    # its compatibility default (False), so the online page can sell a position
    # on its acquisition day when an exit condition is met. Mirror that exact
    # page contract here instead of imposing paper-trading semantics.
    for config in [online_config, *configs]:
        config.enforce_t_plus_one = False
    online_prepared = common._prepared(service, [online_config], online_market)
    prepared_by_name, prepared_objects = common._prepared_groups(
        service,
        names,
        configs,
        research_market,
    )
    try:
        results = {}
        online_result = service.run(
            online_config,
            prepared=online_prepared,
            result_policy=CURVE_POLICY,
        )
        if online_result.error:
            raise RuntimeError(
                f"online_n_day_low_reversal: {online_result.error}"
            )
        results["online_n_day_low_reversal"] = {
            "stats": {
                key: online_result.stats.get(key)
                for key in common.POLICY.required_stats
            },
            "equity_curve": online_result.equity_curve,
        }
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
            }
        portfolios = {
            "core_satellite_65_10_25": _portfolio(
                results,
                {
                    "online_n_day_low_reversal": 0.65,
                    "secondary_ignition": 0.10,
                    "accumulation_secondary_ignition": 0.25,
                },
            ),
            "breadth_secondary_75_25": _portfolio(
                results,
                {
                    "breadth_oversold_repair": 0.75,
                    "secondary_ignition": 0.25,
                },
            ),
            "breadth_accumulation_50_50": _portfolio(
                results,
                {
                    "breadth_oversold_repair": 0.5,
                    "accumulation_secondary_ignition": 0.5,
                },
            ),
            "three_sleeve_equal": _portfolio(
                results,
                {
                    "breadth_oversold_repair": 1 / 3,
                    "accumulation_secondary_ignition": 1 / 3,
                    "secondary_ignition": 1 / 3,
                },
            ),
        }
        payload = {
            "phase": "exact_online_year_benchmark",
            "range": [START.isoformat(), END.isoformat()],
            "execution": (
                "online page contract; next open; T+1 sell lock disabled; "
                "same fees and slippage"
            ),
            "online_override": online_override,
            "point_in_time_context": pit_context,
            "results": {name: row["stats"] for name, row in results.items()},
            "portfolios": portfolios,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        for prepared in prepared_objects:
            prepared.compute_cache.close()
        online_prepared.compute_cache.close()
        loader.compute_cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/exact_online_year_benchmark.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
