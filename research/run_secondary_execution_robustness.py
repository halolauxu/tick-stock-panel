"""Robustness screen for secondary-ignition execution parameters.

The signal definition stays frozen.  Only concentration, holding horizon, and
stop-loss are varied over a small predeclared grid to test whether the mechanism
survives less concentrated portfolio construction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import run_exact_online_year_benchmark as exact
import run_independent_alpha_study as study
import run_reversal_study as common
import run_winner_pool_study as winner_study

LONG_START = date(2014, 1, 1)
RECENT_START = date(2025, 8, 27)
END = date(2026, 8, 27)
LOAD_START = date(2013, 1, 1)
SIGNAL_SPEC = winner_study.CANDIDATES["secondary_ignition"]
GRID = tuple(
    (positions, hold_days, stop_loss)
    for positions in (10, 15, 20)
    for hold_days in (15, 20, 30)
    for stop_loss in (-0.06, -0.08)
)


def _name(positions: int, hold_days: int, stop_loss: float) -> str:
    return f"p{positions}_h{hold_days}_s{abs(int(stop_loss * 100))}"


def _config(
    start: date,
    end: date,
    positions: int,
    hold_days: int,
    stop_loss: float,
):
    spec = {
        **SIGNAL_SPEC,
        "execution": {
            "max_positions": positions,
            "max_hold_days": hold_days,
            "stop_loss": stop_loss,
        },
    }
    config = study._config(spec, start, end)
    config.enforce_t_plus_one = False
    return config


def _run_grid(service, market, start: date, end: date) -> dict:
    configs = [_config(start, end, *values) for values in GRID]
    prepared = common._prepared(service, [configs[0]], market)
    try:
        results = {}
        for values, config in zip(GRID, configs, strict=True):
            compatible = replace(
                prepared, signature=service._matrix_prepare_signature(config)
            )
            result = service.run(
                config, prepared=compatible, result_policy=exact.CURVE_POLICY
            )
            if result.error:
                raise RuntimeError(f"{_name(*values)}: {result.error}")
            results[_name(*values)] = {
                key: result.stats.get(key) for key in common.POLICY.required_stats
            }
        return results
    finally:
        prepared.compute_cache.close()


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader, market, pit_context = winner_study._base_market(
        service, data_dir, LOAD_START
    )
    try:
        recent = _run_grid(service, market, RECENT_START, END)
        long = _run_grid(service, market, LONG_START, END)
    finally:
        loader.compute_cache.close()
    benchmark = {
        "recent_year": {
            "total_return": 0.0836,
            "sharpe": 0.72,
            "max_drawdown": -0.1119,
        },
        "long_full": {
            "total_return": 0.4061,
            "sharpe": 0.26,
            "max_drawdown": -0.6225,
        },
    }
    qualified = [
        name
        for name in recent
        if float(recent[name]["total_return"])
        > benchmark["recent_year"]["total_return"]
        and float(long[name]["total_return"])
        > benchmark["long_full"]["total_return"]
        and float(recent[name]["sharpe"]) > benchmark["recent_year"]["sharpe"]
        and float(long[name]["sharpe"]) > benchmark["long_full"]["sharpe"]
        and float(recent[name]["max_drawdown"])
        > benchmark["recent_year"]["max_drawdown"]
        and float(long[name]["max_drawdown"])
        > benchmark["long_full"]["max_drawdown"]
    ]
    payload = {
        "phase": "secondary_execution_robustness",
        "signal_spec": SIGNAL_SPEC,
        "grid": [
            {
                "name": _name(*values),
                "max_positions": values[0],
                "max_hold_days": values[1],
                "stop_loss": values[2],
            }
            for values in GRID
        ],
        "execution": "online page contract with same-day T+1 sell lock disabled",
        "point_in_time_context": pit_context,
        "benchmark": benchmark,
        "recent_year": recent,
        "long_full": long,
        "qualified": qualified,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "qualified": qualified,
                "recent_year": recent,
                "long_full": long,
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
        default=Path("/app/data/research/secondary_execution_robustness.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
