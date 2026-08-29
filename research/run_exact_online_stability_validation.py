"""Validate a frozen research portfolio against the exact online strategy.

The benchmark mirrors the backtest page contract, including its compatibility
default that leaves the same-day T+1 sell lock disabled.  Research sleeves use
the point-in-time universe so delisted, renamed, and historical share-capital
states do not leak through today's security master.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import date
from pathlib import Path

import run_exact_online_year_benchmark as exact
import run_independent_alpha_study as study
import run_reversal_study as common
import run_winner_pool_study as winner_study
from app.strategy import config as strategy_config

START = date(2014, 1, 1)
END = date(2026, 8, 27)
LOAD_START = date(2013, 1, 1)
RECENT_START = date(2025, 8, 27)
SLEEVES = {
    "breadth_oversold_repair": winner_study.CANDIDATES[
        "breadth_oversold_repair"
    ],
    "secondary_ignition": winner_study.CANDIDATES["secondary_ignition"],
}
FROZEN_WEIGHTS = {
    "breadth_oversold_repair": 0.75,
    "secondary_ignition": 0.25,
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


def _online_config(data_dir: Path, start: date, end: date):
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
    config.enforce_t_plus_one = False
    return config


def _run_one(service, config, prepared) -> dict:
    result = service.run(config, prepared=prepared, result_policy=exact.CURVE_POLICY)
    if result.error:
        raise RuntimeError(f"{config.strategy_id}: {result.error}")
    return {
        "stats": {
            key: result.stats.get(key) for key in common.POLICY.required_stats
        },
        "equity_curve": result.equity_curve,
    }


def _run_period(
    service, data_dir: Path, online_market, research_market, start: date, end: date
):
    online_config = _online_config(data_dir, start, end)
    names = list(SLEEVES)
    configs = [study._config(SLEEVES[name], start, end) for name in names]
    for config in configs:
        config.enforce_t_plus_one = False
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
        return {
            "individual": {
                name: row["stats"] for name, row in results.items()
            },
            "breadth_secondary_75_25": exact._portfolio(
                results, FROZEN_WEIGHTS
            ),
        }
    finally:
        online_prepared.compute_cache.close()
        for prepared in prepared_objects:
            prepared.compute_cache.close()


def _fold_summary(rows: list[dict]) -> dict:
    candidate = [
        float(row["breadth_secondary_75_25"]["total_return"]) for row in rows
    ]
    benchmark = [
        float(
            row["individual"]["online_n_day_low_reversal"]["total_return"]
        )
        for row in rows
    ]
    positive = sum(value > 0 for value in candidate)
    beats = sum(
        left > right for left, right in zip(candidate, benchmark, strict=True)
    )
    median_excess = statistics.median(
        left - right
        for left, right in zip(candidate, benchmark, strict=True)
    )
    return {
        "folds": len(rows),
        "positive_folds": positive,
        "beats_online_new_low_folds": beats,
        "median_return": round(statistics.median(candidate), 6),
        "median_excess_vs_online_new_low": round(median_excess, 6),
        "passes_frozen_gate": (
            positive >= math.ceil(len(rows) * 0.50)
            and beats >= math.ceil(len(rows) * 0.55)
            and median_excess > 0
        ),
    }


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
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
    try:
        recent_year = _run_period(
            service, data_dir, online_market, research_market, RECENT_START, END
        )
        long_full = _run_period(
            service, data_dir, online_market, research_market, START, END
        )
        fold_rows = []
        for label, start, end in FOLDS:
            row = _run_period(
                service, data_dir, online_market, research_market, start, end
            )
            fold_rows.append(
                {"label": label, "range": [start.isoformat(), end.isoformat()], **row}
            )
    finally:
        loader.compute_cache.close()

    fold_summary = _fold_summary(fold_rows)
    recent_portfolio = recent_year["breadth_secondary_75_25"]
    recent_benchmark = recent_year["individual"]["online_n_day_low_reversal"]
    long_portfolio = long_full["breadth_secondary_75_25"]
    long_benchmark = long_full["individual"]["online_n_day_low_reversal"]
    passes = (
        fold_summary["passes_frozen_gate"]
        and float(recent_portfolio["total_return"])
        > float(recent_benchmark["total_return"])
        and float(long_portfolio["total_return"])
        > float(long_benchmark["total_return"])
        and float(long_portfolio["max_drawdown"])
        > float(long_benchmark["max_drawdown"])
    )
    payload = {
        "phase": "exact_online_stability_validation",
        "execution": (
            "online page contract; next open; T+1 sell lock disabled; "
            "same fees and slippage"
        ),
        "portfolio": {
            "name": "广度修复75_二次点火25",
            "weights": FROZEN_WEIGHTS,
            "rebalance": "fixed initial sleeves within each tested window",
        },
        "point_in_time_context": pit_context,
        "recent_year": recent_year,
        "long_full": long_full,
        "fold_summary": fold_summary,
        "folds": fold_rows,
        "passes_all_gates": passes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "recent_year": recent_year,
                "long_full": long_full,
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
        default=Path("/app/data/research/exact_online_stability_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
