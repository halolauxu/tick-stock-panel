"""Frozen tests of garbage-removal and alternative recent-winner pools."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common

LONG_LOAD_START = date(2013, 1, 1)
LONG_START = date(2014, 1, 1)
END = recent.END


def _independent(family: str, *, hold: int, stop: float, **params) -> dict:
    return {
        "strategy_id": "independent_alpha_families",
        "params": {"family": family, "eligibility_mode": "pit", **params},
        "execution": {
            "max_positions": 10,
            "max_hold_days": hold,
            "stop_loss": stop,
        },
    }


CANDIDATES = {
    "reversal_recovery_p15": study.CANDIDATES["reversal_recovery_p15"],
    "sentiment_anti_chase": _independent(
        "sentiment_timing",
        hold=60,
        stop=-0.10,
        sentiment_max_momentum_60d=0.50,
        sentiment_max_distance_ma20=0.15,
        sentiment_rsi_max=70.0,
    ),
    "sentiment_secondary_ignition": _independent(
        "sentiment_secondary_ignition", hold=20, stop=-0.08
    ),
    "secondary_ignition": _independent(
        "secondary_ignition", hold=20, stop=-0.08
    ),
    "secondary_strong_market": _independent(
        "secondary_ignition",
        hold=20,
        stop=-0.08,
        secondary_min_above_ma20=0.70,
    ),
    "accumulation_secondary_ignition": _independent(
        "accumulation_secondary_ignition", hold=20, stop=-0.08
    ),
    "accumulation_strong_market": _independent(
        "accumulation_strong_market", hold=20, stop=-0.08
    ),
    "breadth_oversold_repair": _independent(
        "breadth_oversold_repair", hold=15, stop=-0.06
    ),
}


def _base_market(service, data_dir: Path, load_start: date):
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                load_start,
                END,
                params={**study.BASELINE_PARAMS, "eligibility_mode": "none"},
                max_positions=10,
                max_hold_days=15,
                stop_loss=-0.06,
                basic_filter_override=study.PIT_FILTER,
            )
        ],
    )
    market = common._attach_industry_context(loader.market_data, data_dir)
    market, pit_context = common._attach_point_in_time_universe(market, data_dir)
    return loader, market, pit_context


def _run_period(service, market, start: date, end: date) -> dict:
    names = list(CANDIDATES)
    configs = [study._config(CANDIDATES[name], start, end) for name in names]
    baseline_config = study._baseline_config(start, end)
    baseline_config.enforce_t_plus_one = True
    for config in configs:
        config.enforce_t_plus_one = True
    prepared_by_name, prepared_objects = common._prepared_groups(
        service, names, configs, market
    )
    baseline_prepared = common._prepared(service, [baseline_config], market)
    try:
        results = {
            "point_in_time_new_low_reversal": study._run(
                service, baseline_config, baseline_prepared
            )
        }
        results.update(
            {
                name: study._run(service, config, prepared_by_name[name])
                for name, config in zip(names, configs, strict=True)
            }
        )
        return results
    finally:
        for prepared in prepared_objects:
            prepared.compute_cache.close()
        baseline_prepared.compute_cache.close()


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    recent_loader, recent_market, recent_context = _base_market(
        service, data_dir, recent.WARMUP_START
    )
    try:
        recent_full = _run_period(service, recent_market, recent.START, END)
        recent_periods = {
            label: {
                "range": [start.isoformat(), end.isoformat()],
                "results": _run_period(service, recent_market, start, end),
            }
            for label, start, end in recent.FOLDS
        }
    finally:
        recent_loader.compute_cache.close()

    long_loader, long_market, long_context = _base_market(
        service, data_dir, LONG_LOAD_START
    )
    try:
        long_full = _run_period(service, long_market, LONG_START, END)
    finally:
        long_loader.compute_cache.close()

    summaries = {}
    for name in recent_full:
        period_returns = [
            float(row["results"][name]["total_return"])
            for row in recent_periods.values()
        ]
        summaries[name] = {
            "recent_year": recent_full[name],
            "recent_period_returns": period_returns,
            "recent_positive_periods": sum(value > 0 for value in period_returns),
            "long_horizon": long_full[name],
        }
    payload = {
        "phase": "winner_pool_research",
        "execution": "open_t_plus_one with A-share T+1 enforced",
        "recent_range": [recent.START.isoformat(), END.isoformat()],
        "long_range": [LONG_START.isoformat(), END.isoformat()],
        "recent_point_in_time_context": recent_context,
        "long_point_in_time_context": long_context,
        "evidence": {
            "same_pool_winner_signature": {
                "momentum_60d": 0.214151,
                "ma20_bias": 0.086107,
                "limit_up_count_20d": 1.0,
                "ret_skew_20d": 0.392986,
            },
            "selected_loser_signature": {
                "momentum_60d": 1.047006,
                "ma20_bias": 0.257607,
                "limit_up_count_20d": 4.5,
                "ret_skew_20d": -0.295424,
            },
        },
        "candidate_specs": CANDIDATES,
        "recent_full": recent_full,
        "recent_periods": recent_periods,
        "long_full": long_full,
        "summaries": summaries,
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
        "--output",
        type=Path,
        default=Path("/app/data/research/winner_pool_study.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
