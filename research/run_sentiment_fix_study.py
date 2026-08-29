"""Causal repair study for the recent sentiment-strategy drawdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common

BASE_PARAMS = {"family": "sentiment_timing", "eligibility_mode": "pit"}
ANTI_CHASE = {
    "sentiment_max_momentum_60d": 0.50,
    "sentiment_max_distance_ma20": 0.15,
    "sentiment_rsi_max": 70.0,
}
CANDIDATES = {
    "sentiment_control": BASE_PARAMS,
    "sentiment_anti_chase": {**BASE_PARAMS, **ANTI_CHASE},
    "sentiment_anti_chase_confirm2": {
        **BASE_PARAMS,
        **ANTI_CHASE,
        "sentiment_confirm_days": 2,
    },
    "sentiment_anti_chase_stock_exit": {
        **BASE_PARAMS,
        **ANTI_CHASE,
        "sentiment_stock_ma20_exit": True,
    },
    "sentiment_anti_chase_confirm2_stock_exit": {
        **BASE_PARAMS,
        **ANTI_CHASE,
        "sentiment_confirm_days": 2,
        "sentiment_stock_ma20_exit": True,
    },
}


def _spec(params: dict) -> dict:
    return {
        "strategy_id": "independent_alpha_families",
        "params": params,
        "execution": {
            "max_positions": 10,
            "max_hold_days": 60,
            "stop_loss": -0.10,
        },
    }


def _run_period(service, market, start, end, *, enforce_t_plus_one: bool = False) -> dict:
    names = list(CANDIDATES)
    configs = [study._config(_spec(CANDIDATES[name]), start, end) for name in names]
    for config in configs:
        config.enforce_t_plus_one = enforce_t_plus_one
    prepared_by_name, prepared_objects = common._prepared_groups(
        service, names, configs, market
    )
    try:
        return {
            name: study._run(service, config, prepared_by_name[name])
            for name, config in zip(names, configs, strict=True)
        }
    finally:
        for prepared in prepared_objects:
            prepared.compute_cache.close()


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                recent.WARMUP_START,
                recent.END,
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
    try:
        full_results = _run_period(service, market, recent.START, recent.END)
        period_results = {
            label: {
                "range": [start.isoformat(), end.isoformat()],
                "results": _run_period(service, market, start, end),
            }
            for label, start, end in recent.FOLDS
        }
        p4_t_plus_one_results = _run_period(
            service,
            market,
            recent.FOLDS[-1][1],
            recent.FOLDS[-1][2],
            enforce_t_plus_one=True,
        )
        summaries = {}
        for name, full in full_results.items():
            period_returns = [
                float(row["results"][name]["total_return"])
                for row in period_results.values()
            ]
            summaries[name] = {
                "full": full,
                "period_returns": period_returns,
                "positive_periods": sum(value > 0 for value in period_returns),
                "repairs_last_period": period_returns[-1] > 0,
                "recent_year_candidate": (
                    float(full["total_return"]) >= 0.20
                    and float(full["sharpe"]) >= 1.0
                    and float(full["max_drawdown"]) >= -0.15
                    and sum(value > 0 for value in period_returns) >= 3
                    and period_returns[-1] > 0
                    and int(full["n_trades"]) >= 50
                ),
            }
        qualified = [
            name for name, summary in summaries.items()
            if summary["recent_year_candidate"]
        ]
        qualified.sort(
            key=lambda name: (
                float(summaries[name]["full"]["total_return"]),
                float(summaries[name]["full"]["sharpe"]),
            ),
            reverse=True,
        )
        payload = {
            "phase": "sentiment_recent_drawdown_repair",
            "range": [recent.START.isoformat(), recent.END.isoformat()],
            "point_in_time_context": pit_context,
            "frozen_diagnosis": {
                "losing_batch_signal_date": "2026-08-20",
                "losing_batch_pnl_amount": -7109.01,
                "median_momentum_60d": 1.047006,
                "median_distance_above_ma20": 0.257607,
                "median_rsi14": 69.231167,
            },
            "candidate_params": CANDIDATES,
            "full_results": full_results,
            "period_results": period_results,
            "p4_t_plus_one_results": p4_t_plus_one_results,
            "summaries": summaries,
            "qualified": qualified,
            "winner": qualified[0] if qualified else None,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
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
        default=Path("/app/data/research/sentiment_fix_study.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
