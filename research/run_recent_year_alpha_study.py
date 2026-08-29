"""Frozen one-year replay of every strategy evaluated in the alpha study."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_reversal_study as common

START = date(2025, 8, 27)
END = date(2026, 8, 26)
WARMUP_START = date(2024, 8, 27)
FOLDS = (
    ("P1", date(2025, 8, 27), date(2025, 11, 26)),
    ("P2", date(2025, 11, 27), date(2026, 2, 26)),
    ("P3", date(2026, 2, 27), date(2026, 5, 26)),
    ("P4", date(2026, 5, 27), date(2026, 8, 26)),
)
VALIDATION_NAMES = (
    "reversal_recovery_p15",
    "sentiment_risk_on",
    "industry_rotation_reclaim_risk_gate",
    "monthly_momentum_12_1",
)


def _independent(
    family: str,
    *,
    max_positions: int,
    max_hold_days: int,
    stop_loss: float,
    **params,
) -> dict:
    return {
        "strategy_id": "independent_alpha_families",
        "params": {"family": family, "eligibility_mode": "pit", **params},
        "execution": {
            "max_positions": max_positions,
            "max_hold_days": max_hold_days,
            "stop_loss": stop_loss,
        },
    }


SPECS = {
    "reversal_recovery_p15": study.CANDIDATES["reversal_recovery_p15"],
    "trend_breakout_60d": _independent(
        "trend", max_positions=10, max_hold_days=60, stop_loss=-0.10
    ),
    "trend_breakout_60d_risk_gate": _independent(
        "trend",
        max_positions=10,
        max_hold_days=60,
        stop_loss=-0.10,
        risk_on_only=True,
    ),
    "industry_rotation_reclaim": _independent(
        "industry_rotation", max_positions=10, max_hold_days=40, stop_loss=-0.08
    ),
    "industry_rotation_reclaim_risk_gate": _independent(
        "industry_rotation",
        max_positions=10,
        max_hold_days=40,
        stop_loss=-0.08,
        risk_on_only=True,
    ),
    "sentiment_risk_on": _independent(
        "sentiment_timing", max_positions=10, max_hold_days=60, stop_loss=-0.10
    ),
    "quality_compounder_reclaim": _independent(
        "quality_compounder",
        max_positions=10,
        max_hold_days=60,
        stop_loss=-0.10,
    ),
    "quality_compounder_reclaim_risk_gate": _independent(
        "quality_compounder",
        max_positions=10,
        max_hold_days=60,
        stop_loss=-0.10,
        risk_on_only=True,
    ),
    "limit_event_first_reclaim": _independent(
        "limit_event", max_positions=10, max_hold_days=20, stop_loss=-0.08
    ),
    "monthly_momentum_12_1": _independent(
        "monthly_momentum",
        max_positions=10,
        max_hold_days=21,
        stop_loss=-0.10,
        quality_gate=False,
    ),
    "monthly_quality_momentum_12_1": _independent(
        "monthly_momentum",
        max_positions=10,
        max_hold_days=21,
        stop_loss=-0.10,
        quality_gate=True,
    ),
    "regime_reversal_quality": study.CANDIDATES["regime_reversal_quality"],
}


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                WARMUP_START,
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
    try:
        names = list(SPECS)
        configs = [study._config(SPECS[name], START, END) for name in names]
        prepared_by_name, prepared_objects = common._prepared_groups(
            service, names, configs, market
        )
        baseline_config = study._baseline_config(START, END)
        baseline_prepared = common._prepared(service, [baseline_config], market)
        try:
            results = {
                "point_in_time_new_low_reversal": study._run(
                    service, baseline_config, baseline_prepared
                )
            }
            for name, config in zip(names, configs, strict=True):
                results[name] = study._run(
                    service, config, prepared_by_name[name]
                )
        finally:
            for prepared in prepared_objects:
                prepared.compute_cache.close()
            baseline_prepared.compute_cache.close()
        ranking = sorted(
            results,
            key=lambda name: (
                float(results[name]["total_return"]),
                float(results[name]["sharpe"]),
            ),
            reverse=True,
        )
        fold_results = {}
        for label, fold_start, fold_end in FOLDS:
            fold_names = list(VALIDATION_NAMES)
            fold_configs = [
                study._config(SPECS[name], fold_start, fold_end)
                for name in fold_names
            ]
            fold_prepared, fold_objects = common._prepared_groups(
                service, fold_names, fold_configs, market
            )
            fold_baseline_config = study._baseline_config(fold_start, fold_end)
            fold_baseline_prepared = common._prepared(
                service, [fold_baseline_config], market
            )
            try:
                fold_results[label] = {
                    "range": [fold_start.isoformat(), fold_end.isoformat()],
                    "point_in_time_new_low_reversal": study._run(
                        service, fold_baseline_config, fold_baseline_prepared
                    ),
                }
                for name, config in zip(fold_names, fold_configs, strict=True):
                    fold_results[label][name] = study._run(
                        service, config, fold_prepared[name]
                    )
            finally:
                for prepared in fold_objects:
                    prepared.compute_cache.close()
                fold_baseline_prepared.compute_cache.close()
        payload = {
            "phase": "recent_year_frozen_replay",
            "range": [START.isoformat(), END.isoformat()],
            "point_in_time_context": pit_context,
            "specs": SPECS,
            "results": results,
            "ranking": ranking,
            "winner": ranking[0],
            "four_period_validation": fold_results,
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
        default=Path("/app/data/research/recent_year_alpha_study.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
