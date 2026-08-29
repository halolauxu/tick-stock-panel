"""Focused T+1 validation of the abnormal-turnover secondary-ignition hypothesis."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common
import run_winner_pool_study as winner_study

CANDIDATES = {
    "reversal_recovery_p15": winner_study.CANDIDATES["reversal_recovery_p15"],
    "secondary_ignition": winner_study.CANDIDATES["secondary_ignition"],
    "secondary_strong_market": winner_study.CANDIDATES[
        "secondary_strong_market"
    ],
    "accumulation_secondary_ignition": winner_study.CANDIDATES[
        "accumulation_secondary_ignition"
    ],
    "accumulation_strong_market": winner_study.CANDIDATES[
        "accumulation_strong_market"
    ],
}


def _run_period(service, market, start, end) -> dict:
    names = list(CANDIDATES)
    configs = [study._config(CANDIDATES[name], start, end) for name in names]
    for config in configs:
        config.enforce_t_plus_one = True
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
    recent_loader, recent_market, recent_context = winner_study._base_market(
        service, data_dir, recent.WARMUP_START
    )
    try:
        recent_full = _run_period(service, recent_market, recent.START, recent.END)
        recent_periods = {
            label: {
                "range": [start.isoformat(), end.isoformat()],
                "results": _run_period(service, recent_market, start, end),
            }
            for label, start, end in recent.FOLDS
        }
    finally:
        recent_loader.compute_cache.close()

    long_loader, long_market, long_context = winner_study._base_market(
        service, data_dir, winner_study.LONG_LOAD_START
    )
    try:
        backward_oos = _run_period(
            service, long_market, winner_study.LONG_START, date(2025, 8, 26)
        )
        long_full = _run_period(
            service, long_market, winner_study.LONG_START, recent.END
        )
    finally:
        long_loader.compute_cache.close()

    payload = {
        "phase": "accumulation_secondary_ignition",
        "execution": "open_t_plus_one with A-share T+1 enforced",
        "hypothesis": (
            "prior abnormal turnover, cooled current turnover, price near VWAP, "
            "positive price-volume relation, improving industry breadth, MA10 reclaim"
        ),
        "recent_point_in_time_context": recent_context,
        "long_point_in_time_context": long_context,
        "candidate_specs": CANDIDATES,
        "recent_full": recent_full,
        "recent_periods": recent_periods,
        "backward_oos_before_recent_year": backward_oos,
        "long_full": long_full,
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
        default=Path("/app/data/research/accumulation_candidate_study.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
