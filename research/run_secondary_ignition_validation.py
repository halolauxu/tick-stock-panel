"""Frozen half-year and backward-OOS validation for secondary ignition."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_reversal_study as common
import run_winner_pool_study as winner_study

FOLDS = tuple(
    (
        f"{year}{half}",
        date(year, 1 if half == "H1" else 7, 1),
        (
            date(year, 6, 30)
            if half == "H1"
            else min(date(year, 12, 31), winner_study.END)
        ),
    )
    for year in range(2014, 2027)
    for half in ("H1", "H2")
    if date(year, 1 if half == "H1" else 7, 1) <= winner_study.END
)
CANDIDATES = {
    "reversal_recovery_p15": winner_study.CANDIDATES["reversal_recovery_p15"],
    "secondary_ignition": winner_study.CANDIDATES["secondary_ignition"],
    "accumulation_secondary_ignition": winner_study.CANDIDATES[
        "accumulation_secondary_ignition"
    ],
    "breadth_oversold_repair": winner_study.CANDIDATES["breadth_oversold_repair"],
}


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
        output = {
            "point_in_time_new_low_reversal": study._run(
                service, baseline_config, baseline_prepared
            )
        }
        output.update(
            {
                name: study._run(service, config, prepared_by_name[name])
                for name, config in zip(names, configs, strict=True)
            }
        )
        return output
    finally:
        for prepared in prepared_objects:
            prepared.compute_cache.close()
        baseline_prepared.compute_cache.close()


def _summary(name: str, rows: list[dict]) -> dict:
    candidate = [float(row["results"][name]["total_return"]) for row in rows]
    benchmark = [
        float(row["results"]["reversal_recovery_p15"]["total_return"])
        for row in rows
    ]
    return {
        "folds": len(rows),
        "positive_folds": sum(value > 0 for value in candidate),
        "beats_reversal_p15_folds": sum(
            left > right for left, right in zip(candidate, benchmark, strict=True)
        ),
        "median_return": round(statistics.median(candidate), 6),
        "median_excess_vs_reversal_p15": round(
            statistics.median(
                left - right
                for left, right in zip(candidate, benchmark, strict=True)
            ),
            6,
        ),
        "passes_fold_gate": (
            sum(value > 0 for value in candidate) >= math.ceil(len(rows) * 0.50)
            and sum(
                left > right
                for left, right in zip(candidate, benchmark, strict=True)
            )
            >= math.ceil(len(rows) * 0.55)
        ),
    }


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader, market, pit_context = winner_study._base_market(
        service, data_dir, winner_study.LONG_LOAD_START
    )
    try:
        fold_rows = [
            {
                "label": label,
                "range": [start.isoformat(), end.isoformat()],
                "results": _run_period(service, market, start, end),
            }
            for label, start, end in FOLDS
        ]
        backward_oos = _run_period(
            service, market, date(2014, 1, 1), date(2025, 12, 31)
        )
        design_year = _run_period(
            service, market, date(2026, 1, 1), winner_study.END
        )
        summaries = {
            name: _summary(name, fold_rows)
            for name in (
                "secondary_ignition",
                "accumulation_secondary_ignition",
                "breadth_oversold_repair",
                "reversal_recovery_p15",
            )
        }
        payload = {
            "phase": "secondary_ignition_frozen_validation",
            "execution": "open_t_plus_one with A-share T+1 enforced",
            "point_in_time_context": pit_context,
            "candidate_specs": CANDIDATES,
            "backward_oos_2014_2025": backward_oos,
            "design_year_2026": design_year,
            "folds": fold_rows,
            "summaries": summaries,
            "winner": (
                next(
                    (
                        name
                        for name in (
                            "secondary_ignition",
                            "accumulation_secondary_ignition",
                            "breadth_oversold_repair",
                        )
                        if summaries[name]["passes_fold_gate"]
                        and float(backward_oos[name]["total_return"])
                        > float(
                            backward_oos["reversal_recovery_p15"]["total_return"]
                        )
                    ),
                    None,
                )
            ),
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
        default=Path("/app/data/research/secondary_ignition_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
