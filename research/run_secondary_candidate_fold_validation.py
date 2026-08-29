"""Frozen 26-fold validation for the diversified secondary-ignition candidate."""
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

START = date(2014, 1, 1)
END = date(2026, 8, 27)
LOAD_START = date(2013, 1, 1)
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
CANDIDATE = {
    **winner_study.CANDIDATES["secondary_ignition"],
    "execution": {
        "max_positions": 15,
        "max_hold_days": 30,
        "stop_loss": -0.06,
    },
}


def _run_period(service, market, start: date, end: date) -> dict:
    config = study._config(CANDIDATE, start, end)
    config.enforce_t_plus_one = False
    prepared = common._prepared(service, [config], market)
    try:
        return study._run(service, config, prepared)
    finally:
        prepared.compute_cache.close()


def run(
    data_dir: Path,
    research_dir: Path,
    benchmark_input: Path,
    output: Path,
) -> None:
    benchmark_payload = json.loads(benchmark_input.read_text(encoding="utf-8"))
    benchmark_folds = {
        row["label"]: row["individual"]["online_n_day_low_reversal"]
        for row in benchmark_payload["folds"]
    }
    _, service = common._engine(data_dir, research_dir)
    loader, market, pit_context = winner_study._base_market(
        service, data_dir, LOAD_START
    )
    try:
        rows = []
        for label, start, end in FOLDS:
            candidate = _run_period(service, market, start, end)
            rows.append(
                {
                    "label": label,
                    "range": [start.isoformat(), end.isoformat()],
                    "candidate": candidate,
                    "online_n_day_low_reversal": benchmark_folds[label],
                }
            )
    finally:
        loader.compute_cache.close()

    candidate_returns = [float(row["candidate"]["total_return"]) for row in rows]
    benchmark_returns = [
        float(row["online_n_day_low_reversal"]["total_return"]) for row in rows
    ]
    positive = sum(value > 0 for value in candidate_returns)
    benchmark_positive = sum(value > 0 for value in benchmark_returns)
    beats = sum(
        left > right
        for left, right in zip(candidate_returns, benchmark_returns, strict=True)
    )
    median_excess = statistics.median(
        left - right
        for left, right in zip(candidate_returns, benchmark_returns, strict=True)
    )
    summary = {
        "folds": len(rows),
        "positive_folds": positive,
        "benchmark_positive_folds": benchmark_positive,
        "beats_online_new_low_folds": beats,
        "median_return": round(statistics.median(candidate_returns), 6),
        "median_excess_vs_online_new_low": round(median_excess, 6),
        "passes_frozen_gate": (
            positive >= max(benchmark_positive, math.ceil(len(rows) * 0.50))
            and beats >= math.ceil(len(rows) * 0.55)
            and median_excess > 0
        ),
    }
    payload = {
        "phase": "secondary_candidate_frozen_fold_validation",
        "candidate": CANDIDATE,
        "execution": "online page contract with same-day T+1 sell lock disabled",
        "point_in_time_context": pit_context,
        "summary": summary,
        "folds": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        default=Path("/app/data/research/exact_online_stability_validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/secondary_candidate_fold_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.benchmark_input, args.output)


if __name__ == "__main__":
    main()
