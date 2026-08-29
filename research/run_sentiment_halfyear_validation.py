"""Half-year stability validation for the frozen sentiment anti-chase repair."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_reversal_study as common
import run_sentiment_fix_study as fixes
import run_sentiment_long_validation as long_run

FOLDS = tuple(
    (
        f"{year}{half}",
        date(year, 1 if half == "H1" else 7, 1),
        date(year, 6, 30) if half == "H1" else min(date(year, 12, 31), long_run.END),
    )
    for year in range(2014, 2027)
    for half in ("H1", "H2")
    if date(year, 1 if half == "H1" else 7, 1) <= long_run.END
)
PARAMS = fixes.CANDIDATES["sentiment_anti_chase"]


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                long_run.LOAD_START,
                long_run.END,
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
    rows = []
    try:
        for label, start, end in FOLDS:
            spec = fixes._spec(PARAMS)
            config = study._config(spec, start, end)
            prepared = common._prepared(service, [config], market)
            try:
                rows.append(
                    {
                        "label": label,
                        "range": [start.isoformat(), end.isoformat()],
                        "stats": study._run(service, config, prepared),
                    }
                )
            finally:
                prepared.compute_cache.close()
        returns = [float(row["stats"]["total_return"]) for row in rows]
        payload = {
            "phase": "sentiment_anti_chase_halfyear_validation",
            "range": [long_run.START.isoformat(), long_run.END.isoformat()],
            "point_in_time_context": pit_context,
            "params": PARAMS,
            "folds": rows,
            "positive_folds": sum(value > 0 for value in returns),
            "negative_folds": sum(value < 0 for value in returns),
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
        default=Path("/app/data/research/sentiment_halfyear_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
