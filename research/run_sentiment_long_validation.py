"""Long-horizon validation of frozen sentiment drawdown repairs."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import run_independent_alpha_study as study
import run_reversal_study as common
import run_sentiment_fix_study as fixes

LOAD_START = date(2013, 1, 1)
START = date(2014, 1, 1)
END = date(2026, 8, 26)


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
                max_positions=10,
                max_hold_days=15,
                stop_loss=-0.06,
                basic_filter_override=study.PIT_FILTER,
            )
        ],
    )
    market = common._attach_industry_context(loader.market_data, data_dir)
    market, pit_context = common._attach_point_in_time_universe(market, data_dir)
    baseline_config = study._baseline_config(START, END)
    baseline_prepared = common._prepared(service, [baseline_config], market)
    try:
        results = fixes._run_period(service, market, START, END)
        results["point_in_time_new_low_reversal"] = study._run(
            service, baseline_config, baseline_prepared
        )
        payload = {
            "phase": "sentiment_repair_long_horizon",
            "range": [START.isoformat(), END.isoformat()],
            "point_in_time_context": pit_context,
            "candidate_params": fixes.CANDIDATES,
            "results": results,
            "ranking": sorted(
                results,
                key=lambda name: (
                    float(results[name]["total_return"]),
                    float(results[name]["sharpe"]),
                ),
                reverse=True,
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        baseline_prepared.compute_cache.close()
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
        default=Path("/app/data/research/sentiment_long_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
