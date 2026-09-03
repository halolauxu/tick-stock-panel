"""Validate the frozen main-board positive-forecast account on 2021-2023."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_main_board_forecast_drift_account as study  # noqa: E402

VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = date(2023, 12, 31)


def _configure_validation_period() -> None:
    study.CAPITALS = (study.PRIMARY_CAPITAL,)
    study.DEVELOPMENT_START = VALIDATION_START
    study.DEVELOPMENT_END = VALIDATION_END
    study.forecast.DEVELOPMENT_START = VALIDATION_START
    study.forecast.DEVELOPMENT_END = VALIDATION_END


def evaluate_validation(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload["capital_tiers"][str(int(study.PRIMARY_CAPITAL))]
    metrics = primary["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    benchmark_annualized = float(payload["benchmark"].get("annualized") or -math.inf)
    round_trips = int(primary["account"].get("trade_count") or 0) // 2
    combination_checks = {
        "annualized_positive": annualized > 0,
        "all_three_years_positive": int(metrics.get("positive_years") or 0) == 3,
        "at_least_150_round_trips": round_trips >= 150,
        "buy_execution_at_least_90pct": primary["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_execution_at_least_90pct": primary["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "no_unresolved_positions": primary["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": primary["integrity"]["max_cash_reconciliation_error"] <= 0.01,
    }
    combination_input = all(combination_checks.values())
    deploy_checks = {
        **combination_checks,
        "annualized_at_least_15pct": annualized >= 0.15,
        "annualized_excess_at_least_5pp": annualized - benchmark_annualized >= 0.05,
        "max_drawdown_within_30pct": float(metrics.get("max_drawdown") or -math.inf)
        >= -0.30,
        "mean_cash_ratio_at_most_70pct": float(metrics.get("mean_cash_ratio") or math.inf)
        <= 0.70,
    }
    deployable = all(deploy_checks.values())
    return {
        "verdict": (
            "DEPLOYABLE_STANDALONE"
            if deployable
            else "RETAIN_AS_COMBINATION_INPUT"
            if combination_input
            else "TERMINATE_MAIN_BOARD_FORECAST"
        ),
        "combination_input_passed": combination_input,
        "deployable_standalone": deployable,
        "annualized_excess": annualized - benchmark_annualized,
        "complete_round_trips": round_trips,
        "combination_checks": combination_checks,
        "deploy_checks": deploy_checks,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    _configure_validation_period()
    intermediate = output.with_suffix(".intermediate.json")
    payload = study.run(data_dir, intermediate)
    payload["schema_version"] = "p0-main-board-forecast-validation-v1"
    payload["contract_frozen"] = "2026-09-03"
    payload["period"] = {
        "start": VALIDATION_START,
        "end": VALIDATION_END,
        "development_reused": False,
        "known_stress_read": False,
    }
    payload["decision"] = evaluate_validation(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    intermediate.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "period": payload["period"],
                "data": payload["data"],
                "benchmark": payload["benchmark"],
                "primary": payload["capital_tiers"][str(int(study.PRIMARY_CAPITAL))],
                "decision": payload["decision"],
                "output": str(output),
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_main_board_forecast_validation_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
