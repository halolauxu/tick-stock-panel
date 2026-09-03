"""Admit forecast events only when their first tradable open is risk-off."""

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

import run_p0_microcap_idiosyncratic_forecast_unified_account as unified  # noqa: E402
import run_p0_risk_gated_idiosyncratic_forecast_overlay as gated  # noqa: E402


def evaluate(
    results: dict[str, dict[str, Any]],
    core: dict[str, dict[str, Any]],
    daily_gated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validation = results["validation"]
    stress = results["known_stress"]
    validation_yearly = {
        row["year"]: row["return"] for row in validation["metrics"]["yearly"]
    }
    stress_yearly = {row["year"]: row["return"] for row in stress["metrics"]["yearly"]}
    metrics = stress["metrics"]
    checks = {
        "validation_annualized_at_least_40pct": (
            validation["metrics"].get("annualized") or -math.inf
        )
        >= 0.40,
        "validation_all_years_positive": all(
            (validation_yearly.get(year) or -math.inf) > 0
            for year in (2021, 2022, 2023)
        ),
        "validation_drawdown_within_25pct": (
            validation["metrics"].get("max_drawdown") or -math.inf
        )
        >= -0.25,
        "stress_annualized_at_least_23pct": (metrics.get("annualized") or -math.inf)
        >= 0.23,
        "stress_drawdown_within_35pct": (metrics.get("max_drawdown") or -math.inf)
        >= -0.35,
        "stress_drawdown_improves_core_by_10pp": (
            (metrics.get("max_drawdown") or -math.inf)
            - (core["known_stress"].get("max_drawdown") or -math.inf)
            >= 0.10
        ),
        "stress_all_years_positive": all(
            (stress_yearly.get(year) or -math.inf) > 0 for year in (2024, 2025, 2026)
        ),
        "stress_not_worse_than_daily_gate": (
            (metrics.get("annualized") or -math.inf)
            >= (daily_gated["known_stress"].get("annualized") or -math.inf)
        ),
        "stress_at_least_20_event_round_trips": stress["event_round_trips"] >= 20,
        "stress_buy_execution_at_least_80pct": stress["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.80,
        "stress_sell_execution_at_least_80pct": stress["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.80,
        "stress_no_unresolved_positions": stress["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "stress_cash_reconciled": stress["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
    }
    return {
        "verdict": "FORWARD_ELIGIBLE" if all(checks.values()) else "TERMINATE",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "stress_annualized_vs_core": metrics.get("annualized")
        - core["known_stress"].get("annualized"),
        "stress_annualized_vs_daily_gate": metrics.get("annualized")
        - daily_gated["known_stress"].get("annualized"),
        "stress_drawdown_vs_core": metrics.get("max_drawdown")
        - core["known_stress"].get("max_drawdown"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(
    data_dir: Path,
    thresholds_path: Path,
    core_result: Path,
    daily_gated_result: Path,
    output: Path,
) -> dict[str, Any]:
    core_payload = json.loads(core_result.read_text(encoding="utf-8"))
    daily_gated_payload = json.loads(daily_gated_result.read_text(encoding="utf-8"))
    core = {
        period: unified._core_metrics(core_payload, period) for period in gated.PERIODS
    }
    daily_gated = {
        period: {
            "annualized": row["metrics"]["annualized"],
            "max_drawdown": row["metrics"]["max_drawdown"],
        }
        for period, row in daily_gated_payload["results"].items()
    }
    results = {}
    risk_audit = {}
    for period, (start, end) in gated.PERIODS.items():
        gate, audit = gated.build_event_gate(data_dir, start, end, thresholds_path)
        results[period] = unified.run_period(
            data_dir, start, end, event_admission_by_date=gate
        )
        risk_audit[period] = audit
    decision = evaluate(results, core, daily_gated)
    payload = {
        "schema_version": "p0-risk-admitted-idiosyncratic-forecast-overlay-v1",
        "contract_frozen": "2026-09-03",
        "core": core,
        "daily_gated": daily_gated,
        "risk": risk_audit,
        "results": results,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "results": {
                    name: {
                        key: value
                        for key, value in row.items()
                        if key not in {"daily_equity", "orders"}
                    }
                    for name, row in results.items()
                },
                "risk": {
                    name: {
                        key: value
                        for key, value in row.items()
                        if key not in {"decisions", "switches"}
                    }
                    for name, row in risk_audit.items()
                },
                "decision": decision,
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
        "--thresholds",
        type=Path,
        default=RESEARCH / "p0_microcap_escape_thresholds.json",
    )
    parser.add_argument(
        "--core-result",
        type=Path,
        default=Path("/app/data/research/p0_main_board_microcap_account_v1.json"),
    )
    parser.add_argument(
        "--daily-gated-result",
        type=Path,
        default=Path(
            "/app/data/research/p0_risk_gated_idiosyncratic_forecast_overlay_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_risk_admitted_idiosyncratic_forecast_overlay_v1.json"
        ),
    )
    args = parser.parse_args()
    run(
        args.data_dir,
        args.thresholds,
        args.core_result,
        args.daily_gated_result,
        args.output,
    )


if __name__ == "__main__":
    main()
