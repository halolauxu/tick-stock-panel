"""Test a causal participation allocator for the micro-cap portfolio sleeve.

This is a known-history redesign prompted by the 2026 slow bleed.  It is not an
independent validation result: only a new forward account can validate it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_microcap_escape as escape  # noqa: E402
import run_p0_microcap_idiosyncratic_forecast_unified_account as unified  # noqa: E402
import run_p0_risk_gated_idiosyncratic_forecast_overlay as gated  # noqa: E402

from app.services import microcap_participation as participation  # noqa: E402

SCHEMA_VERSION = "p1-microcap-participation-allocator-v1"
VALIDATION = (date(2021, 1, 1), date(2023, 12, 31))
STRESS_START = date(2024, 1, 1)
RECENT_START = date(2026, 3, 1)
UPGRADE_CONFIRMATION_DAYS = participation.UPGRADE_CONFIRMATION_DAYS
FULL_SLOTS = participation.FULL_SLOTS
NO_SLOTS = participation.NO_SLOTS
SLOTS_PER_CONFIRMATION = participation.SLOTS_PER_CONFIRMATION


def attach_participation_features(features: pl.DataFrame) -> pl.DataFrame:
    return participation.attach_participation_features(features)


def raw_slot_budget(feature: dict[str, Any]) -> int:
    return participation.raw_slot_budget(feature)


def build_allocation_clock(
    features: pl.DataFrame,
) -> tuple[dict[date, int], list[dict[str, Any]]]:
    return participation.build_allocation_clock(features)


def build_allocator(
    data_dir: Path,
    *,
    end: date,
) -> tuple[dict[date, int], list[dict[str, Any]]]:
    source = baseline.load_daily(data_dir, end=end)
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    features = attach_participation_features(escape.build_daily_features(panel))
    del panel
    gc.collect()
    return build_allocation_clock(features)


def _slice_result(result: dict[str, Any], start: date, end: date) -> dict[str, Any]:
    rows = [
        row
        for row in result["daily_equity"]
        if start <= row["date"] <= end
    ]
    if not rows:
        raise ValueError("requested result slice has no daily equity")
    previous = [row for row in result["daily_equity"] if row["date"] < start]
    starting_equity = float(previous[-1]["equity"] if previous else rows[0]["equity"])
    returns = []
    prior = starting_equity
    for row in rows:
        equity = float(row["equity"])
        returns.append(equity / prior - 1.0)
        prior = equity
    orders = [
        row for row in result["orders"] if start <= row["date"] <= end
    ]
    return {
        "start": start,
        "end": end,
        "trading_days": len(rows),
        "total_return": rows[-1]["equity"] / starting_equity - 1.0,
        "max_drawdown": baseline._max_drawdown(returns),
        "mean_cash_ratio": sum(float(row["cash_ratio"]) for row in rows) / len(rows),
        "mean_position_count": sum(int(row["position_count"]) for row in rows) / len(rows),
        "filled_buys": sum(
            row["side"] == "BUY" and row["status"] == "FILLED" for row in orders
        ),
        "filled_sells": sum(
            row["side"] == "SELL" and row["status"] == "FILLED" for row in orders
        ),
    }


def _allocation_summary(
    decisions: list[dict[str, Any]], start: date, end: date
) -> dict[str, Any]:
    scoped = [row for row in decisions if start <= row["action_date"] <= end]
    level_days = {
        str(slots): sum(row["microcap_slots"] == slots for row in scoped)
        for slots in range(NO_SLOTS, FULL_SLOTS + 1, SLOTS_PER_CONFIRMATION)
    }
    return {
        "trading_days": len(scoped),
        "days_by_microcap_slots": level_days,
        "mean_microcap_slots": (
            sum(int(row["microcap_slots"]) for row in scoped) / len(scoped)
            if scoped
            else 0.0
        ),
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
    *,
    end: date,
    output: Path,
) -> dict[str, Any]:
    periods = {
        "validation_context": VALIDATION,
        "known_stress": (STRESS_START, end),
    }
    results: dict[str, dict[str, Any]] = {}
    allocation_audit: dict[str, Any] = {}
    risk_audit: dict[str, Any] = {}
    for name, (start, finish) in periods.items():
        event_admission, event_risk = gated.build_event_gate(
            data_dir, start, finish, thresholds_path
        )
        slots_by_open, decisions = build_allocator(data_dir, end=finish)
        results[name] = unified.run_period(
            data_dir,
            start,
            finish,
            event_admission_by_date=event_admission,
            microcap_slots_by_date=slots_by_open,
        )
        allocation_audit[name] = {
            **_allocation_summary(decisions, start, finish),
            "decisions": [
                row for row in decisions if start <= row["action_date"] <= finish
            ],
        }
        risk_audit[name] = event_risk
    recent = _slice_result(results["known_stress"], RECENT_START, end)
    stress_yearly = {
        row["year"]: row["return"]
        for row in results["known_stress"]["metrics"]["yearly"]
    }
    checks = {
        "validation_context_all_years_positive": all(
            row["return"] > 0
            for row in results["validation_context"]["metrics"]["yearly"]
        ),
        "stress_all_years_positive": all(
            (stress_yearly.get(year) or -math.inf) > 0
            for year in range(STRESS_START.year, end.year + 1)
        ),
        "stress_drawdown_within_30pct": (
            results["known_stress"]["metrics"]["max_drawdown"] >= -0.30
        ),
        "recent_return_nonnegative": recent["total_return"] >= 0,
        "recent_drawdown_within_15pct": recent["max_drawdown"] >= -0.15,
        "stress_buy_execution_at_least_80pct": (
            results["known_stress"]["execution"]["buy"]["execution_rate"] >= 0.80
        ),
        "stress_sell_execution_at_least_80pct": (
            results["known_stress"]["execution"]["sell"]["execution_rate"] >= 0.80
        ),
        "stress_cash_reconciled": (
            results["known_stress"]["integrity"]["max_cash_reconciliation_error"]
            <= 0.01
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_class": "known_history_redesign_requires_forward_validation",
        "contract": {
            "signal_time": "daily_close",
            "action_time": "next_trading_open",
            "features": [
                "20d_microcap_absolute_return_positive",
                "20d_microcap_return_above_equal_weight_market",
                "20d_microcap_positive_breadth_at_least_half",
                "20d_microcap_liquidity_at_least_60d_average",
            ],
            "allocation": {
                "zero_confirmations": 0,
                "one_confirmation": 5,
                "two_confirmations": 10,
                "three_confirmations": 15,
                "four_confirmations": 20,
            },
            "upgrade_confirmation_days": UPGRADE_CONFIRMATION_DAYS,
            "downgrade": "immediate",
            "missing_feature": "cash",
            "event_admission": "existing_frozen_acute_risk_off_clock",
            "execution": "next_open_cash_account_with_costs_t_plus_one_limits_and_capacity",
        },
        "periods": periods,
        "results": results,
        "allocation": allocation_audit,
        "event_risk": risk_audit,
        "recent": recent,
        "decision": {
            "passed_historical_safety_screen": all(checks.values()),
            "checks": checks,
            "verdict": (
                "CREATE_NEW_FORWARD_SHADOW_ACCOUNT"
                if all(checks.values())
                else "REJECT_ALLOCATOR"
            ),
        },
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
                        "metrics": result["metrics"],
                        "execution": result["execution"],
                        "integrity": result["integrity"],
                    }
                    for name, result in results.items()
                },
                "allocation": {
                    name: {key: value for key, value in row.items() if key != "decisions"}
                    for name, row in allocation_audit.items()
                },
                "recent": recent,
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
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=RESEARCH / "p0_microcap_escape_thresholds.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.thresholds,
        end=args.end,
        output=args.output,
    )


if __name__ == "__main__":
    main()
