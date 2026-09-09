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

SCHEMA_VERSION = "p1-microcap-participation-allocator-v1"
VALIDATION = (date(2021, 1, 1), date(2023, 12, 31))
STRESS_START = date(2024, 1, 1)
RECENT_START = date(2026, 3, 1)
TREND_WINDOW = 20
LIQUIDITY_BASE_WINDOW = 60
UPGRADE_CONFIRMATION_DAYS = 3
FULL_SLOTS = 20
HALF_SLOTS = 10
NO_SLOTS = 0


def attach_participation_features(features: pl.DataFrame) -> pl.DataFrame:
    """Describe whether the whole micro-cap cohort has tradable participation."""
    return (
        features.sort("date")
        .with_columns(
            (
                (pl.col("microcap_daily_return") + 1.0).rolling_map(
                    lambda values: values.product(),
                    window_size=TREND_WINDOW,
                    min_samples=TREND_WINDOW,
                )
                - 1.0
            ).alias("microcap_absolute_20d"),
            (
                (pl.col("microcap_daily_return") + 1.0).rolling_map(
                    lambda values: values.product(),
                    window_size=TREND_WINDOW,
                    min_samples=TREND_WINDOW,
                )
                / (pl.col("market_daily_return") + 1.0).rolling_map(
                    lambda values: values.product(),
                    window_size=TREND_WINDOW,
                    min_samples=TREND_WINDOW,
                )
                - 1.0
            ).alias("microcap_relative_20d"),
            pl.col("microcap_breadth")
            .rolling_mean(
                window_size=TREND_WINDOW,
                min_samples=TREND_WINDOW,
            )
            .alias("microcap_breadth_20d"),
            (
                pl.col("microcap_median_amount").rolling_mean(
                    window_size=TREND_WINDOW,
                    min_samples=TREND_WINDOW,
                )
                / pl.col("microcap_median_amount").rolling_mean(
                    window_size=LIQUIDITY_BASE_WINDOW,
                    min_samples=LIQUIDITY_BASE_WINDOW,
                )
            ).alias("microcap_liquidity_20d_60d"),
        )
        .with_columns(
            pl.sum_horizontal(
                (pl.col("microcap_absolute_20d") > 0).cast(pl.UInt8),
                (pl.col("microcap_relative_20d") > 0).cast(pl.UInt8),
                (pl.col("microcap_breadth_20d") >= 0.5).cast(pl.UInt8),
                (pl.col("microcap_liquidity_20d_60d") >= 1.0).cast(pl.UInt8),
            )
            .fill_null(0)
            .alias("participation_score")
        )
    )


def raw_slot_budget(feature: dict[str, Any]) -> int:
    """Map interpretable participation evidence to a capital budget."""
    required = (
        "microcap_absolute_20d",
        "microcap_relative_20d",
        "microcap_breadth_20d",
        "microcap_liquidity_20d_60d",
    )
    if any(feature.get(key) is None for key in required):
        return NO_SLOTS
    if bool(feature.get("severe_limit_down")):
        return NO_SLOTS
    score = int(feature.get("participation_score") or 0)
    if score >= 3:
        return FULL_SLOTS
    if score == 2:
        return HALF_SLOTS
    return NO_SLOTS


def build_allocation_clock(
    features: pl.DataFrame,
) -> tuple[dict[date, int], list[dict[str, Any]]]:
    """Apply close-known evidence at the next open with cautious upgrades."""
    rows = features.sort("date").to_dicts()
    slots_by_open: dict[date, int] = {}
    decisions: list[dict[str, Any]] = []
    active_slots = NO_SLOTS
    pending_upgrade = NO_SLOTS
    upgrade_days = 0
    audit_fields = (
        "microcap_absolute_20d",
        "microcap_relative_20d",
        "microcap_breadth_20d",
        "microcap_liquidity_20d_60d",
    )
    for index, row in enumerate(rows[:-1]):
        raw_slots = raw_slot_budget(row)
        if raw_slots <= active_slots:
            active_slots = raw_slots
            pending_upgrade = raw_slots
            upgrade_days = 0
        else:
            if pending_upgrade == raw_slots:
                upgrade_days += 1
            else:
                pending_upgrade = raw_slots
                upgrade_days = 1
            if upgrade_days >= UPGRADE_CONFIRMATION_DAYS:
                active_slots = raw_slots
                upgrade_days = 0
        action_date = rows[index + 1]["date"]
        slots_by_open[action_date] = active_slots
        decisions.append(
            {
                "decision_date": row["date"],
                "action_date": action_date,
                "participation_score": int(row.get("participation_score") or 0),
                "raw_microcap_slots": raw_slots,
                "microcap_slots": active_slots,
                "upgrade_confirmation_days": upgrade_days,
                **{key: row.get(key) for key in audit_fields},
            }
        )
    return slots_by_open, decisions


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
    return {
        "trading_days": len(scoped),
        "full_days": sum(row["microcap_slots"] == FULL_SLOTS for row in scoped),
        "half_days": sum(row["microcap_slots"] == HALF_SLOTS for row in scoped),
        "cash_days": sum(row["microcap_slots"] == NO_SLOTS for row in scoped),
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
            "allocation": {"four_or_three": 20, "two": 10, "zero_or_one": 0},
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
