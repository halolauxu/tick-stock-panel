"""Causal capital budget for the main-board micro-cap sleeve."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

TREND_WINDOW = 20
LIQUIDITY_BASE_WINDOW = 60
UPGRADE_CONFIRMATION_DAYS = 3
FULL_SLOTS = 20
NO_SLOTS = 0
SLOTS_PER_CONFIRMATION = FULL_SLOTS // 4

FEATURE_FIELDS = (
    "microcap_absolute_20d",
    "microcap_relative_20d",
    "microcap_breadth_20d",
    "microcap_liquidity_20d_60d",
)


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
    """Give one quarter of the sleeve to each independent confirmation."""
    if any(feature.get(key) is None for key in FEATURE_FIELDS):
        return NO_SLOTS
    if bool(feature.get("severe_limit_down")):
        return NO_SLOTS
    score = int(feature.get("participation_score") or 0)
    return max(NO_SLOTS, min(FULL_SLOTS, score * SLOTS_PER_CONFIRMATION))


def advance_allocation_state(
    state: dict[str, Any], feature: dict[str, Any]
) -> tuple[dict[str, int], dict[str, Any]]:
    """Cut risk immediately; require three closed sessions before adding risk."""
    active_slots = int(state.get("microcap_slots", NO_SLOTS))
    pending_upgrade = int(state.get("pending_microcap_slots", active_slots))
    upgrade_days = int(state.get("upgrade_days", 0))
    raw_slots = raw_slot_budget(feature)
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
            pending_upgrade = raw_slots
            upgrade_days = 0
    next_state = {
        "microcap_slots": active_slots,
        "pending_microcap_slots": pending_upgrade,
        "upgrade_days": upgrade_days,
    }
    audit = {
        "decision_date": str(feature["date"]),
        "participation_score": int(feature.get("participation_score") or 0),
        "raw_microcap_slots": raw_slots,
        **next_state,
        **{key: feature.get(key) for key in FEATURE_FIELDS},
    }
    return next_state, audit


def build_allocation_clock(
    features: pl.DataFrame,
) -> tuple[dict[date, int], list[dict[str, Any]]]:
    """Map each close-known allocation decision to the next trading open."""
    rows = features.sort("date").to_dicts()
    slots_by_open: dict[date, int] = {}
    decisions: list[dict[str, Any]] = []
    state: dict[str, int] = {
        "microcap_slots": NO_SLOTS,
        "pending_microcap_slots": NO_SLOTS,
        "upgrade_days": 0,
    }
    for index, row in enumerate(rows[:-1]):
        state, audit = advance_allocation_state(state, row)
        action_date = rows[index + 1]["date"]
        slots_by_open[action_date] = state["microcap_slots"]
        decisions.append(
            {
                "decision_date": audit["decision_date"],
                "action_date": action_date,
                "participation_score": audit["participation_score"],
                "raw_microcap_slots": audit["raw_microcap_slots"],
                "microcap_slots": audit["microcap_slots"],
                "upgrade_confirmation_days": audit["upgrade_days"],
                **{key: audit.get(key) for key in FEATURE_FIELDS},
            }
        )
    return slots_by_open, decisions
