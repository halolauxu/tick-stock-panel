from __future__ import annotations

from datetime import date

import polars as pl

from app.services import microcap_participation as participation


def _feature(day: date, score: int, *, ready: bool = True) -> dict:
    value = 1.0 if ready else None
    return {
        "date": day,
        "participation_score": score,
        "microcap_absolute_20d": value,
        "microcap_relative_20d": value,
        "microcap_breadth_20d": value,
        "microcap_liquidity_20d_60d": value,
        "severe_limit_down": False,
    }


def test_one_independent_confirmation_funds_one_quarter_of_sleeve() -> None:
    assert participation.raw_slot_budget(_feature(date(2026, 9, 8), 0)) == 0
    assert participation.raw_slot_budget(_feature(date(2026, 9, 8), 1)) == 5
    assert participation.raw_slot_budget(_feature(date(2026, 9, 8), 2)) == 10
    assert participation.raw_slot_budget(_feature(date(2026, 9, 8), 3)) == 15
    assert participation.raw_slot_budget(_feature(date(2026, 9, 8), 4)) == 20


def test_missing_evidence_and_severe_limit_down_fail_closed() -> None:
    missing = _feature(date(2026, 9, 8), 4, ready=False)
    severe = {**_feature(date(2026, 9, 8), 4), "severe_limit_down": True}

    assert participation.raw_slot_budget(missing) == 0
    assert participation.raw_slot_budget(severe) == 0


def test_allocation_clock_uses_prior_close_and_asymmetric_confirmation() -> None:
    days = [date(2026, 9, day) for day in range(1, 8)]
    features = pl.DataFrame(
        [
            _feature(days[0], 4),
            _feature(days[1], 4),
            _feature(days[2], 4),
            _feature(days[3], 2),
            _feature(days[4], 1),
            _feature(days[5], 4),
            _feature(days[6], 4),
        ]
    )

    slots, decisions = participation.build_allocation_clock(features)

    assert slots[days[1]] == 0
    assert slots[days[2]] == 0
    assert slots[days[3]] == 20
    assert slots[days[4]] == 10
    assert slots[days[5]] == 5
    assert slots[days[6]] == 5
    assert decisions[0]["decision_date"] == str(days[0])
    assert decisions[0]["action_date"] == days[1]
