from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_deepseek_main_board_short_horizon_screen.py"
)
SPEC = importlib.util.spec_from_file_location("p0_deepseek_short_horizon", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_composite_score_respects_frozen_directions() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, 2)] * 3,
            "symbol": ["600001.SH", "600002.SH", "600003.SH"],
            "higher": [1.0, 2.0, 3.0],
            "lower": [3.0, 2.0, 1.0],
        }
    )

    result = MODULE.add_composite_score(
        panel, {"higher": (1, 0.5), "lower": (-1, 0.5)}
    ).sort("symbol")

    assert result.get_column("composite_score").to_list() == pytest.approx([
        100.0 / 3.0,
        200.0 / 3.0,
        100.0,
    ])


def test_candidate_and_control_take_opposite_score_tails() -> None:
    scored = pl.DataFrame(
        {
            "date": [date(2020, 1, 2)] * 4,
            "symbol": ["600001.SH", "600002.SH", "600003.SH", "600004.SH"],
            "amount": [100_000_000.0] * 4,
            "composite_score": [90.0, 80.0, 20.0, 10.0],
        }
    )
    next_dates = pl.DataFrame(
        {"date": [date(2020, 1, 2)], "entry_date": [date(2020, 1, 3)]}
    )

    candidate = MODULE.build_candidates(scored, next_dates, control=False)
    control = MODULE.build_candidates(scored, next_dates, control=True)

    assert candidate.get_column("symbol").to_list() == ["600001.SH", "600002.SH"]
    assert control.get_column("symbol").to_list() == ["600004.SH", "600003.SH"]
