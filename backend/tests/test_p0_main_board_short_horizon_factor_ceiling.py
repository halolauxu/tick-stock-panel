from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_short_horizon_factor_ceiling.py"
)
SPEC = importlib.util.spec_from_file_location("p0_short_factor_ceiling", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_direction_is_selected_only_from_discovery() -> None:
    start = date(2014, 1, 1)
    rows = []
    for day in range(110):
        row_date = start + timedelta(days=day)
        for index in range(12):
            rows.append(
                {
                    "date": row_date,
                    "symbol": f"600{index:03d}.SH",
                    "factor": float(index),
                    "net_return_2": float(index) / 1000,
                    "exit_date_2": row_date,
                }
            )
    panel = pl.DataFrame(rows)
    benchmark = {
        "discovery": {
            "annualized": 0.0,
            "max_drawdown": 0.0,
            "positive_years": 1,
            "rebalances": 110,
        },
        "confirmation": {
            "annualized": 0.0,
            "max_drawdown": 0.0,
            "positive_years": 1,
            "rebalances": 110,
        },
    }

    result = MODULE.screen_factor(
        panel,
        "factor",
        2,
        [start + timedelta(days=value) for value in range(110)],
        [start + timedelta(days=value) for value in range(110)],
        benchmark,
    )

    assert result["direction"] == "HIGH"
    assert result["direction_selected_on_discovery_only"] is True


def test_targets_start_at_next_open() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=value) for value in range(12)]
    panel = pl.DataFrame(
        {
            "symbol": ["600001.SH"] * len(dates),
            "date": dates,
            "open": [float(value) for value in range(10, 22)],
        }
    )

    targeted = MODULE.attach_targets(panel, dates)
    first = targeted.sort("date").row(0, named=True)

    assert first["entry_date_2"] == dates[1]
    assert first["exit_date_2"] == dates[3]
    assert first["net_return_2"] == (13.0 / 11.0 - 1.0 - MODULE.ROUND_TRIP_COST)
