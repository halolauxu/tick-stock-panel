from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_actual_corporate_buying_development.py"
)
SPEC = importlib.util.spec_from_file_location("p0_actual_corporate_buying", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holder_thresholds_are_mutually_exclusive() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH", "600003.SH"],
            "ann_date": [date(2020, 1, 1)] * 3,
            "category": ["corporate_increase"] * 3,
            "increase_float_ratio_pct": [0.5, 0.05, 0.2],
        }
    )

    result = MODULE.classify_holder_events(events)

    assert result.get_column("signal_group").to_list() == [
        MODULE.HOLDER_STRONG,
        MODULE.HOLDER_WEAK,
    ]


def test_repurchase_requires_both_intensity_dimensions() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH", "600003.SH"],
            "ann_date": [date(2020, 1, 1)] * 3,
            "category": ["completion"] * 3,
            "repurchase_amount_cny": [2_000_000.0, 2_000_000.0, 20_000.0],
            "market_cap": [1_000_000_000.0] * 3,
            "mean_amount_20d": [10_000_000.0, 20_000_000.0, 10_000_000.0],
        }
    )

    result = MODULE.classify_repurchase_events(events)

    assert result.select("symbol", "signal_group").to_dicts() == [
        {"symbol": "600001.SH", "signal_group": MODULE.REPURCHASE_STRONG},
        {"symbol": "600003.SH", "signal_group": MODULE.REPURCHASE_WEAK},
    ]


def test_entry_is_strictly_after_announcement() -> None:
    events = pl.DataFrame(
        {
            "ann_date": [date(2020, 1, 3), date(2020, 1, 4)],
            "symbol": ["600001.SH", "600002.SH"],
        }
    )
    dates = [date(2020, 1, 3), date(2020, 1, 6)]

    result = MODULE.map_entry_dates(events, dates)

    assert result.get_column("entry_date").to_list() == [
        date(2020, 1, 6),
        date(2020, 1, 6),
    ]
