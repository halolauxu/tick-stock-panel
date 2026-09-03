from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_fundamental_negative_reaction_account.py"
)
SPEC = importlib.util.spec_from_file_location("p0_fundamental_negative_account", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "ann_date": date(2020, 1, 3),
                "report_announce_date": date(2020, 1, 2),
                "category": MODULE.reaction.NEGATIVE_CONTROL,
                "reaction_return": -0.04,
                "signal_amount": 100_000_000.0,
            },
            {
                "symbol": "600002.SH",
                "ann_date": date(2020, 1, 3),
                "report_announce_date": date(2020, 1, 2),
                "category": MODULE.reaction.NEGATIVE_CONTROL,
                "reaction_return": -0.01,
                "signal_amount": 100_000_000.0,
            },
            {
                "symbol": "600003.SH",
                "ann_date": date(2020, 1, 3),
                "report_announce_date": date(2020, 1, 2),
                "category": MODULE.reaction.CANDIDATE,
                "reaction_return": 0.03,
                "signal_amount": 100_000_000.0,
            },
        ],
        infer_schema_length=None,
    )


def test_negative_and_positive_arms_are_distinct_and_ranked() -> None:
    trading_dates = [date(2020, 1, 3), date(2020, 1, 6)]
    events = _events()

    negative = MODULE.build_candidates(
        events, trading_dates, MODULE.NEGATIVE_REACTION
    )
    positive = MODULE.build_candidates(
        events, trading_dates, MODULE.POSITIVE_REACTION
    )

    assert negative.get_column("symbol").to_list() == ["600001.SH", "600002.SH"]
    assert positive.get_column("symbol").to_list() == ["600003.SH"]
    assert negative.get_column("entry_date").unique().to_list() == [date(2020, 1, 6)]


def test_snapshot_must_exist_on_reaction_day() -> None:
    snapshots = pl.DataFrame(
        {
            "symbol": ["600001.SH"],
            "snapshot_date": [date(2020, 1, 2)],
            "signal_amount": [100_000_000.0],
        }
    )

    attached = MODULE.attach_exact_investable_snapshot(_events(), snapshots)

    assert attached.is_empty()

