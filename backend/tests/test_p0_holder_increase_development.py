from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_holder_increase_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_holder_increase", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _detail(
    symbol: str,
    ann_date: date,
    direction: str,
    holder_type: str,
) -> dict:
    return {
        "symbol": symbol,
        "ann_date": ann_date,
        "holder_name": f"holder-{holder_type}",
        "holder_type": holder_type,
        "direction": direction,
        "change_vol": 100.0,
        "change_ratio": 0.1,
        "after_share": 1000.0,
        "after_ratio": 1.0,
        "avg_price": 10.0,
        "total_share": 1000.0,
        "begin_date": ann_date,
        "close_date": ann_date,
    }


def test_aggregate_prioritizes_management_and_excludes_mixed_direction() -> None:
    event_date = date(2020, 1, 1)
    details = pl.DataFrame(
        [
            _detail("A", event_date, "IN", "C"),
            _detail("A", event_date, "IN", "G"),
            _detail("B", event_date, "IN", "P"),
            _detail("B", event_date, "DE", "C"),
            _detail("C", event_date, "DE", "C"),
        ]
    )

    result = study.aggregate_events(details)
    categories = dict(zip(result["symbol"], result["category"], strict=True))

    assert categories == {
        "A": "management_increase",
        "C": "decrease_control",
    }


def test_aggregate_uses_corporate_then_personal_categories() -> None:
    event_date = date(2020, 1, 1)
    details = pl.DataFrame(
        [
            _detail("A", event_date, "IN", "C"),
            _detail("B", event_date, "IN", "P"),
        ]
    )

    result = study.aggregate_events(details)
    categories = dict(zip(result["symbol"], result["category"], strict=True))

    assert categories == {
        "A": "corporate_increase",
        "B": "personal_increase",
    }


def test_cooldown_keeps_only_first_event_inside_180_days() -> None:
    details = pl.DataFrame(
        [
            _detail("A", date(2020, 1, 1), "IN", "G"),
            _detail("A", date(2020, 6, 1), "IN", "G"),
            _detail("A", date(2020, 6, 29), "IN", "G"),
        ]
    )

    result = study.aggregate_events(details)

    assert result["ann_date"].to_list() == [date(2020, 1, 1), date(2020, 6, 29)]
