from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "research" / "run_p0_holder_concentration_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_holder_concentration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_events_use_previous_announcement_not_future_measurement():
    frame = pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 4,
            "ann_date": [
                date(2013, 12, 31),
                date(2014, 3, 1),
                date(2014, 3, 1),
                date(2014, 6, 15),
            ],
            "end_date": [
                date(2013, 12, 1),
                date(2014, 2, 1),
                date(2014, 2, 15),
                date(2014, 5, 15),
            ],
            "holder_num": [10000.0, 9000.0, 8000.0, 7900.0],
        }
    )

    events = study.build_events(frame)

    assert events.height == 1
    event = events.row(0, named=True)
    assert event["ann_date"] == date(2014, 3, 1)
    assert event["end_date"] == date(2014, 2, 15)
    assert event["holder_count_change"] == pytest.approx(-0.2)


def test_promotion_rejects_insufficient_absolute_return():
    metrics = {
        "tradable_events": 300,
        "announcement_days": 150,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_return": 0.029,
        "mean_excess_return": 0.021,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.4,
    }

    assert study.promotion_passed(metrics) is False
