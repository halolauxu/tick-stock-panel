from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_state_aware_return_screen.py"
    )
    spec = importlib.util.spec_from_file_location("p0_state_aware", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_revised_momentum_excludes_limit_up_day() -> None:
    start = date(2013, 1, 1)
    rows = 121
    frame = pl.DataFrame(
        {
            "symbol": ["A.SZ"] * rows,
            "date": [start + timedelta(days=index) for index in range(rows)],
            "_global_index": list(range(rows)),
            "daily_return": [0.01] * rows,
            "raw_close": [10.0] * rows,
            "limit_up_price": [11.0] * 120 + [10.0],
            "amount": [100_000_000.0] * rows,
            "market_cap": [10_000_000_000.0] * rows,
            "list_date": [date(2010, 1, 1)] * rows,
        }
    )

    result = study.attach_return_features(frame).tail(1)

    expected = (1.01**119) - 1.0
    assert result["is_limit_up"][0]
    assert result["revised_momentum_120d"][0] == pytest.approx(expected)


def test_signal_observations_executes_after_signal() -> None:
    frame = pl.DataFrame(
        {
            "date": [
                date(2014, 1, 30),
                date(2014, 1, 31),
                date(2014, 2, 3),
                date(2014, 2, 4),
            ],
            "symbol": ["A.SZ"] * 4,
        }
    )

    observations, actions = study.signal_observations(frame, "monthly")

    assert observations.select("date", "entry_date").to_dicts() == [
        {"date": date(2014, 1, 31), "entry_date": date(2014, 2, 3)}
    ]
    assert actions == [date(2014, 2, 3)]
