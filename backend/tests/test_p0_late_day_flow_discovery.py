from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_late_day_flow_discovery.py"
    )
    spec = importlib.util.spec_from_file_location("p0_late_day_flow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_rank_arms_keeps_pre_registered_weak_and_strong_tails() -> None:
    day = date(2026, 1, 5)
    frame = pl.DataFrame(
        {
            "date": [day] * 10,
            "symbol": [f"S{index}" for index in range(10)],
            "late_return": [float(index) for index in range(10)],
        }
    )

    result = study.rank_arms(frame)

    weak = set(
        result.filter(pl.col("arm") == "weak_flow_reversal")["symbol"].to_list()
    )
    strong = set(
        result.filter(pl.col("arm") == "strong_flow_continuation")[
            "symbol"
        ].to_list()
    )
    assert weak == {"S0", "S1"}
    assert strong == {"S8", "S9"}
