from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_cn_commodity_futures_reversal_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_reversal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_reversal_longs_losers_and_shorts_winners() -> None:
    signal_day = date(2020, 1, 10)
    entry_day = date(2020, 1, 13)
    # Build a 6-day history so the frozen five-day lag exists for every series.
    history = []
    for index in range(8):
        for offset in range(6):
            history.append(
                {
                    "date": date(2020, 1, 5 + offset),
                    "series": f"S{index}",
                    "return_index": 1.0
                    + (index - 3.5) * offset / 500.0,
                    "_global_index": offset,
                    "volatility_20d": 0.02,
                }
            )
    schedule = pl.DataFrame(
        {"signal_date": [signal_day], "entry_date": [entry_day]}
    )

    signals = study.build_reversal_signals(
        pl.DataFrame(history, infer_schema_length=None), schedule
    )
    weights = dict(zip(signals["series"], signals["target_weight"], strict=True))

    assert all(weights[f"S{index}"] > 0 for index in range(4))
    assert all(weights[f"S{index}"] < 0 for index in range(4, 8))
    assert sum(abs(value) for value in weights.values()) == pytest.approx(3.0)
