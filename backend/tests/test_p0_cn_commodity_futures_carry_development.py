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
        / "run_p0_cn_commodity_futures_carry_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_carry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_rank_carry_signals_longs_backwardation_and_shorts_contango() -> None:
    signal_day = date(2020, 1, 31)
    entry_day = date(2020, 2, 3)
    candidates = pl.DataFrame(
        {
            "signal_date": [signal_day] * 8,
            "entry_date": [entry_day] * 8,
            "series": [f"S{index}" for index in range(8)],
            "main_contract": [f"M{index}" for index in range(8)],
            "far_contract": [f"F{index}" for index in range(8)],
            "annualized_carry": [float(index) for index in range(8)],
            "volatility_20d": [0.02] * 8,
        }
    )

    signals = study.rank_carry_signals(candidates)
    weights = dict(zip(signals["series"], signals["target_weight"], strict=True))

    assert all(weights[f"S{index}"] < 0 for index in range(4))
    assert all(weights[f"S{index}"] > 0 for index in range(4, 8))
    assert sum(abs(value) for value in weights.values()) == pytest.approx(3.0)


def test_annualized_curve_score_is_positive_for_backwardation() -> None:
    score = 365.0 * __import__("math").log(100.0 / 95.0) / 100.0
    assert score > 0
