from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_overnight_information_momentum_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0_overnight_information_momentum_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_overnight_feature_uses_current_open_and_prior_close_only() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(21)]
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 21,
            "date": dates,
            "_global_index": list(range(21)),
            "open": [100.0] + [110.0] * 20,
            "close": [100.0] * 21,
            "amount": [100_000_000.0] * 21,
        }
    )

    result = study.attach_overnight_features(frame)

    assert result["overnight_return"][-1] == pytest.approx(0.10)
    assert result["overnight_momentum_20d"][-1] == pytest.approx(0.10)


def test_candidates_rank_high_overnight_momentum_first() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * 2,
            "entry_date": [date(2020, 2, 3)] * 2,
            "symbol": ["LOW", "HIGH"],
            "market_cap": [2_000_000_000.0] * 2,
            "mean_amount_20d": [100_000_000.0] * 2,
            "raw_close": [10.0] * 2,
            "overnight_momentum_20d": [0.01, 0.02],
            "amount": [100_000_000.0] * 2,
        }
    )

    candidates = study.build_candidates(frame)

    assert candidates.get_column("symbol").to_list() == ["HIGH", "LOW"]
