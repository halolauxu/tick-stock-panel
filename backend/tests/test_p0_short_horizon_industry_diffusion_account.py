from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_short_horizon_industry_diffusion_account.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_short_horizon_industry_diffusion_account", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expansion_keeps_one_peer_per_industry_and_five_total() -> None:
    study = _load_module()
    seeds = pl.DataFrame(
        {
            "ann_date": [date(2020, 1, 1)] * 7,
            "entry_date": [date(2020, 1, 2)] * 7,
            "symbol": [f"S{index}" for index in range(7)],
            "l1_code": ["I1", "I1", "I2", "I3", "I4", "I5", "I6"],
            "l1_name": ["行业"] * 7,
            "source_p_change_min": [70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
            "source_p_change_max": [80.0] * 7,
            "prior_roe": [10.0] * 7,
            "five_day_industry_residual": [-0.01] * 7,
            "amount": [100_000_000.0] * 7,
        }
    )
    dates = [date(2020, 1, 2) + timedelta(days=index) for index in range(30)]

    expanded, audit = study.expand_candidates(seeds, dates, horizon=2)

    first_day = expanded.filter(pl.col("entry_date") == date(2020, 1, 2))
    assert first_day.height == 5
    assert first_day.get_column("l1_code").n_unique() == 5
    assert audit["seed_rows"] == 7
