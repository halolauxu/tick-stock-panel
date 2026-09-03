from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_industry_leader_pullback_development.py"
)
SPEC = importlib.util.spec_from_file_location("p0_industry_pullback", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _signal() -> pl.DataFrame:
    common = {
        "date": date(2020, 1, 3),
        "entry_date": date(2020, 1, 6),
        "l1_code": "801010",
        "l1_name": "行业",
        "industry_member_count": 20,
        "industry_momentum_20d": 0.10,
        "industry_breadth_5d": 0.70,
        "market_cap_percentile": 0.60,
        "market_cap": 5_000_000_000.0,
        "mean_amount_20d": 100_000_000.0,
        "raw_close": 10.0,
        "close": 10.0,
        "ma60": 9.0,
        "return_60d": 0.20,
        "amount": 100_000_000.0,
    }
    return pl.DataFrame(
        [
            {**common, "symbol": "600001.SH", "return_5d": -0.05},
            {**common, "symbol": "600002.SH", "return_5d": 0.05},
            {**common, "symbol": "600003.SH", "return_5d": 0.20},
        ],
        infer_schema_length=None,
    )


def test_pullback_and_chase_are_distinct() -> None:
    signal = _signal()

    pullback = MODULE.build_candidates(signal, MODULE.PULLBACK)
    chase = MODULE.build_candidates(signal, MODULE.CHASE)

    assert pullback.get_column("symbol").to_list() == ["600001.SH"]
    assert chase.get_column("symbol").to_list() == ["600002.SH"]
