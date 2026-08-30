from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_st_removal_repricing_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_st_removal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_build_events_keeps_only_real_st_removal() -> None:
    names = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ", "B.SZ", "B.SZ", "C.SZ", "C.SZ"],
            "name": ["*ST甲", "甲股份", "*ST乙", "乙退", "丙股份", "丙科技"],
            "start_date": [
                date(2019, 1, 1),
                date(2020, 1, 2),
                date(2019, 1, 1),
                date(2020, 1, 2),
                date(2019, 1, 1),
                date(2020, 1, 2),
            ],
            "end_date": [None] * 6,
        }
    )

    result = study.build_events(names)

    assert result.height == 1
    assert result["symbol"][0] == "A.SZ"
    assert result["ann_date"][0] == date(2020, 1, 2)
    assert result["category"][0] == "st_removal"
