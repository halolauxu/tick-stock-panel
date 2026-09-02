from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_microcap_expanding_seasonality_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_microcap_expanding_seasonality", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_expanding_rule_excludes_current_year_and_requires_three_years() -> None:
    rows = []
    for year, value in (
        (2014, -0.10),
        (2015, -0.10),
        (2016, -0.10),
        (2017, 0.50),
    ):
        signal = date(year, 1, 3)
        rows.append(
            {
                "date": signal,
                "entry_date": signal + timedelta(days=3),
                "exit_date": signal + timedelta(days=10),
                "microcap_return": value,
            }
        )
    result = study.build_expanding_variants(pl.DataFrame(rows))
    mean = result["expanding_same_month_mean"]

    assert mean.get_column("selected_asset").to_list() == [
        "microcap",
        "microcap",
        "microcap",
        "cash",
    ]
    assert mean.get_column("weekly_return").to_list()[-1] == 0.0


def test_positive_frequency_requires_strict_majority() -> None:
    rows = []
    for year, value in (
        (2014, 0.10),
        (2015, -0.10),
        (2016, 0.10),
        (2017, -0.10),
        (2018, 0.20),
    ):
        signal = date(year, 2, 1)
        rows.append(
            {
                "date": signal,
                "entry_date": signal,
                "exit_date": signal + timedelta(days=7),
                "microcap_return": value,
            }
        )
    result = study.build_expanding_variants(pl.DataFrame(rows))
    frequency = result["expanding_same_month_positive_frequency"]

    assert frequency.get_column("selected_asset").to_list()[-1] == "cash"
