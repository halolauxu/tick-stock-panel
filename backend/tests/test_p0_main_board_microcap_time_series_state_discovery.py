from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_microcap_time_series_state_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_microcap_time_series_state", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_state_rules_only_use_previously_completed_returns() -> None:
    start = date(2020, 1, 3)
    values = [0.10, -0.20, 0.05, 0.05, -0.10]
    frame = pl.DataFrame(
        {
            "date": [start + timedelta(days=7 * i) for i in range(5)],
            "entry_date": [
                start + timedelta(days=7 * i + 3) for i in range(5)
            ],
            "exit_date": [
                start + timedelta(days=7 * i + 10) for i in range(5)
            ],
            "microcap_return": values,
        }
    )
    result = study.build_state_variants(frame)

    assert result["previous_week_momentum"].get_column(
        "weekly_return"
    ).to_list() == [0.0, -0.20, 0.0, 0.05, -0.10]
    assert result["previous_week_reversal"].get_column(
        "weekly_return"
    ).to_list() == [0.0, 0.0, 0.05, 0.0, 0.0]
    assert result["four_week_reversal"].get_column(
        "weekly_return"
    ).to_list()[-1] == -0.10
