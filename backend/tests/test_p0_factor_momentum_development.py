from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_factor_momentum_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_factor_momentum", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_selector_uses_only_completed_history() -> None:
    months = [date(2020, month, 28) for month in range(1, 9)]
    completed = []
    for completion in months[:6]:
        completed.extend(
            [
                {
                    "sleeve_id": "high_52week",
                    "completion_date": completion,
                    "sleeve_return": 0.02,
                },
                {
                    "sleeve_id": "low_lottery",
                    "completion_date": completion,
                    "sleeve_return": 0.01,
                },
            ]
        )
    completed.append(
        {
            "sleeve_id": "low_lottery",
            "completion_date": months[7],
            "sleeve_return": 10.0,
        }
    )

    selections = study.select_factor_by_trailing_returns(
        completed, [months[5], months[6]]
    )

    assert selections[months[5]]["sleeve_id"] == "high_52week"
    assert selections[months[6]]["sleeve_id"] == "high_52week"


def test_selector_stays_in_cash_when_all_sleeves_negative() -> None:
    months = [date(2020, month, 28) for month in range(1, 7)]
    completed = [
        {
            "sleeve_id": "high_52week",
            "completion_date": completion,
            "sleeve_return": -0.01,
        }
        for completion in months
    ]

    selections = study.select_factor_by_trailing_returns(
        completed, [months[-1]]
    )

    assert selections == {}
