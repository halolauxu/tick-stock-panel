from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_baseline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


microcap = _load_module()


def test_chinext_limit_change_is_date_aware() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["300001.SZ", "300001.SZ", "688001.SH", "600001.SH"],
            "date": [
                date(2020, 8, 21),
                date(2020, 8, 24),
                date(2020, 8, 21),
                date(2020, 8, 24),
            ],
        }
    ).with_columns(microcap.price_limit_pct().alias("limit"))

    assert frame["limit"].to_list() == [0.10, 0.20, 0.20, 0.10]


def test_weekly_observation_uses_next_week_open_without_lookahead() -> None:
    start = date(2024, 1, 1)
    dates = []
    cursor = start
    while len(dates) < 20:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += timedelta(days=1)
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * len(dates),
            "date": dates,
            "name": ["浦发银行"] * len(dates),
            "_global_index": list(range(len(dates))),
            "list_date": [date(1999, 1, 1)] * len(dates),
            "open": [10.0] * len(dates),
            "raw_open": [10.0] * len(dates),
            "close": [10.0] * len(dates),
            "raw_close": [10.0] * len(dates),
            "volume": [100_000.0] * len(dates),
            "amount": [20_000_000.0] * len(dates),
            "total_shares": [1_000_000.0] * len(dates),
            "float_shares": [800_000.0] * len(dates),
            "market_cap": [10_000_000.0] * len(dates),
            "daily_return": [0.0] * len(dates),
            "limit_up_price": [11.0] * len(dates),
            "limit_down_price": [9.0] * len(dates),
            "is_limit_down": [False] * len(dates),
        }
    )

    observations = microcap.build_weekly_observations(panel)

    first = observations.sort("date").row(0, named=True)
    assert first["date"] == date(2024, 1, 5)
    assert first["entry_date"] == date(2024, 1, 8)
    assert first["exit_date"] == date(2024, 1, 15)
    assert first["cap_decile"] == 0
    assert first["tradable"] is True
    assert first["net_return"] < 0


def test_gate_requires_validation_and_stress() -> None:
    template = {
        "weeks": 100,
        "bottom_annualized_net": 0.30,
        "market_annualized_net": 0.10,
        "annualized_net_excess": 0.20,
        "bottom_annualized_mark": 0.31,
        "market_annualized_mark": 0.10,
        "annualized_mark_excess": 0.21,
        "bottom_max_drawdown": -0.20,
        "mean_bottom_tradable_rate": 0.90,
        "positive_excess_years": 2,
        "yearly": [],
    }
    metrics = [
        {"period": "validation", **template},
        {
            "period": "known_stress",
            **template,
            "annualized_net_excess": 0.05,
        },
    ]

    decision = microcap.evaluate_gate(metrics)

    assert decision["passed"] is False
    assert decision["verdict"] == "DOWNGRADE"


def test_compound_and_drawdown_are_path_aware() -> None:
    assert microcap._compound([0.10, -0.10]) == pytest.approx(-0.01)
    assert microcap._max_drawdown([0.10, -0.20, 0.05]) == pytest.approx(-0.20)
