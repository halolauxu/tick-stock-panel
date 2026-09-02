from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"


def _load(name: str):
    script = RESEARCH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load("collect_p0_microcap_defensive_etf_data")
study = _load("run_p0_microcap_defensive_etf_rotation_discovery")


def test_normalize_daily_converts_amount_to_cny() -> None:
    frame = collector.normalize_daily(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": "20260828",
                "open": "4.0",
                "high": "4.1",
                "low": "3.9",
                "close": "4.05",
                "vol": "1000",
                "amount": "123.5",
            }
        ]
    )

    assert frame.row(0, named=True)["amount"] == 123_500.0


def test_adjustment_requests_are_split_by_calendar_year() -> None:
    ranges = collector.year_ranges(date(2013, 7, 1), date(2015, 3, 2))

    assert ranges == [
        (date(2013, 7, 1), date(2013, 12, 31)),
        (date(2014, 1, 1), date(2014, 12, 31)),
        (date(2015, 1, 1), date(2015, 3, 2)),
    ]


def test_prepare_etf_panel_uses_only_trailing_adjusted_prices() -> None:
    start = date(2020, 1, 1)
    rows = []
    adjustments = []
    for index in range(121):
        day = start + timedelta(days=index)
        rows.append(
            {
                "symbol": "510300.SH",
                "date": day,
                "open": float(index + 1),
                "high": float(index + 1),
                "low": float(index + 1),
                "close": float(index + 1),
                "volume": 1000.0,
                "amount": 100_000_000.0,
            }
        )
        adjustments.append(
            {"symbol": "510300.SH", "date": day, "adj_factor": 1.0}
        )
    panel = study.prepare_etf_panel(
        pl.DataFrame(rows), pl.DataFrame(adjustments)
    )

    assert panel.row(-1, named=True)["momentum_120d"] == pytest.approx(120.0)
    assert panel.row(-2, named=True)["momentum_120d"] is None


def test_variant_switches_are_deterministic() -> None:
    days = [date(2026, 1, 2), date(2026, 1, 9), date(2026, 1, 16)]
    microcap = pl.DataFrame(
        {
            "date": days,
            "entry_date": [day + timedelta(days=3) for day in days],
            "exit_date": [day + timedelta(days=10) for day in days],
            "microcap_momentum_120d": [0.2, -0.1, 0.1],
            "microcap_return": [0.03, -0.04, 0.02],
        }
    )
    etf = pl.DataFrame(
        {
            "date": days,
            "symbol": ["518880.SH", "518880.SH", "518880.SH"],
            "etf_momentum_120d": [0.1, 0.2, 0.3],
            "etf_return": [0.01, 0.05, 0.04],
        }
    )
    variants = study.build_variant_returns(microcap, etf)

    assert variants["absolute_switch"].get_column(
        "weekly_return"
    ).to_list() == [0.03, 0.05, 0.02]
    assert variants["relative_rotation"].get_column(
        "weekly_return"
    ).to_list() == [0.03, 0.05, 0.04]
    assert variants["microcap_70_defensive_30"].get_column(
        "weekly_return"
    ).to_list() == pytest.approx([0.024, -0.013, 0.026])
