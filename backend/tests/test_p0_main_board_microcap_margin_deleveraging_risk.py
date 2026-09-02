from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "run_p0_main_board_microcap_margin_deleveraging_risk",
    ROOT / "research" / "run_p0_main_board_microcap_margin_deleveraging_risk.py",
)
assert SPEC and SPEC.loader
study = module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_daily_metrics_use_five_comparable_days() -> None:
    start = date(2020, 1, 1)
    rows = []
    for index in range(6):
        for symbol, multiplier in (("600001.SH", 0.90), ("600002.SH", 1.01)):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": start + timedelta(days=index),
                    "rzye": 100.0 * multiplier**index,
                }
            )

    result = study.build_daily_metrics(pl.DataFrame(rows))

    assert result.height == 1
    assert result.row(0, named=True)["deleverage_breadth_5d"] == pytest.approx(0.5)


def test_risk_state_uses_only_previous_252_metric_days() -> None:
    start = date(2019, 1, 1)
    rows = []
    for index in range(253):
        rows.append(
            {
                "trade_date": start + timedelta(days=index),
                "deleverage_breadth_5d": 0.50 if index == 252 else 0.01,
                "aggregate_balance_change_5d": -0.10 if index == 252 else 0.0,
            }
        )

    result = study.build_risk_state(pl.DataFrame(rows))

    assert result["risk_off"].head(252).sum() == 0
    assert result.row(252, named=True)["risk_off"] is True
    assert result.row(252, named=True)["breadth_historical_threshold"] == 0.01
    assert result.row(252, named=True)["balance_historical_threshold"] == 0.0


def test_weekly_state_never_uses_future_margin_date() -> None:
    weekly = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "entry_date": [date(2020, 1, 6)],
        }
    )
    risk = pl.DataFrame(
        {
            "trade_date": [date(2020, 1, 2), date(2020, 1, 6)],
            "risk_off": [False, True],
        }
    )

    result = study.attach_weekly_state(weekly, risk)

    assert result.row(0, named=True)["trade_date"] == date(2020, 1, 2)
    assert result.row(0, named=True)["risk_off"] is False
