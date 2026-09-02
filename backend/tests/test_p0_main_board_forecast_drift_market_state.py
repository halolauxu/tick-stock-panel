from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_main_board_forecast_drift_market_state.py"
    )
    spec = importlib.util.spec_from_file_location("forecast_state", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_market_state_is_lagged_to_the_next_entry_day() -> None:
    days = [date(2013, 9, 1) + timedelta(days=index) for index in range(130)]
    rows = []
    for symbol, slope in (("600000.SH", 0.01), ("600001.SH", 0.02)):
        for index, day in enumerate(days):
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "close": 10.0 + slope * index,
                    "daily_return": 0.001,
                }
            )
    panel = pl.DataFrame(rows)

    result = study.build_market_states(panel)

    assert result.height > 0
    assert all(
        row["entry_date"] == row["state_date"] + timedelta(days=1) for row in result.to_dicts()
    )
    assert result.get_column("trend_and_breadth").all()


def test_event_gate_uses_state_of_mapped_next_trading_day() -> None:
    friday = date(2020, 1, 3)
    monday = date(2020, 1, 6)
    events = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH"],
            "ann_date": [friday, monday],
            "p_change_min": [20.0, 30.0],
        }
    )
    states = pl.DataFrame(
        {
            "entry_date": [monday, date(2020, 1, 7)],
            "trend_120_positive": [True, False],
            "breadth_120_majority": [True, True],
            "trend_and_breadth": [True, False],
        }
    )

    result, audit = study.filter_events_by_entry_state(
        events,
        states,
        [monday, date(2020, 1, 7)],
        "trend_and_breadth",
    )

    assert result.get_column("symbol").to_list() == ["600000.SH"]
    assert audit["eligible_events"] == 1


def test_gate_requires_absolute_and_control_improvement() -> None:
    control = {"metrics": {"annualized": 0.15, "max_drawdown": -0.37}}
    result = {
        "metrics": {
            "annualized": 0.25,
            "max_drawdown": -0.25,
            "positive_years": 6,
            "mean_cash_ratio": 0.60,
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }

    decision = study.evaluate_variant(result, {"annualized": 0.12}, control)

    assert decision["passed"] is True
    result["metrics"]["max_drawdown"] = -0.34
    assert study.evaluate_variant(result, {"annualized": 0.12}, control)["passed"] is False
