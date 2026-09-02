from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_main_board_forecast_drift_regime_discovery as study  # noqa: E402


def test_market_state_is_mapped_to_next_session() -> None:
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=index) for index in range(125)]
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * len(dates),
            "date": dates,
            "close": [100.0 + index for index in range(len(dates))],
            "daily_return": [0.01] * len(dates),
            "_global_index": list(range(len(dates))),
        }
    )

    states = study.build_market_states(panel)

    assert states.height == len(dates) - 1
    assert states.get_column("signal_date")[0] == dates[0]
    assert states.get_column("entry_date")[0] == dates[1]
    mature = states.filter(pl.col("entry_date") == dates[121]).row(0, named=True)
    assert mature["signal_date"] == dates[120]
    assert mature["market_120d_positive"] is True
    assert mature["breadth_60d_at_least_half"] is True


def test_gate_removes_candidates_on_closed_dates() -> None:
    candidates = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "entry_date": [date(2020, 1, 2), date(2020, 1, 3)],
            "symbol": ["600000.SH", "600001.SH"],
            "cap_rank": [1, 1],
        }
    )
    states = pl.DataFrame(
        {
            "entry_date": [date(2020, 1, 2), date(2020, 1, 3)],
            "gate": [False, True],
        }
    )

    gated = study.gate_candidates(candidates, states, "gate")

    assert gated.get_column("symbol").to_list() == ["600001.SH"]


def test_consecutive_retries_are_one_intent() -> None:
    dates = [date(2020, 1, day) for day in range(2, 7)]
    orders = [
        {
            "date": dates[0],
            "symbol": "600000.SH",
            "side": "BUY",
            "status": "REJECTED",
        },
        {
            "date": dates[1],
            "symbol": "600000.SH",
            "side": "BUY",
            "status": "REJECTED",
        },
        {
            "date": dates[2],
            "symbol": "600000.SH",
            "side": "BUY",
            "status": "FILLED",
        },
        {
            "date": dates[4],
            "symbol": "600000.SH",
            "side": "BUY",
            "status": "REJECTED",
        },
    ]

    summary = study.intent_execution_summary(orders, dates)

    assert summary["buy"]["intents"] == 2
    assert summary["buy"]["executed"] == 1
    assert summary["buy"]["execution_rate"] == 0.5


def test_gate_checks_require_improvement_and_integrity() -> None:
    control = {
        "metrics": {"annualized": 0.15, "max_drawdown": -0.38},
    }
    result = {
        "metrics": {
            "annualized": 0.23,
            "max_drawdown": -0.25,
            "positive_years": 5,
            "mean_cash_ratio": 0.60,
        },
        "intent_execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.96},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }

    checks = study.gate_checks(
        result, control, {"annualized": 0.12}, active=0.50
    )

    assert all(checks.values())

