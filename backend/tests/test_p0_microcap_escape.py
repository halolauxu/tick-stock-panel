from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_escape.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_escape", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


escape = _load_module()


def test_thresholds_use_only_development_rows() -> None:
    rows = []
    start = date(2020, 1, 1)
    for index in range(100):
        value = index / 100.0
        rows.append(
            {
                "date": start + timedelta(days=index),
                "microcap_excess_5d": value,
                "microcap_breadth_3d": value,
                "microcap_limit_down_3d": value,
                "microcap_liquidity_5d_60d": value,
            }
        )
    rows.append(
        {
            "date": date(2024, 1, 1),
            "microcap_excess_5d": -99.0,
            "microcap_breadth_3d": -99.0,
            "microcap_limit_down_3d": 99.0,
            "microcap_liquidity_5d_60d": -99.0,
        }
    )

    thresholds = escape.calibrate_thresholds(pl.DataFrame(rows))

    assert thresholds["microcap_excess_5d_p10"] == 0.10
    assert thresholds["microcap_breadth_3d_p10"] == 0.10
    assert thresholds["microcap_limit_down_3d_p90"] == 0.89
    assert thresholds["microcap_limit_down_3d_p95"] == 0.94
    assert thresholds["microcap_liquidity_5d_60d_p10"] == 0.10


def test_risk_clock_executes_next_open_and_requires_hold_plus_clean_closes() -> None:
    start = date(2020, 1, 1)
    rows = []
    for index in range(11):
        rows.append(
            {
                "date": start + timedelta(days=index),
                "ordinary_alarm_count": 2 if index == 0 else 0,
                "severe_limit_down": False,
                **{column: 0.0 for column in escape.FEATURE_COLUMNS},
            }
        )
    state, _, switches = escape.build_risk_clock(pl.DataFrame(rows))

    assert state[start + timedelta(days=1)] is False
    assert switches[0]["switch"] == "RISK_OFF"
    assert switches[0]["decision_date"] == start
    assert switches[0]["action_date"] == start + timedelta(days=1)
    assert switches[1]["switch"] == "RISK_ON"
    assert switches[1]["action_date"] == start + timedelta(days=6)


def test_empty_action_day_forces_exit_and_retries_blocked_sell() -> None:
    account = escape.account
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 9), date(2024, 1, 10)
    candidates = pl.DataFrame(
        [
            {
                "date": d0,
                "entry_date": d1,
                "symbol": "A.SZ",
                "cap_rank": 1,
                "signal_amount": 10_000_000.0,
            }
        ]
    )
    execution = pl.DataFrame(
        [
            {
                "symbol": "A.SZ",
                "entry_date": day,
                "quote_date": day,
                "open": raw_open,
                "raw_open": raw_open,
                "close": raw_open,
                "raw_close": raw_open,
                "entry_volume": 1_000_000.0,
                "entry_amount": 10_000_000.0,
                "limit_up_price": 11.0,
                "limit_down_price": limit_down,
                "exact_quote": True,
                "is_excluded_name": False,
            }
            for day, raw_open, limit_down in (
                (d1, 10.0, 9.0),
                (d2, 9.0, 9.0),
                (d3, 9.5, 9.0),
            )
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=[d1, d2, d3],
    )

    assert [row["side"] for row in result["trades"]] == ["BUY", "SELL"]
    rejected = [row for row in result["orders"] if row["status"] == "REJECTED"]
    assert rejected[0]["date"] == d2
    assert rejected[0]["reason"] == "limit_down"
    assert result["ending_positions"] == []
