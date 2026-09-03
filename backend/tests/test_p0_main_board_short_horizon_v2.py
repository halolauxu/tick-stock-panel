from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_short_horizon_v2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_short_horizon_v2", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_expand_weekly_targets_carries_only_latest_frozen_selection() -> None:
    weekly = pl.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 9)],
            "entry_date": [date(2026, 1, 5), date(2026, 1, 12)],
            "symbol": ["A.SZ", "B.SZ"],
            "market_cap": [10.0, 9.0],
            "signal_amount": [100_000.0, 120_000.0],
            "cap_rank": [1, 1],
        }
    )
    action_dates = [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 9),
        date(2026, 1, 12),
        date(2026, 1, 13),
    ]

    result = study.expand_weekly_targets(weekly, action_dates)

    assert result.select("entry_date", "symbol").rows() == [
        (date(2026, 1, 5), "A.SZ"),
        (date(2026, 1, 6), "A.SZ"),
        (date(2026, 1, 9), "A.SZ"),
        (date(2026, 1, 12), "B.SZ"),
        (date(2026, 1, 13), "B.SZ"),
    ]
    assert result.filter(pl.col("symbol") == "A.SZ").get_column(
        "source_entry_date"
    ).unique().to_list() == [date(2026, 1, 5)]


def test_development_gate_is_frozen_and_fail_closed() -> None:
    baseline_metrics = {
        "account_annualized": 0.40,
        "account_max_drawdown": -0.45,
    }
    candidate = {
        "metrics": {
            "account_annualized": 0.36,
            "account_max_drawdown": -0.39,
            "yearly": [
                {"year": 2017, "account_return": -0.02},
                {"year": 2018, "account_return": 0.01},
            ],
        },
        "execution": {
            "buy": {"execution_rate": 0.90},
            "sell": {"execution_rate": 0.85},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
        "lifecycle": {
            "holding_buckets": {"under_2": 0},
            "normal_under_2_cycles": 0,
            "open_over_10_cycles": 0,
            "unexpected_over_10_cycles": 0,
            "reconstruction_issues": [],
        },
    }

    passed = study.evaluate_development_gate(candidate, baseline_metrics)
    assert passed["verdict"] == "PASS_TO_M2"
    assert passed["failures"] == []

    candidate["metrics"]["account_max_drawdown"] = -0.41
    failed = study.evaluate_development_gate(candidate, baseline_metrics)
    assert failed["verdict"] == "REJECT_M1"
    assert "drawdown_improves_at_least_5pp" in failed["failures"]


def test_lifecycle_report_separates_unavoidable_delayed_exit() -> None:
    days = [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 12),
        date(2026, 1, 13),
        date(2026, 1, 14),
        date(2026, 1, 15),
        date(2026, 1, 16),
        date(2026, 1, 19),
    ]
    simulation = {
        "orders": [
            {
                "date": days[0],
                "symbol": "A.SZ",
                "side": "BUY",
                "status": "FILLED",
                "cash_delta": -1000.0,
            },
            {
                "date": days[9],
                "symbol": "A.SZ",
                "side": "SELL",
                "status": "REJECTED",
                "reason": "limit_down",
                "exit_trigger": "max_holding_sessions",
            },
            {
                "date": days[10],
                "symbol": "A.SZ",
                "side": "SELL",
                "status": "FILLED",
                "reason": None,
                "exit_trigger": "max_holding_sessions",
                "cash_delta": 900.0,
            },
        ],
        "settlements": [],
    }

    result = study.build_lifecycle_report(simulation, days)

    assert result["delayed_max_hold_exit_cycles"] == 1
    assert result["unexpected_over_10_cycles"] == 0
    assert result["max_holding_sessions"] == 11
