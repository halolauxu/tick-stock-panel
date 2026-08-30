from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_broker_gold_stock_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0_broker_gold_stock_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_next_trading_day_is_strictly_after_availability() -> None:
    calendar = [date(2021, 7, 2), date(2021, 7, 5), date(2021, 7, 6)]

    assert study._next_trading_day(calendar, date(2021, 7, 3)) == date(2021, 7, 5)
    assert study._next_trading_day(calendar, date(2021, 7, 5)) == date(2021, 7, 6)


def test_daily_targets_remain_active_until_liquidation() -> None:
    monthly = pl.DataFrame(
        {
            "date": [date(2021, 7, 3)],
            "rebalance_date": [date(2021, 7, 5)],
            "symbol": ["000001.SZ"],
            "broker_count": [3],
            "market_cap": [10_000_000_000.0],
            "signal_amount": [100_000_000.0],
            "mean_amount_20d": [100_000_000.0],
            "cap_rank": [1],
        }
    )
    actions = [date(2021, 7, 5), date(2021, 7, 6), date(2021, 8, 4)]

    expanded = study.expand_daily_targets(
        monthly,
        [date(2021, 7, 5)],
        actions,
        liquidation_start=date(2021, 8, 4),
    )

    assert expanded.get_column("entry_date").to_list() == actions[:2]


def test_gate_never_counts_development_as_goal_completion() -> None:
    result = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_years": 3,
            "mean_cash_ratio": 0.10,
        },
        "execution": {
            "buy": {"execution_rate": 1.0},
            "sell": {"execution_rate": 1.0},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }
    decision = study.evaluate_gate(result, {"annualized": 0.20})

    assert decision["passed"] is True
    assert decision["counts_toward_50pct_goal"] is False
