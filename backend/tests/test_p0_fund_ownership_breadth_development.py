from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_fund_ownership_breadth_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0_fund_ownership_breadth_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_next_trading_day_is_strictly_after_disclosure_deadline() -> None:
    calendar = [date(2020, 4, 30), date(2020, 5, 6), date(2020, 5, 7)]

    assert study._next_trading_day(calendar, date(2020, 4, 30)) == date(2020, 5, 6)


def test_daily_targets_expire_at_frozen_liquidation() -> None:
    targets = pl.DataFrame(
        {
            "date": [date(2020, 4, 30)],
            "rebalance_date": [date(2020, 5, 6)],
            "symbol": ["000001.SZ"],
            "coverage_share_growth": [1.5],
            "fund_count_increase": [20],
            "market_cap": [10_000_000_000.0],
            "signal_amount": [100_000_000.0],
            "mean_amount_20d": [100_000_000.0],
            "cap_rank": [1],
        }
    )
    actions = [date(2020, 5, 6), date(2020, 5, 7), date(2020, 11, 2)]

    expanded = study.expand_daily_targets(
        targets,
        [date(2020, 5, 6)],
        actions,
        liquidation_start=date(2020, 11, 2),
    )

    assert expanded.get_column("entry_date").to_list() == actions[:2]


def test_development_pass_never_counts_as_goal_completion() -> None:
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
