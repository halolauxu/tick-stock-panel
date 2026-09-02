from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_microcap_portfolio_risk_budget.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_microcap_portfolio_risk_budget", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_volatility_schedule_uses_signal_close_and_caps_weight() -> None:
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=index) for index in range(22)]
    rows = []
    for index, day in enumerate(dates):
        for rank in range(10):
            rows.append(
                {
                    "date": day,
                    "symbol": f"{rank:06d}.SZ",
                    "market_cap": float(rank + 1),
                    "amount": 1_000_000.0,
                    "daily_return": 0.10 if index % 2 else -0.10,
                }
            )
    panel = pl.DataFrame(rows)
    candidates = pl.DataFrame(
        {
            "date": [dates[20]],
            "entry_date": [dates[21]],
            "symbol": ["000000.SZ"],
            "market_cap": [1.0],
            "signal_amount": [1_000_000.0],
            "cap_rank": [1],
        }
    )

    schedule = study.build_volatility_exposure_schedule(panel, candidates)

    assert schedule.height == 1
    row = schedule.row(0, named=True)
    assert row["entry_date"] == dates[21]
    assert row["realized_volatility"] > 0.90
    assert 0 < row["target_exposure"] < study.MAX_MICROCAP_WEIGHT


def test_evaluate_requires_drawdown_return_execution_and_lot_feasibility() -> None:
    dynamic = {
        "metrics": {
            "account_annualized": 0.05,
            "account_max_drawdown": -0.08,
            "positive_account_years": 5,
        },
        "execution": {
            "buy": {
                "orders": 100,
                "execution_rate": 0.90,
                "rejection_reasons": {"zero_lot_or_cash": 10},
            },
            "sell": {"execution_rate": 0.90},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
        "exposure": {"median_active_positions": 12.0},
    }
    fixed = {
        "metrics": {
            "account_annualized": 0.06,
            "account_max_drawdown": -0.09,
        }
    }

    assert study.evaluate("development", dynamic, fixed)["passed"] is True

    dynamic["exposure"]["median_active_positions"] = 9.0
    failed = study.evaluate("development", dynamic, fixed)
    assert failed["passed"] is False
    assert "median_active_positions_at_least_10" in failed["failures"]
