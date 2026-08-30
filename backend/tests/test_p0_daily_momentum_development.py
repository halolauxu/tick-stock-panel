from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_daily_momentum_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_daily_momentum", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_candidates_require_up_market_large_liquid_non_limit_winners() -> None:
    signal_day = date(2020, 1, 2)
    entry_day = date(2020, 1, 3)
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ", "B.SZ", "C.SZ", "D.SZ"],
            "date": [signal_day] * 4,
            "daily_return": [0.08, 0.07, 0.01, 0.095],
            "market_return": [0.01] * 4,
            "market_cap": [200.0, 80.0, 300.0, 400.0],
            "median_market_cap": [150.0] * 4,
            "mean_amount_20d": [2e8] * 4,
            "raw_close": [10.0] * 4,
            "amount": [2e8] * 4,
        }
    ).vstack(
        pl.DataFrame(
            {
                "symbol": ["A.SZ"],
                "date": [entry_day],
                "daily_return": [0.0],
                "market_return": [0.0],
                "market_cap": [200.0],
                "median_market_cap": [150.0],
                "mean_amount_20d": [2e8],
                "raw_close": [10.0],
                "amount": [2e8],
            }
        )
    )

    result = study.build_candidates(panel)

    assert result.get_column("symbol").to_list() == ["A.SZ"]
    assert result.get_column("entry_date").to_list() == [entry_day]


def test_action_grid_includes_days_without_new_candidates() -> None:
    candidates = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "entry_date": [date(2020, 1, 3)],
        }
    )
    quotes = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "date": [date(2020, 1, 3), date(2020, 1, 6)],
            "amount": [1e8, 1e8],
            "volume": [1e6, 1e6],
        }
    )

    result = study.build_action_grid(
        candidates, quotes, [date(2020, 1, 3), date(2020, 1, 6)]
    )

    assert result.get_column("entry_date").to_list() == [
        date(2020, 1, 3),
        date(2020, 1, 6),
    ]
    assert result.get_column("exact_quote").to_list() == [True, True]


def test_state_strategy_does_not_require_full_time_investment() -> None:
    result = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_years": 6,
            "mean_cash_ratio": 0.70,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }
    benchmark = {"annualized": 0.20}

    decision = study.evaluate_gate(result, benchmark)

    assert decision["passed"] is True
    assert "mean_cash_ratio_at_most_25pct" not in decision["checks"]
