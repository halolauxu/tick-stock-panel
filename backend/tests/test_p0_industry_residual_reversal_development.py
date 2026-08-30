from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_industry_residual_reversal_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_industry_reversal", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_candidates_enforce_two_per_industry_before_global_ten():
    signal_date = date(2020, 1, 3)
    entry_date = date(2020, 1, 6)
    rows = []
    for index in range(12):
        industry = "A" if index < 5 else f"I{index}"
        rows.append(
            {
                "symbol": f"{index:06d}.SZ",
                "date": signal_date,
                "entry_hint": entry_date,
                "l1_code": industry,
                "l1_name": industry,
                "industry_members": 30,
                "stock_return_5d": -0.20 + index * 0.005,
                "industry_return_5d": 0.0,
                "residual_return_5d": -0.20 + index * 0.005,
                "mean_amount_20d": 100_000_000.0,
                "raw_close": 10.0,
                "amount": 100_000_000.0 - index,
                "market_cap": 1_000_000_000.0 + index,
            }
        )
    panel = pl.DataFrame(rows)
    extra = pl.DataFrame(
        {
            "symbol": ["999999.SZ"],
            "date": [entry_date],
            "entry_hint": [None],
            "l1_code": ["X"],
            "l1_name": ["X"],
            "industry_members": [30],
            "stock_return_5d": [0.0],
            "industry_return_5d": [0.0],
            "residual_return_5d": [0.0],
            "mean_amount_20d": [100_000_000.0],
            "raw_close": [10.0],
            "amount": [100_000_000.0],
            "market_cap": [1_000_000_000.0],
        }
    )

    candidates = study.build_candidates(pl.concat([panel, extra]))

    assert candidates.height == 9
    assert candidates.filter(pl.col("l1_code") == "A").height == 2
    assert candidates.get_column("cap_rank").max() == 9


def test_gate_requires_five_positive_years():
    result = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_years": 4,
            "mean_cash_ratio": 0.10,
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

    decision = study.evaluate_gate(result, {"annualized": 0.10})

    assert decision["passed"] is False
    assert decision["checks"]["at_least_five_positive_years"] is False
