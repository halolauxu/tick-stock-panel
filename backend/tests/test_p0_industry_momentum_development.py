from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_industry_momentum_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_industry_momentum", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_stock_momentum_requires_twenty_adjacent_market_observations() -> None:
    start = date(2020, 1, 1)
    rows = 21
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ"] * rows,
            "date": [start + timedelta(days=index) for index in range(rows)],
            "_global_index": list(range(rows)),
            "close": [10.0] * 20 + [12.0],
            "amount": [100_000_000.0] * rows,
        }
    )

    result = study.attach_stock_features(panel)

    assert result["stock_momentum_20d"][:20].null_count() == 20
    assert result["stock_momentum_20d"][-1] == pytest.approx(0.20)
    assert result["ma20"][-1] == pytest.approx(10.1)


def test_gate_enforces_return_drawdown_execution_and_integrity() -> None:
    result = {
        "metrics": {
            "annualized": 0.61,
            "max_drawdown": -0.22,
            "positive_years": 6,
            "mean_cash_ratio": 0.12,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.96},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }
    benchmark = {"annualized": 0.20}

    passed = study.evaluate_gate(result, benchmark)
    assert passed["passed"] is True
    assert passed["annualized_excess"] == pytest.approx(0.41)

    result["metrics"]["max_drawdown"] = -0.31
    failed = study.evaluate_gate(result, benchmark)
    assert failed["passed"] is False
    assert failed["checks"]["max_drawdown_no_worse_than_30pct"] is False
