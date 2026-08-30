from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_defensive_trend_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0_microcap_defensive_trend_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_defensive_filter_keeps_positive_trend_and_lower_half_volatility() -> None:
    day = date(2020, 1, 3)
    frame = pl.DataFrame(
        {
            "date": [day] * 5,
            "entry_date": [date(2020, 1, 6)] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "cap_decile": [0, 0, 0, 0, 1],
            "cap_rank": [1, 2, 3, 4, 5],
            "momentum_60d": [0.1, -0.1, 0.2, 0.3, 0.5],
            "annual_vol_20d": [0.1, 0.2, 0.3, 0.4, 0.01],
        }
    )

    selected = study.defensive_filter(frame)

    assert selected.get_column("symbol").to_list() == ["A"]
    assert selected.get_column("micro_vol_median").to_list() == [0.25]


def test_gate_requires_return_risk_execution_and_integrity() -> None:
    metrics = {
        "account_annualized": 0.55,
        "annualized_excess": 0.30,
        "max_drawdown": -0.25,
        "positive_years": 5,
    }
    execution = {
        "buy": {"execution_rate": 0.95},
        "sell": {"execution_rate": 0.95},
    }
    integrity = {
        "ending_unresolved_positions": 0,
        "max_cash_reconciliation_error": 0.0,
    }

    passed = study.evaluate_gate(metrics, execution, integrity)
    metrics["account_annualized"] = 0.49
    failed = study.evaluate_gate(metrics, execution, integrity)

    assert passed["promoted"] is True
    assert failed["promoted"] is False
    assert failed["checks"]["annualized"] is False
