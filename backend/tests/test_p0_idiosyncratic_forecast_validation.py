from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_idiosyncratic_forecast_validation.py"
)
SPEC = importlib.util.spec_from_file_location("p0_idiosyncratic_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload(annualized: float, positive_years: int = 2) -> dict:
    return {
        "capital_tiers": {
            "200000": {
                "metrics": {
                    "annualized": annualized,
                    "positive_years": positive_years,
                    "max_drawdown": -0.20,
                    "mean_cash_ratio": 0.60,
                },
                "account": {"trade_count": 240},
                "intent_execution": {
                    "buy": {"execution_rate": 0.95},
                    "sell": {"execution_rate": 0.97},
                },
                "integrity": {
                    "ending_unresolved_positions": 0,
                    "max_cash_reconciliation_error": 0.0,
                },
            }
        },
        "benchmark": {"annualized": 0.10},
    }


def test_positive_sparse_signal_is_retained_as_enhancement() -> None:
    result = MODULE.evaluate_validation(_payload(0.12))

    assert result["enhancement_passed"] is True
    assert result["deployable_standalone"] is False
    assert result["verdict"] == "RETAIN_AS_ENHANCEMENT"


def test_negative_signal_is_terminated() -> None:
    result = MODULE.evaluate_validation(_payload(-0.01, positive_years=1))

    assert result["enhancement_passed"] is False
    assert result["verdict"] == "TERMINATE_IDIOSYNCRATIC_FORECAST"

