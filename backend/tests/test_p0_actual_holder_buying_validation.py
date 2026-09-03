from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_actual_holder_buying_validation.py"
)
SPEC = importlib.util.spec_from_file_location("p0_actual_holder_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _arm(annualized: float, positive_years: int = 2) -> dict:
    return {
        "metrics": {
            "annualized": annualized,
            "positive_years": positive_years,
            "max_drawdown": -0.20,
            "mean_cash_ratio": 0.50,
        },
        "account": {"trade_count": 240},
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.96},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }


def test_validation_distinguishes_enhancement_from_standalone() -> None:
    payload = {
        "results": {
            MODULE.development.HOLDER_STRONG: _arm(0.12),
            MODULE.development.HOLDER_WEAK: _arm(0.05),
        },
        "benchmark": {"annualized": 0.10},
    }

    result = MODULE.evaluate_validation(payload)

    assert result["enhancement_passed"] is True
    assert result["deployable_standalone"] is False
    assert result["verdict"] == "RETAIN_AS_ENHANCEMENT"


def test_validation_terminates_when_direction_does_not_repeat() -> None:
    payload = {
        "results": {
            MODULE.development.HOLDER_STRONG: _arm(0.04),
            MODULE.development.HOLDER_WEAK: _arm(0.03),
        },
        "benchmark": {"annualized": 0.02},
    }

    result = MODULE.evaluate_validation(payload)

    assert result["enhancement_passed"] is False
    assert result["verdict"] == "TERMINATE_HOLDER_BUYING_INTENSITY"

