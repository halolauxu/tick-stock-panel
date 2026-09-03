from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_forecast_validation.py"
)
SPEC = importlib.util.spec_from_file_location("p0_main_board_forecast_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload(annualized: float, positive_years: int = 3, trades: int = 400) -> dict:
    return {
        "capital_tiers": {
            "200000": {
                "metrics": {
                    "annualized": annualized,
                    "positive_years": positive_years,
                    "max_drawdown": -0.20,
                    "mean_cash_ratio": 0.60,
                },
                "account": {"trade_count": trades},
                "execution": {
                    "buy": {"execution_rate": 0.95},
                    "sell": {"execution_rate": 0.97},
                },
                "integrity": {
                    "ending_unresolved_positions": 0,
                    "max_cash_reconciliation_error": 0.0,
                },
            }
        },
        "benchmark": {"annualized": 0.08},
    }


def test_stable_but_weak_account_is_retained_for_combination() -> None:
    result = MODULE.evaluate_validation(_payload(0.11))

    assert result["combination_input_passed"] is True
    assert result["deployable_standalone"] is False
    assert result["verdict"] == "RETAIN_AS_COMBINATION_INPUT"


def test_unstable_account_is_terminated() -> None:
    result = MODULE.evaluate_validation(_payload(0.11, positive_years=2))

    assert result["combination_input_passed"] is False
    assert result["verdict"] == "TERMINATE_MAIN_BOARD_FORECAST"
