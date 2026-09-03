from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_idiosyncratic_forecast_surprise_account as study  # noqa: E402


def _account_result(
    *, annualized: float = 0.24, max_drawdown: float = -0.25
) -> dict:
    return {
        "metrics": {
            "annualized": annualized,
            "max_drawdown": max_drawdown,
            "positive_years": 5,
            "mean_cash_ratio": 0.60,
        },
        "account": {"trade_count": 240},
        "intent_execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.98},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }


def test_account_gate_promotes_only_complete_result() -> None:
    decision = study.evaluate(_account_result(), {"annualized": 0.12})

    assert decision["passed"] is True
    assert decision["verdict"] == "FREEZE_VALIDATION_CONTRACT"


def test_account_gate_rejects_insufficient_excess() -> None:
    decision = study.evaluate(
        _account_result(annualized=0.205), {"annualized": 0.12}
    )

    assert decision["passed"] is False
    assert "annualized_excess_at_least_10pp" in decision["failures"]

