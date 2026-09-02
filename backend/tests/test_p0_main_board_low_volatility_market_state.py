from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_low_volatility_market_state.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_low_volatility_market_state", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _result(*, drawdown: float = -0.20) -> dict:
    return {
        "metrics": {
            "annualized": 0.18,
            "max_drawdown": drawdown,
            "positive_years": 5,
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


def test_state_gate_requires_risk_and_active_coverage() -> None:
    assert study.evaluate(
        _result(), {"annualized": 0.12}, 0.50
    )["passed"] is True

    decision = study.evaluate(
        _result(drawdown=-0.26), {"annualized": 0.12}, 0.34
    )
    assert decision["passed"] is False
    assert "max_drawdown_within_25pct" in decision["failures"]
    assert "active_rebalance_ratio_at_least_35pct" in decision["failures"]


def test_state_gate_does_not_cap_cash_ratio() -> None:
    result = _result()
    result["metrics"]["mean_cash_ratio"] = 0.80

    decision = study.evaluate(result, {"annualized": 0.12}, 0.50)

    assert decision["passed"] is True
    assert "mean_cash_ratio_at_most_25pct" not in decision["checks"]
