from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_academic_factor_main_board_relaxed.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_academic_factor_main_board_relaxed", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _result(
    *, annualized: float = 0.16, drawdown: float = -0.20
) -> dict:
    return {
        "metrics": {
            "annualized": annualized,
            "max_drawdown": drawdown,
            "positive_years": 5,
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


def test_relaxed_gate_keeps_strict_drawdown_limit() -> None:
    assert study.evaluate(_result(), {"annualized": 0.10})["passed"] is True

    decision = study.evaluate(
        _result(drawdown=-0.2501), {"annualized": 0.10}
    )
    assert decision["passed"] is False
    assert "max_drawdown_within_25pct" in decision["failures"]


def test_relaxed_gate_requires_five_positive_years_and_excess() -> None:
    result = _result(annualized=0.16)
    result["metrics"]["positive_years"] = 4

    decision = study.evaluate(result, {"annualized": 0.12})

    assert decision["passed"] is False
    assert "excess_at_least_5pp" in decision["failures"]
    assert "at_least_five_positive_years" in decision["failures"]
