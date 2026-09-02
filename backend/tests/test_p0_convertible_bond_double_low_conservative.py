from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_convertible_bond_double_low_conservative.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_cb_double_low_conservative", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _tier(*, annualized: float = 0.16, drawdown: float = -0.20) -> dict:
    return {
        "capital": 200_000.0,
        "metrics": {
            "annualized": annualized,
            "max_drawdown": drawdown,
            "positive_years": 3,
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


def test_gate_uses_relaxed_return_but_strict_account_thresholds() -> None:
    benchmark = {"annualized": 0.10}
    assert study.evaluate([_tier()], benchmark)["passed"] is True

    failed = study.evaluate([_tier(drawdown=-0.30)], benchmark)
    assert failed["verdict"] == "TERMINATE"
    assert "max_drawdown_within_25pct" in failed["failures"]


def test_gate_requires_five_point_excess() -> None:
    decision = study.evaluate([_tier(annualized=0.16)], {"annualized": 0.12})

    assert decision["passed"] is False
    assert "excess_at_least_5pp" in decision["failures"]
