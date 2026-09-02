from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_fund_ownership_breadth_main_board.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_fund_ownership_breadth_main_board", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _tier(
    *,
    capital: float = 200_000.0,
    annualized: float = 0.16,
    drawdown: float = -0.20,
) -> dict:
    return {
        "capital": capital,
        "metrics": {
            "annualized": annualized,
            "max_drawdown": drawdown,
            "positive_years": 2,
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


def test_relaxed_gate_still_rejects_excess_drawdown() -> None:
    passed = study.evaluate([_tier()], {"annualized": 0.10})
    assert passed["passed"] is True

    failed = study.evaluate(
        [_tier(drawdown=-0.2501)], {"annualized": 0.10}
    )
    assert failed["verdict"] == "TERMINATE"
    assert "max_drawdown_within_25pct" in failed["failures"]


def test_gate_requires_five_point_benchmark_excess() -> None:
    decision = study.evaluate(
        [_tier(annualized=0.16)], {"annualized": 0.111}
    )

    assert decision["passed"] is False
    assert "excess_at_least_5pp" in decision["failures"]


def test_only_primary_capital_controls_promotion() -> None:
    tiers = [
        _tier(),
        _tier(capital=300_000.0, annualized=-0.10, drawdown=-0.50),
    ]

    decision = study.evaluate(tiers, {"annualized": 0.10})

    assert decision["passed"] is True
    assert decision["capacity_checks"]["300000"][
        "cash_reconciled"
    ] is True
