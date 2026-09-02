from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "research" / "run_p0_main_board_microcap_risk_overlay.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_main_board_microcap_risk_overlay", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _tier(*, annualized: float = 0.20, drawdown: float = -0.20) -> dict:
    return {
        "capital": 200_000.0,
        "metrics": {
            "account_annualized": annualized,
            "annualized_excess": 0.12,
            "account_max_drawdown": drawdown,
            "positive_account_years": 2,
        },
        "execution": {
            "buy": {"execution_rate": 0.90},
            "sell": {"execution_rate": 0.90},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }


def test_evaluate_period_requires_return_drawdown_execution_and_integrity() -> None:
    assert study.evaluate_period([_tier()])["verdict"] == "PASS"

    failed = study.evaluate_period([_tier(drawdown=-0.40)])
    assert failed["verdict"] == "FAIL"
    assert "max_drawdown_within_25pct" in failed["failures"]


def test_frozen_threshold_file_hash_is_enforced(tmp_path: Path) -> None:
    threshold_path = SCRIPT.parent / "p0_microcap_escape_thresholds.json"
    thresholds = study.load_frozen_thresholds(threshold_path)
    assert thresholds["microcap_excess_5d_p10"] < 0

    changed = tmp_path / "changed.json"
    changed.write_bytes(threshold_path.read_bytes() + b"\n")
    try:
        study.load_frozen_thresholds(changed)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("changed threshold file was accepted")
