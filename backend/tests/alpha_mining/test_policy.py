from __future__ import annotations

from app.alpha_mining.policy import evaluate_historical_gates


def test_missing_robustness_evidence_stays_pending_not_passed() -> None:
    gates = evaluate_historical_gates(
        {
            "stitched_oos_return": 0.5,
            "stitched_oos_sharpe": 1.2,
            "max_drawdown": -0.1,
            "positive_half_year_ratio": 0.8,
            "beat_champion_half_year_ratio": 0.7,
            "recent_1y_return": 0.2,
            "recent_3m_return": 0.03,
        },
        {
            "stitched_oos_return": 0.3,
            "stitched_oos_sharpe": 0.7,
            "max_drawdown": -0.15,
            "recent_1y_return": 0.1,
        },
    )
    by_id = {gate["id"]: gate for gate in gates}
    assert by_id["return_vs_champion"]["status"] == "passed"
    assert by_id["double_cost"]["status"] == "pending"
    assert by_id["forward_shadow"]["status"] == "pending"
