from __future__ import annotations

# Requirements: AM-S6-006 through AM-S6-009.
from datetime import date

import polars as pl

from app.alpha_mining.policy import evaluate_historical_gates
from app.alpha_mining.runtime import _concentration_evidence, _perturb_definition


def test_parameter_perturbation_changes_frozen_execution_definition() -> None:
    definition = {
        "kind": "factor_rank",
        "scoring": {"momentum": 1.0},
        "directions": {"momentum": "high"},
        "parameters": {"entry_score": 70.0, "exit_score": 40.0, "top_rank": 20},
    }
    changed = _perturb_definition(
        definition,
        entry_delta=5,
        exit_delta=5,
        rank_ratio=0.5,
        delay=1,
    )
    assert changed["parameters"] == {
        "entry_score": 75.0,
        "exit_score": 45.0,
        "top_rank": 10,
        "entry_delay_days": 1,
    }
    assert definition["parameters"]["entry_score"] == 70.0


def test_concentration_gate_uses_trade_contributions_across_symbols_years_and_industries() -> None:
    trades = []
    panel_rows = []
    for index in range(20):
        symbol = f"{index:06d}.SZ"
        entry = date(2020 + index % 5, 1, 2)
        trades.append({
            "symbol": symbol,
            "entry_date": entry.isoformat(),
            "exit_date": entry.isoformat(),
            "pnl_amount": 1.0,
        })
        panel_rows.append({
            "symbol": symbol,
            "date": entry,
            "l1_code": f"industry-{index % 4}",
        })
    evidence = _concentration_evidence(
        [{"metrics": {"trades": trades}}],
        pl.DataFrame(panel_rows),
    )
    assert evidence["passed"] is True
    assert evidence["top_symbol_share"] == 0.05
    assert evidence["top_industry_share"] == 0.25

    concentrated = _concentration_evidence(
        [{"metrics": {"trades": [{**trades[0], "pnl_amount": 100.0}, *trades[1:]]}}],
        pl.DataFrame(panel_rows),
    )
    assert concentrated["passed"] is False


def test_all_historical_stress_gates_are_binary_when_evidence_is_present() -> None:
    metrics = {
        "stitched_oos_return": 0.20,
        "stitched_oos_sharpe": 1.20,
        "max_drawdown": -0.10,
        "positive_half_year_ratio": 0.80,
        "beat_champion_half_year_ratio": 0.70,
        "recent_1y_return": 0.15,
        "recent_3m_return": 0.03,
        "double_cost_return": 0.10,
        "delayed_entry_return": 0.08,
        "worst_parameter_return": 0.06,
        "capacity_passed": True,
        "concentration_passed": True,
    }
    champion = {
        "stitched_oos_return": 0.10,
        "stitched_oos_sharpe": 0.70,
        "max_drawdown": -0.15,
        "recent_1y_return": 0.08,
    }
    gates = evaluate_historical_gates(metrics, champion)
    historical = [gate for gate in gates if gate["id"] != "forward_shadow"]
    assert historical
    assert all(gate["status"] == "passed" for gate in historical)
