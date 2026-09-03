from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "run_p0_short_horizon_event_account.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_short_horizon_event_account", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_result(mean_industry_excess: float = 0.01) -> dict:
    return {
        "event_study": {
            "tradable_events": 100,
            "tradable_rate": 0.95,
            "market_benchmark_coverage": 1.0,
            "industry_benchmark_coverage": 1.0,
            "unresolved_exits": 0,
            "mean_net_return": 0.02,
            "mean_market_excess": 0.01,
            "mean_industry_excess": mean_industry_excess,
            "market_excess_cluster_t": 2.5,
            "positive_market_excess_years": 6,
            "max_year_positive_share": 0.30,
        },
        "account": {
            "complete_round_trips": 80,
            "annualized": 0.12,
            "total_return": 0.50,
            "max_drawdown": -0.15,
            "positive_years": 6,
            "buy_intent_execution": 0.95,
            "sell_intent_execution": 0.96,
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
            "unexpected_over_horizon_cycles": 0,
            "top5_positive_profit_share": 0.25,
            "max_industry_asset_share": 0.20,
            "microcap_daily_correlation": 0.10,
            "double_cost_total_return": 0.30,
        },
    }


def test_development_gate_and_horizon_selection_are_frozen() -> None:
    study = _load_module()
    results = {
        "2": _passing_result(0.008),
        "5": _passing_result(0.015),
        "10": _passing_result(0.012),
    }

    decision = study.evaluate_development(results)

    assert decision["verdict"] == "PROMOTE_HORIZON_TO_VALIDATION"
    assert decision["selected_horizon"] == 5


def test_no_horizon_is_selected_when_concentration_gate_fails() -> None:
    study = _load_module()
    results = {"2": _passing_result()}
    results["2"]["account"]["top5_positive_profit_share"] = 0.31

    decision = study.evaluate_development(results)

    assert decision["verdict"] == "REJECT_EVENT_ACCOUNT"
    assert decision["selected_horizon"] is None
    assert "top5_positive_profit_share_at_most_30pct" in decision["horizons"]["2"]["failures"]
