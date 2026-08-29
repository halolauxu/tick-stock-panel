from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl

from app.alpha_mining.runtime import (
    AlphaRuntimeRequest,
    _aggregate_fold_metrics,
    _alpha_progress,
    _build_candidate_correlations,
    _build_discovery_summary,
    _build_failure_closure,
    _build_market_attribution,
    _curve_daily_returns,
    _sanitize_non_finite,
)


def _fold(start: date, days: int, daily_return: float = 0.001) -> dict:
    curve = []
    value = 1.0
    for offset in range(days):
        day = start + timedelta(days=offset)
        curve.append({"date": day.isoformat(), "value": value})
        value *= 1.0 + daily_return
    return {
        "test_start": start.isoformat(),
        "test_end": (start + timedelta(days=days - 1)).isoformat(),
        "metrics": {"equity_curve": curve},
    }


def test_recent_period_metrics_are_not_claimed_without_full_coverage() -> None:
    metrics = _aggregate_fold_metrics([_fold(date(2026, 5, 29), 92)])

    assert metrics["oos_start"] == "2026-05-29"
    assert metrics["oos_end"] == "2026-08-28"
    assert metrics["recent_1y_available"] is False
    assert metrics["recent_3m_available"] is False
    assert metrics["recent_1y_return"] is None
    assert metrics["recent_3m_return"] is None


def test_recent_period_metrics_require_and_report_complete_windows() -> None:
    metrics = _aggregate_fold_metrics([_fold(date(2025, 8, 28), 366)])

    assert metrics["recent_1y_available"] is True
    assert metrics["recent_3m_available"] is True
    assert metrics["recent_1y_return"] is not None
    assert metrics["recent_3m_return"] is not None
    assert metrics["oos_calendar_days"] == 366


def test_missing_backtest_curve_keeps_period_metrics_unavailable() -> None:
    metrics = _aggregate_fold_metrics([
        {
            "test_start": "2025-08-28",
            "test_end": "2026-08-28",
            "error": "inner selection produced no finite candidate",
        },
    ])

    assert metrics["oos_start"] is None
    assert metrics["oos_end"] is None
    assert metrics["recent_1y_return"] is None
    assert metrics["recent_3m_return"] is None


def test_training_rejection_is_not_mislabeled_as_engine_error() -> None:
    rows = _build_discovery_summary(
        ["cross_sectional_rank"],
        [{
            "engine_id": "cross_sectional_rank",
            "stage": "discovery",
            "status": "failed",
            "recipe_id": "preregistered.test",
            "evidence": {"reason": "preregistered_effect_floor"},
        }],
        {"cross_sectional_rank": [{
            "outer_index": 0,
            "selection_rejection": "inner selection produced no finite candidate",
        }]},
        SimpleNamespace(get=lambda _engine_id: SimpleNamespace(
            manifest=SimpleNamespace(name="截面排序发现"),
        )),
    )

    assert rows[0]["errors"] == 0
    assert rows[0]["selected_folds"] == 0


def test_selection_rejection_stays_signal_invalid_without_engine_error_category() -> None:
    request = AlphaRuntimeRequest(
        run_id="alpha-training-rejected",
        engine_ids=("cross_sectional_rank",),
        factor_names=("momentum_5d",),
        strategy_ids=(),
        champion_strategy_id=None,
        symbols=None,
        asset_type="stock",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        profile="exploratory",
        forward_horizon=5,
        commission_pct=0.0002,
        stamp_tax_pct=0.0005,
        slippage_bps=5,
        max_positions=10,
        max_candidates_per_engine=2,
        max_trials_per_engine=24,
    )
    analysis, _suggestions = _build_failure_closure(
        request=request,
        candidates=[{
            "engine_id": "cross_sectional_rank",
            "engine_name": "截面排序发现",
            "state": "rejected",
            "frozen_candidate": None,
            "metrics": {},
            "gates": [],
            "folds": [{"selection_rejection": "inner selection produced no finite candidate"}],
        }],
        market_attribution={},
        candidate_correlations=[],
        engine_failures=[],
        registry=SimpleNamespace(list=lambda: ()),
        catalog_datasets={},
        available_features=("momentum_5d",),
    )

    ids = {row["id"] for row in analysis["categories"]}
    assert "signal_invalid" in ids
    assert "engine_or_data_error" not in ids


def test_alpha_progress_reports_real_engine_budgets_and_counts() -> None:
    request = AlphaRuntimeRequest(
        run_id="alpha-progress",
        engine_ids=("engine-a", "engine-b"),
        factor_names=("momentum_5d",),
        strategy_ids=(),
        champion_strategy_id=None,
        symbols=None,
        asset_type="stock",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        profile="exploratory",
        forward_horizon=5,
        commission_pct=0.0002,
        stamp_tax_pct=0.0005,
        slippage_bps=5,
        max_positions=10,
        max_candidates_per_engine=2,
        max_trials_per_engine=24,
    )
    progress = _alpha_progress(
        phase="validation",
        label="滚动验证",
        done=2,
        total=4,
        request=request,
        engines={
            "engine-a": {
                "engine_id": "engine-a",
                "status": "completed",
                "folds_done": 1,
                "folds_total": 2,
                "trials": 7,
                "selected": 1,
                "backtests": 1,
                "errors": 0,
            },
            "engine-b": {
                "engine_id": "engine-b",
                "status": "running",
                "folds_done": 0,
                "folds_total": 2,
                "trials": 5,
                "selected": 0,
                "backtests": 0,
                "errors": 1,
            },
        },
        current_engine_id="engine-b",
    )

    assert progress["percent"] == 50.0
    assert progress["current_engine_id"] == "engine-b"
    assert progress["trials_used"] == 12
    assert progress["trial_limit"] == 96
    assert progress["frozen_candidates"] == 1
    assert progress["candidate_limit"] == 8
    assert progress["backtests"] == 1
    assert progress["engine_errors"] == 1
    assert progress["engines"][1]["status"] == "running"


def test_alpha_worker_result_replaces_non_finite_metrics_with_missing_evidence() -> None:
    result = _sanitize_non_finite({
        "score": float("inf"),
        "metrics": [1.0, float("-inf"), float("nan")],
    })

    assert result == {"score": None, "metrics": [1.0, None, None]}


def test_curve_returns_preserve_first_oos_day_from_initial_wealth() -> None:
    returns = _curve_daily_returns([
        {"date": "2026-01-02", "value": 1.01},
        {"date": "2026-01-05", "value": 0.9999},
    ])

    assert returns[0][0] == date(2026, 1, 2)
    assert round(returns[0][1], 6) == 0.01
    assert round(returns[1][1], 6) == -0.01


def test_market_attribution_uses_contemporaneous_breadth_and_refuses_concept_backfill() -> None:
    panel = pl.DataFrame({
        "date": [date(2026, 1, 2)] * 4 + [date(2026, 1, 5)] * 4,
        "symbol": ["A", "B", "C", "D"] * 2,
        "change_pct": [0.02, 0.01, 0.03, -0.01, -0.02, -0.01, -0.03, 0.01],
        "l1_name": ["行业甲", "行业甲", "行业乙", "行业乙"] * 2,
    })
    candidate = {
        "metrics": {"equity_curve": [
            {"date": "2026-01-02", "value": 1.01},
            {"date": "2026-01-05", "value": 1.00495},
        ]},
        "folds": [{"metrics": {"trades": [{
            "symbol": "A", "entry_date": "2026-01-02", "pnl_amount": 100.0,
        }]}}],
    }

    evidence = _build_market_attribution(candidate, panel)

    assert evidence["available"] is True
    assert evidence["daily"][0]["state"] == "strong_up"
    assert evidence["daily"][1]["state"] == "strong_down"
    assert evidence["industries"]["rows"][0]["industry"] == "行业甲"
    assert evidence["concepts"]["available"] is False
    assert "禁止倒填" in evidence["concepts"]["reason"]


def test_candidate_correlation_requires_aligned_oos_days() -> None:
    curve_a = [
        {"date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(), "value": 1 + index * 0.01}
        for index in range(6)
    ]
    curve_b = [
        {"date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(), "value": 1 + index * 0.02}
        for index in range(6)
    ]

    rows = _build_candidate_correlations([
        {"engine_id": "a", "metrics": {"equity_curve": curve_a}},
        {"engine_id": "b", "metrics": {"equity_curve": curve_b}},
    ])

    assert rows[0]["overlap_days"] == 6
    assert rows[0]["correlation"] is not None


def test_zero_pass_failure_closure_is_actionable_and_does_not_mutate_old_evidence() -> None:
    request = AlphaRuntimeRequest(
        run_id="alpha-failed-source",
        engine_ids=("cross_sectional_rank", "matched_outcomes"),
        factor_names=("momentum_5d",),
        strategy_ids=(),
        champion_strategy_id=None,
        symbols=None,
        asset_type="stock",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        profile="exploratory",
        forward_horizon=5,
        commission_pct=0.0002,
        stamp_tax_pct=0.0005,
        slippage_bps=5,
        max_positions=10,
        max_candidates_per_engine=2,
        max_trials_per_engine=24,
    )
    candidates = [
        {
            "engine_id": engine_id,
            "engine_name": engine_id,
            "state": "rejected",
            "frozen_candidate": {"recipe_id": f"recipe-{engine_id}"},
            "metrics": {"oos_days": 250, "stitched_oos_return": score},
            "gates": [
                {"id": "concentration", "status": "failed"},
                {"id": "recent_year", "status": "pending"},
            ],
            "folds": [],
        }
        for engine_id, score in (("cross_sectional_rank", 0.10), ("matched_outcomes", 0.08))
    ]
    original = deepcopy(candidates)

    analysis, suggestions = _build_failure_closure(
        request=request,
        candidates=candidates,
        market_attribution={
            engine_id: {"regimes": [{"state": "strong_up", "label": "强势上涨", "days": 250}]}
            for engine_id in request.engine_ids
        },
        candidate_correlations=[{
            "left_engine_id": "cross_sectional_rank",
            "right_engine_id": "matched_outcomes",
            "overlap_days": 250,
            "correlation": 0.99,
        }],
        engine_failures=[],
        registry=SimpleNamespace(list=lambda: ()),
        catalog_datasets={},
        available_features=("momentum_5d",),
    )

    assert candidates == original
    assert analysis["zero_pass"] is True
    assert {row["id"] for row in analysis["categories"]} >= {
        "capacity_concentration", "regime_dependency", "insufficient_coverage",
    }
    assert analysis["best_failed_candidate"]["engine_id"] == "cross_sectional_rank"
    assert analysis["excluded_recipe_ids"] == [
        "recipe-cross_sectional_rank", "recipe-matched_outcomes",
    ]
    assert suggestions
    suggestion = suggestions[0]
    assert suggestion["request_patch"]["engine_ids"] == ["cross_sectional_rank"]
    assert suggestion["keep"] and suggestion["changes"] and suggestion["why"]


def test_passing_candidate_does_not_trigger_failure_followup() -> None:
    request = AlphaRuntimeRequest(
        run_id="alpha-pass",
        engine_ids=("cross_sectional_rank",),
        factor_names=("momentum_5d",),
        strategy_ids=(),
        champion_strategy_id=None,
        symbols=None,
        asset_type="stock",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        profile="exploratory",
        forward_horizon=5,
        commission_pct=0.0002,
        stamp_tax_pct=0.0005,
        slippage_bps=5,
        max_positions=10,
        max_candidates_per_engine=2,
        max_trials_per_engine=24,
    )
    analysis, suggestions = _build_failure_closure(
        request=request,
        candidates=[{"engine_id": "cross_sectional_rank", "state": "research_candidate"}],
        market_attribution={},
        candidate_correlations=[],
        engine_failures=[],
        registry=SimpleNamespace(list=lambda: ()),
        catalog_datasets={},
        available_features=(),
    )

    assert analysis["zero_pass"] is False
    assert analysis["categories"] == []
    assert suggestions == []
