from __future__ import annotations

# Requirements: AM-S5-001 through AM-S5-013.
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl

from app.alpha_mining.contracts import TrainOnlyContext, TrialBudget
from app.alpha_mining.registry import load_builtin_registry

EXPECTED_ENGINES = {
    "cross_sectional_rank",
    "event_sequence",
    "financial_revision",
    "market_sector_timing",
    "matched_outcomes",
    "network_diffusion",
    "nonlinear_interaction",
    "portfolio_residual",
}


def _context() -> TrainOnlyContext:
    start = date(2025, 1, 1)
    rows = []
    for day_index in range(70):
        day = (start + timedelta(days=day_index)).isoformat()
        for symbol_index in range(24):
            feature = float(symbol_index)
            rows.append({
                "date": day,
                "momentum": feature,
                "reversal": -feature + (day_index % 3) * 0.01,
                "quality": float(symbol_index % 7),
                "event_count_20d": feature,
                "industry_momentum_20d": feature,
                "pb_latest": feature,
                "_target_5d": feature / 100.0,
            })
    frame = pl.DataFrame(rows)
    return TrainOnlyContext(
        frame=frame,
        date_labels=tuple(frame.get_column("date").unique().sort().to_list()),
        feature_names=(
            "momentum",
            "reversal",
            "quality",
            "event_count_20d",
            "industry_momentum_20d",
            "pb_latest",
        ),
        target_column="_target_5d",
        asset_type="stock",
        metadata={},
    )


def test_builtin_registry_auto_discovers_runnable_engines() -> None:
    registry, failures = load_builtin_registry()
    assert failures == []
    assert {engine.manifest.engine_id for engine in registry.list()} == EXPECTED_ENGINES


def test_builtin_engines_discover_and_materialize_without_outer_test_context() -> None:
    registry, _ = load_builtin_registry()
    context = _context()
    budget = TrialBudget(max_candidates=2, max_trials=24, min_cross_section=20, min_dates=60)

    for engine in registry.list():
        candidates = engine.discover(context, budget)
        assert candidates, engine.manifest.engine_id
        frozen = engine.materialize(candidates[0], context)
        assert frozen.definition["kind"] == "factor_rank"
        assert frozen.engine_id == engine.manifest.engine_id


def test_builtin_engine_manifests_are_complete_and_constrained() -> None:
    registry, failures = load_builtin_registry()
    assert failures == []
    for engine in registry.list():
        manifest = engine.manifest.to_dict()
        assert manifest["frequencies"] == ["1d"]
        assert manifest["decision_clocks"] == ["after_close"]
        assert manifest["required_datasets"]
        assert manifest["forecast_horizons"]
        assert manifest["output_candidate_types"]
        assert manifest["parameter_contract_version"]
        assert manifest["artifact_contract_version"]
        assert manifest["auto_run_allowed"] is False
        assert manifest["information_domains"]
        assert manifest["mechanism_classes"]
        assert manifest["discovery_classes"]
        assert manifest["prediction_objects"]


def test_known_alpha_is_detected_but_seeded_random_noise_is_not_promoted() -> None:
    registry, _ = load_builtin_registry()
    strong = _context()
    budget = TrialBudget(max_candidates=2, max_trials=24, min_cross_section=20, min_dates=60)
    assert all(engine.discover(strong, budget) for engine in registry.list())

    rng = np.random.default_rng(20260829)
    rows = []
    for day_index in range(120):
        day = (date(2024, 1, 1) + timedelta(days=day_index)).isoformat()
        for symbol_index in range(50):
            rows.append({
                "date": day,
                "symbol": f"{symbol_index:06d}.SZ",
                "noise": float(rng.normal()),
                "event_count_20d": float(rng.normal()),
                "industry_momentum_20d": float(rng.normal()),
                "pb_latest": float(rng.normal()),
                "_target_5d": float(rng.normal()),
            })
    frame = pl.DataFrame(rows)
    noise = TrainOnlyContext(
        frame=frame,
        date_labels=tuple(frame["date"].unique().sort().to_list()),
        feature_names=("noise", "event_count_20d", "industry_momentum_20d", "pb_latest"),
        target_column="_target_5d",
        asset_type="stock",
    )
    assert all(engine.discover(noise, budget) == [] for engine in registry.list())


def test_every_discovery_attempt_including_failure_is_written_to_audit_sink() -> None:
    registry, _ = load_builtin_registry()
    budget = TrialBudget(max_candidates=2, max_trials=24, min_cross_section=20, min_dates=60)
    for engine in registry.list():
        audit: list[dict] = []
        context = replace(_context(), metadata={"trial_audit": audit})
        candidates = engine.discover(context, budget)
        assert audit, engine.manifest.engine_id
        assert {row["status"] for row in audit} <= {"eligible", "failed"}
        assert len(audit) >= len(candidates)
        assert all(row["recipe_id"] for row in audit)
