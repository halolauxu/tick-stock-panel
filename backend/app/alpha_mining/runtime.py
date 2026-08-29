"""Spawn-only Alpha discovery runtime with train-only engines and central OOS execution."""
# Requirements: AM-S5-001 through AM-S6-012.
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from app.alpha_mining.contracts import (
    CandidateSpec,
    DataCatalogContext,
    TrainOnlyContext,
    TrialBudget,
    qualify_manifest_datasets,
)
from app.alpha_mining.data_catalog import AlphaResearchDataCatalog
from app.alpha_mining.engines._shared import daily_rank_ic
from app.alpha_mining.labels import attach_alpha_labels
from app.alpha_mining.policy import (
    ALPHA_ALGORITHM_VERSION,
    evaluate_historical_gates,
)
from app.alpha_mining.registry import load_builtin_registry
from app.backtest.factor import FACTOR_COLUMNS, FactorBacktestService, FactorBatchConfig
from app.backtest.mining import NestedValidationConfig, generate_nested_folds
from app.backtest.mining_runtime import (
    MatcherCandidateEvaluator,
    _load_compact_factor_panel,
    _prepare_base_market,
)
from app.backtest.strategy import BacktestResultPolicy, StrategyBacktestService
from app.services.mining_preflight import enriched_partition_dates
from app.strategy.engine import StrategyEngine

ProgressCallback = Any
CancelCheck = Any
_FACTOR_IDS = frozenset(str(item["id"]) for item in FACTOR_COLUMNS)
_PROFILES = frozenset({"exploratory", "balanced", "strict"})
_HORIZONS = frozenset({1, 3, 5, 10, 20, 60})


class AlphaMiningCancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlphaRuntimeRequest:
    run_id: str
    engine_ids: tuple[str, ...]
    factor_names: tuple[str, ...]
    strategy_ids: tuple[str, ...]
    champion_strategy_id: str
    symbols: list[str] | None
    asset_type: Literal["stock", "etf"]
    start: date
    end: date
    profile: Literal["exploratory", "balanced", "strict"]
    forward_horizon: int
    commission_pct: float
    stamp_tax_pct: float
    slippage_bps: float
    max_positions: int
    max_candidates_per_engine: int
    max_trials_per_engine: int


def run_alpha_mining_runtime(
    payload: Mapping[str, Any],
    *,
    data_dir: Path,
    service: StrategyBacktestService,
    strategy_engine: StrategyEngine,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    rss_sampler: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    emit = progress_cb or (lambda _message: None)
    request = _decode_request(payload, data_dir, strategy_engine)
    fingerprint = payload.get("data_fingerprint")
    expected_generation = fingerprint.get("generation") if isinstance(fingerprint, Mapping) else None
    if not isinstance(expected_generation, str) or not expected_generation:
        raise ValueError("Alpha worker payload is missing its data generation")

    registry, load_failures = load_builtin_registry()
    for engine_id in request.engine_ids:
        registry.get(engine_id)

    phase_ms: dict[str, float] = {}
    _raise_if_cancelled(cancel_check)
    emit({"phase": "panel", "label": "准备点时研究面板", "done": 0, "total": 1})
    panel_started = time.perf_counter()
    factor_service = FactorBacktestService(service.engine)
    factor_config = FactorBatchConfig(
        factor_names=list(request.factor_names),
        symbols=request.symbols,
        start=request.start,
        end=request.end,
        asset_type=request.asset_type,
        commission_pct=request.commission_pct,
        stamp_tax_pct=request.stamp_tax_pct,
        slippage_bps=request.slippage_bps,
    )
    generation = factor_service._data_generation(request.asset_type)
    if generation != expected_generation:
        raise ValueError("Alpha data generation changed after the run was queued")
    source_panel = _load_compact_factor_panel(
        factor_service,
        factor_config,
        request.factor_names,
        expected_generation=generation,
        cancel_check=cancel_check,
    )
    if source_panel.is_empty():
        raise ValueError("Alpha date range contains no enriched data")
    trading_dates = enriched_partition_dates(
        data_dir,
        request.asset_type,
        request.start,
        request.end,
    )
    catalog = AlphaResearchDataCatalog(data_dir)
    catalog_snapshot = catalog.snapshot(request.start, request.end, request.asset_type)
    universe_qualification = catalog_snapshot.datasets["historical_universe"]
    if not universe_qualification.ready:
        raise ValueError("正式Alpha研究数据门禁失败: " + "; ".join(universe_qualification.reasons))
    source_panel, universe_audit = catalog.apply_formal_pit_context(source_panel)
    requested_manifests = [registry.get(engine_id).manifest for engine_id in request.engine_ids]
    required_dataset_ids = {
        requirement.dataset_id
        for manifest in requested_manifests
        for requirement in manifest.required_datasets
    }
    context_audits: dict[str, Any] = {"historical_universe": universe_audit}
    if "financial_pit" in required_dataset_ids and catalog_snapshot.datasets["financial_pit"].ready:
        source_panel, context_audits["financial_pit"] = catalog.attach_financial_context(
            source_panel
        )
    if "industry_pit" in required_dataset_ids and catalog_snapshot.datasets["industry_pit"].ready:
        source_panel, context_audits["industry_pit"] = catalog.attach_industry_context(source_panel)
    if "event_history" in required_dataset_ids and catalog_snapshot.datasets["event_history"].ready:
        source_panel, context_audits["event_history"] = catalog.attach_event_context(
            source_panel,
            start=request.start,
            end=request.end,
        )
    for dataset_id in ("financial_pit", "industry_pit", "event_history"):
        if dataset_id not in context_audits:
            continue
        required_coverage = max(
            requirement.minimum_coverage
            for manifest in requested_manifests
            for requirement in manifest.required_datasets
            if requirement.dataset_id == dataset_id
        )
        actual_coverage = float(context_audits[dataset_id].get("coverage") or 0.0)
        if actual_coverage < required_coverage:
            raise ValueError(
                f"正式Alpha研究数据门禁失败: {dataset_id}覆盖"
                f"{actual_coverage:.4f}<{required_coverage:.4f}"
            )
    derived_features = tuple(
        name
        for name in source_panel.columns
        if name.startswith((
            "event_",
            "industry_",
            "network_",
            "financial_revision_",
            "roe_latest",
            "gross_margin_latest",
            "net_margin_latest",
            "revenue_yoy_latest",
            "net_income_yoy_latest",
            "debt_ratio_latest",
        ))
        and source_panel.schema[name].is_numeric()
    )
    feature_names = tuple(dict.fromkeys((*request.factor_names, *derived_features)))
    target_column = f"target_return_{request.forward_horizon}d"
    panel = attach_alpha_labels(
        source_panel,
        trading_dates,
        commission_pct=request.commission_pct,
        stamp_tax_pct=request.stamp_tax_pct,
        slippage_bps=request.slippage_bps,
    ).with_columns(
        pl.col(f"_target_date_{request.forward_horizon}d").alias("_target_date"),
        pl.col(f"target_residual_return_{request.forward_horizon}d").alias(
            "target_residual_return"
        ),
    )
    del source_panel
    service.engine.clear_panel_cache()
    phase_ms["panel"] = round((time.perf_counter() - panel_started) * 1000, 1)
    emit({
        "phase": "panel",
        "label": "点时研究面板已就绪",
        "done": 1,
        "total": 1,
        "rows": panel.height,
        "dates": len(trading_dates),
    })

    validation = _validation_config(request.profile, request.forward_horizon)
    nested_folds = generate_nested_folds(
        [value.isoformat() for value in trading_dates],
        validation,
    )
    if not nested_folds:
        raise ValueError("Alpha date range cannot form an outer validation fold")

    _raise_if_cancelled(cancel_check)
    emit({"phase": "matrix", "label": "准备统一撮合矩阵", "done": 0, "total": 1})
    matrix_started = time.perf_counter()
    base_market = _prepare_base_market(
        service,
        strategy_engine,
        data_dir,
        request,
        expected_generation=generation,
        cancel_check=cancel_check,
    )
    base_market = _augment_market_fields(base_market, panel, derived_features)
    phase_ms["matrix"] = round((time.perf_counter() - matrix_started) * 1000, 1)
    emit({
        "phase": "matrix",
        "label": "统一撮合矩阵已就绪",
        "done": 1,
        "total": 1,
        "matrix_bytes": base_market.nbytes,
    })

    evaluator = MatcherCandidateEvaluator(
        service,
        strategy_engine,
        data_dir,
        request,
        base_market,
        cancel_check,
        result_policy=BacktestResultPolicy(
            required_stats=frozenset({"total_return", "sharpe", "max_drawdown", "n_trades"}),
            include_monte_carlo=False,
            include_curves=True,
            include_trades=True,
            include_per_symbol_stats=True,
            include_return_distribution=False,
            include_benchmark=False,
            include_strategy_info=False,
        ),
    )
    budget = TrialBudget(
        max_candidates=request.max_candidates_per_engine,
        max_trials=request.max_trials_per_engine,
        min_cross_section=20,
        min_dates=40 if request.profile == "exploratory" else 60,
    )
    engine_folds: dict[str, list[dict[str, Any]]] = {
        engine_id: [] for engine_id in request.engine_ids
    }
    benchmark_folds: list[dict[str, Any]] = []
    discovery_failures: list[dict[str, str]] = list(load_failures)
    trial_ledger: list[dict[str, Any]] = []

    search_started = time.perf_counter()
    total_steps = len(nested_folds)
    emit({"phase": "validation", "label": "滚动发现与外层样本外验证", "done": 0, "total": total_steps})
    for fold_number, nested in enumerate(nested_folds, start=1):
        _raise_if_cancelled(cancel_check)
        outer = nested.outer
        discover_labels, selection_labels = _discovery_selection_split(
            outer.train_labels,
            horizon=request.forward_horizon,
            profile=request.profile,
        )
        discover = _training_frame(
            panel,
            discover_labels,
            target_column,
            label_end=discover_labels[-1],
        )
        selection = _frame_for_labels(panel, selection_labels)
        context = TrainOnlyContext(
            frame=discover,
            date_labels=discover_labels,
            feature_names=feature_names,
            target_column=target_column,
            asset_type=request.asset_type,
            metadata={
                "outer_index": outer.outer_index,
                "selection_dates_hidden": len(selection_labels),
                "outer_test_dates_hidden": len(outer.test_labels),
            },
        )
        benchmark_eval = evaluator.evaluate_candidate_labels(
            outer.train_labels,
            outer.test_labels,
            {"kind": "existing_strategy", "strategy_id": request.champion_strategy_id},
        )
        benchmark_folds.append(_evaluation_row(outer, None, benchmark_eval, "champion"))

        for engine_id in request.engine_ids:
            engine = registry.get(engine_id)
            engine_trials: list[dict[str, Any]] = []
            engine_context = replace(
                context,
                metadata={**context.metadata, "trial_audit": engine_trials},
            )
            try:
                qualification = engine.preflight(DataCatalogContext(
                    asset_type=request.asset_type,
                    start=discover_labels[0],
                    end=discover_labels[-1],
                    available_features=feature_names,
                    datasets=catalog_snapshot.datasets,
                    observations={"rows": discover.height},
                ))
                manifest_qualification = qualify_manifest_datasets(
                    engine.manifest,
                    catalog_snapshot.datasets,
                )
                if request.forward_horizon not in engine.manifest.forecast_horizons:
                    raise ValueError(
                        f"引擎不支持{request.forward_horizon}日预测期限"
                    )
                if not manifest_qualification.ready:
                    raise ValueError("; ".join(manifest_qualification.reasons))
                if not qualification.ready:
                    raise ValueError("; ".join(qualification.reasons))
                candidates = engine.discover(engine_context, budget)[: budget.max_candidates]
                trial_ledger.extend({
                    "outer_index": outer.outer_index,
                    "engine_id": engine_id,
                    "stage": "discovery",
                    **row,
                } for row in engine_trials)
                engine_trials.clear()
                selected, selection_evidence = _select_on_hidden_inner_window(
                    candidates,
                    selection,
                    target_column,
                    budget,
                )
                trial_ledger.extend(
                    {
                        "outer_index": outer.outer_index,
                        "engine_id": engine_id,
                        **row,
                    }
                    for row in selection_evidence
                )
                if selected is None:
                    engine_folds[engine_id].append({
                        "outer_index": outer.outer_index,
                        "train_start": outer.train_start,
                        "train_end": outer.train_end,
                        "test_start": outer.test_start,
                        "test_end": outer.test_end,
                        "error": "inner selection produced no finite candidate",
                    })
                    continue
                frozen = engine.materialize(selected, engine_context)
                evaluation = evaluator.evaluate_candidate_labels(
                    outer.train_labels,
                    outer.test_labels,
                    frozen.definition,
                )
                engine_folds[engine_id].append(
                    _evaluation_row(outer, selected, evaluation, "candidate")
                )
            except Exception as exc:
                trial_ledger.extend({
                    "outer_index": outer.outer_index,
                    "engine_id": engine_id,
                    "stage": "discovery",
                    **row,
                } for row in engine_trials)
                discovery_failures.append({
                    "engine_id": engine_id,
                    "stage": f"outer_{outer.outer_index}",
                    "error": str(exc)[:500],
                })
                engine_folds[engine_id].append({
                    "outer_index": outer.outer_index,
                    "train_start": outer.train_start,
                    "train_end": outer.train_end,
                    "test_start": outer.test_start,
                    "test_end": outer.test_end,
                    "error": str(exc)[:500],
                })
        emit({
            "phase": "validation",
            "label": f"完成外层窗口 {fold_number}/{total_steps}",
            "done": fold_number,
            "total": total_steps,
        })
    phase_ms["validation"] = round((time.perf_counter() - search_started) * 1000, 1)

    champion_metrics = _aggregate_fold_metrics(benchmark_folds)
    candidates_summary: list[dict[str, Any]] = []
    stress_backtest_count = 0
    for engine_id, folds in engine_folds.items():
        metrics = _aggregate_fold_metrics(folds, champion_folds=benchmark_folds)
        stress_metrics, stress_count = _run_stress_suite(
            folds=folds,
            request=request,
            service=service,
            strategy_engine=strategy_engine,
            data_dir=data_dir,
            base_market=base_market,
            panel=panel,
            cancel_check=cancel_check,
        )
        metrics.update(stress_metrics)
        stress_backtest_count += stress_count
        gates = evaluate_historical_gates(metrics, champion_metrics)
        historical = [gate for gate in gates if gate["id"] != "forward_shadow"]
        failed = any(gate["status"] == "failed" for gate in historical)
        complete = bool(historical) and all(gate["status"] == "passed" for gate in historical)
        state = "rejected" if failed else ("research_candidate" if complete else "outer_evaluated")
        frozen_candidate = _representative_candidate(folds)
        candidates_summary.append({
            "engine_id": engine_id,
            "engine_name": registry.get(engine_id).manifest.name,
            "state": state,
            "metrics": metrics,
            "gates": gates,
            "folds": folds,
            "frozen_candidate": frozen_candidate,
        })
    candidates_summary.sort(
        key=lambda item: _sortable_return(item.get("metrics", {}).get("stitched_oos_return")),
        reverse=True,
    )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    phase_ms["total"] = elapsed_ms
    return {
        "status": "succeeded",
        "research_state": "outer_evaluated",
        "algorithm_version": ALPHA_ALGORITHM_VERSION,
        "data_as_of": request.end.isoformat(),
        "request_summary": {
            "asset_type": request.asset_type,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "profile": request.profile,
            "forward_horizon": request.forward_horizon,
            "engine_ids": list(request.engine_ids),
            "factor_count": len(request.factor_names),
            "derived_feature_count": len(derived_features),
            "champion_strategy_id": request.champion_strategy_id,
            "commission_pct": request.commission_pct,
            "stamp_tax_pct": request.stamp_tax_pct,
            "slippage_bps": request.slippage_bps,
        },
        "summary": {
            "outer_fold_count": len(nested_folds),
            "candidate_engine_count": len(candidates_summary),
            "trial_count": len(trial_ledger),
            "backtest_count": evaluator.backtest_count + stress_backtest_count,
            "stress_backtest_count": stress_backtest_count,
            "panel_rows": panel.height,
            "matrix_bytes": base_market.nbytes,
            "elapsed_ms": elapsed_ms,
            "phase_ms": phase_ms,
        },
        "data_catalog": catalog_snapshot.to_dict(),
        "data_context_audits": context_audits,
        "champion": {
            "strategy_id": request.champion_strategy_id,
            "metrics": champion_metrics,
            "folds": benchmark_folds,
        },
        "candidates": candidates_summary,
        "trial_ledger": trial_ledger,
        "engine_failures": discovery_failures,
        "worker_phase_peak_rss_bytes": (
            rss_sampler.phase_peak_rss_bytes() if rss_sampler is not None else None
        ),
    }


def _decode_request(
    payload: Mapping[str, Any],
    data_dir: Path,
    strategy_engine: StrategyEngine,
) -> AlphaRuntimeRequest:
    run_id = str(payload.get("run_id") or "")
    values = payload.get("request")
    if not run_id or not isinstance(values, Mapping):
        raise ValueError("Alpha worker payload is missing run_id or request")
    registry, _failures = load_builtin_registry()
    engine_ids = tuple(str(value) for value in values.get("engine_ids") or ())
    if not engine_ids or len(set(engine_ids)) != len(engine_ids):
        raise ValueError("engine_ids must be non-empty and unique")
    for engine_id in engine_ids:
        registry.get(engine_id)
    factor_names = tuple(str(value) for value in values.get("factor_names") or ())
    if not factor_names or len(set(factor_names)) != len(factor_names):
        raise ValueError("factor_names must be non-empty and unique")
    unknown = sorted(set(factor_names) - _FACTOR_IDS)
    if unknown:
        raise ValueError(f"unknown Alpha factors: {unknown}")
    asset_type = str(values.get("asset_type") or "stock")
    if asset_type not in {"stock", "etf"}:
        raise ValueError("Alpha asset_type must be stock or etf")
    all_dates = enriched_partition_dates(data_dir, asset_type)
    if not all_dates:
        raise ValueError(f"no enriched {asset_type} dates are available")
    requested_start = _optional_date(values.get("start"))
    requested_end = _optional_date(values.get("end"))
    start = max(requested_start or all_dates[0], all_dates[0])
    end = min(requested_end or all_dates[-1], all_dates[-1])
    if start > end:
        raise ValueError("Alpha date range contains no enriched data")
    profile = str(values.get("budget_profile") or "balanced")
    if profile not in _PROFILES:
        raise ValueError(f"unsupported Alpha profile: {profile}")
    horizon = int(values.get("forward_horizon") or 5)
    if horizon not in _HORIZONS:
        raise ValueError(f"forward_horizon must be one of {sorted(_HORIZONS)}")
    champion = str(values.get("champion_strategy_id") or "n_day_low_reversal")
    champion_strategy = strategy_engine.get(champion)
    if champion_strategy.meta.get("research_only"):
        raise ValueError("Alpha champion cannot be a research-only strategy")
    symbols_value = values.get("symbols")
    symbols = None if symbols_value is None else list(dict.fromkeys(str(v) for v in symbols_value if v))
    return AlphaRuntimeRequest(
        run_id=run_id,
        engine_ids=engine_ids,
        factor_names=factor_names,
        strategy_ids=(champion,),
        champion_strategy_id=champion,
        symbols=symbols or None,
        asset_type=asset_type,  # type: ignore[arg-type]
        start=start,
        end=end,
        profile=profile,  # type: ignore[arg-type]
        forward_horizon=horizon,
        commission_pct=_bounded_float(values.get("commission_pct", 0.0002), "commission_pct", 0, 0.05),
        stamp_tax_pct=_bounded_float(values.get("stamp_tax_pct", 0.0005), "stamp_tax_pct", 0, 0.05),
        slippage_bps=_bounded_float(values.get("slippage_bps", 5.0), "slippage_bps", 0, 1000),
        max_positions=_bounded_int(values.get("max_positions", 10), "max_positions", 1, 50),
        max_candidates_per_engine=_bounded_int(
            values.get("max_candidates_per_engine", 4), "max_candidates_per_engine", 1, 12
        ),
        max_trials_per_engine=_bounded_int(
            values.get("max_trials_per_engine", 64), "max_trials_per_engine", 4, 256
        ),
    )


def _validation_config(profile: str, horizon: int) -> NestedValidationConfig:
    purge = max(30, horizon)
    if profile == "exploratory":
        return NestedValidationConfig(
            outer_train_bars=126,
            outer_test_bars=63,
            outer_step_bars=63,
            inner_train_bars=63,
            inner_test_bars=21,
            inner_step_bars=21,
            purge_bars=purge,
            embargo_bars=5,
            min_train_bars=63,
        )
    if profile == "strict":
        return NestedValidationConfig(
            outer_train_bars=756,
            outer_test_bars=126,
            outer_step_bars=126,
            inner_train_bars=504,
            inner_test_bars=63,
            inner_step_bars=63,
            purge_bars=purge,
            embargo_bars=5,
            min_train_bars=126,
        )
    return NestedValidationConfig(
        outer_train_bars=504,
        outer_test_bars=126,
        outer_step_bars=126,
        inner_train_bars=252,
        inner_test_bars=63,
        inner_step_bars=63,
        purge_bars=purge,
        embargo_bars=5,
        min_train_bars=126,
    )


def _discovery_selection_split(
    labels: Sequence[str],
    *,
    horizon: int,
    profile: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selection_bars = 42 if profile == "exploratory" else 63
    gap = max(horizon, 5)
    if len(labels) <= selection_bars + gap + 60:
        raise ValueError("outer training window is too short for discovery/selection isolation")
    return tuple(labels[: -selection_bars - gap]), tuple(labels[-selection_bars:])


def _training_frame(
    panel: pl.DataFrame,
    labels: Sequence[str],
    target_column: str,
    *,
    label_end: str,
) -> pl.DataFrame:
    return _frame_for_labels(panel, labels).filter(
        pl.col(target_column).is_not_null()
        & pl.col("_target_date").is_not_null()
        & (pl.col("_target_date") <= date.fromisoformat(label_end))
    )


def _frame_for_labels(panel: pl.DataFrame, labels: Sequence[str]) -> pl.DataFrame:
    label_dates = [date.fromisoformat(value) for value in labels]
    return panel.filter(pl.col("date").is_in(label_dates))


def _select_on_hidden_inner_window(
    candidates: Sequence[CandidateSpec],
    selection: pl.DataFrame,
    target_column: str,
    budget: TrialBudget,
) -> tuple[CandidateSpec | None, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    selected: CandidateSpec | None = None
    best_score = -math.inf
    trial_count = max(len(candidates), 1)
    penalty = math.sqrt(2.0 * math.log(trial_count + 1) / max(selection["date"].n_unique(), 1))
    for candidate in candidates:
        scored = _candidate_score_frame(selection, candidate, target_column)
        metric = daily_rank_ic(
            scored,
            "_candidate_score",
            target_column,
            min_cross_section=budget.min_cross_section,
        )
        raw_score = None if metric is None else float(metric["ic_mean"])
        adjusted = None if raw_score is None else raw_score - penalty
        evidence.append({
            "recipe_id": candidate.recipe_id,
            "raw_selection_ic": raw_score,
            "multiple_test_penalty": penalty,
            "penalized_score": adjusted,
        })
        if adjusted is not None and adjusted > best_score:
            best_score = adjusted
            selected = candidate
    return selected, evidence


def _candidate_score_frame(
    frame: pl.DataFrame,
    candidate: CandidateSpec,
    target_column: str,
) -> pl.DataFrame:
    total = sum(abs(float(value)) for value in candidate.weights) or 1.0
    expressions = []
    for feature, direction, weight in zip(
        candidate.features,
        candidate.directions,
        candidate.weights,
        strict=True,
    ):
        percentile = pl.col(feature).rank(method="average").over("date") / pl.len().over("date")
        directed = percentile if direction > 0 else 1.0 - percentile
        expressions.append(directed * (abs(float(weight)) / total))
    return frame.select(
        "date",
        target_column,
        pl.sum_horizontal(expressions).alias("_candidate_score"),
    ).drop_nulls()


def _evaluation_row(outer, candidate, evaluation, kind: str) -> dict[str, Any]:
    return {
        "outer_index": outer.outer_index,
        "train_start": outer.train_start,
        "train_end": outer.train_end,
        "test_start": outer.test_start,
        "test_end": outer.test_end,
        "kind": kind,
        "recipe_id": candidate.recipe_id if candidate is not None else None,
        "candidate": candidate.to_dict() if candidate is not None else None,
        "score": evaluation.score,
        "metrics": evaluation.metrics,
        "error": evaluation.error,
    }


def _aggregate_fold_metrics(
    folds: Sequence[Mapping[str, Any]],
    *,
    champion_folds: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    returns = _stitch_fold_returns(folds)
    champion_returns = _stitch_fold_returns(champion_folds or ())
    if not returns:
        return {
            "stitched_oos_return": None,
            "stitched_oos_sharpe": None,
            "max_drawdown": None,
            "positive_half_year_ratio": None,
            "beat_champion_half_year_ratio": None,
            "recent_1y_return": None,
            "recent_3m_return": None,
            "oos_days": 0,
        }
    values = [value for _, value in returns]
    total_return = _compound(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    sharpe = mean / math.sqrt(variance) * math.sqrt(252) if variance > 0 else None
    max_drawdown = _max_drawdown(values)
    half_years = _period_returns(returns, "half_year")
    champion_half_years = _period_returns(champion_returns, "half_year")
    positive_ratio = sum(value > 0 for value in half_years.values()) / len(half_years) if half_years else None
    comparable = sorted(set(half_years) & set(champion_half_years))
    beat_ratio = (
        sum(half_years[key] > champion_half_years[key] for key in comparable) / len(comparable)
        if comparable else None
    )
    end = returns[-1][0]
    return {
        "stitched_oos_return": total_return,
        "stitched_oos_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "positive_half_year_ratio": positive_ratio,
        "beat_champion_half_year_ratio": beat_ratio,
        "recent_1y_return": _compound([value for day, value in returns if day > end - timedelta(days=365)]),
        "recent_3m_return": _compound([value for day, value in returns if day > end - timedelta(days=92)]),
        "half_year_windows": len(half_years),
        "oos_days": len(returns),
        "equity_curve": _equity_curve(returns),
    }


def _stitch_fold_returns(folds: Sequence[Mapping[str, Any]]) -> list[tuple[date, float]]:
    by_date: dict[date, float] = {}
    for fold in folds:
        metrics = fold.get("metrics")
        curve = metrics.get("equity_curve") if isinstance(metrics, Mapping) else None
        if not isinstance(curve, list):
            continue
        previous: float | None = None
        for row in curve:
            if not isinstance(row, Mapping):
                continue
            try:
                day = date.fromisoformat(str(row.get("date"))[:10])
                value = float(row.get("value"))
            except (TypeError, ValueError):
                continue
            if previous is not None and previous > 0 and day not in by_date:
                daily_return = value / previous - 1.0
                if math.isfinite(daily_return):
                    by_date[day] = daily_return
            previous = value
    return sorted(by_date.items())


def _period_returns(returns: Sequence[tuple[date, float]], kind: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for day, value in returns:
        key = f"{day.year}-H{1 if day.month <= 6 else 2}" if kind == "half_year" else str(day.year)
        grouped.setdefault(key, []).append(value)
    return {key: _compound(values) for key, values in grouped.items()}


def _compound(values: Sequence[float]) -> float:
    wealth = 1.0
    for value in values:
        wealth *= 1.0 + value
    return wealth - 1.0


def _max_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        worst = min(worst, wealth / peak - 1.0)
    return worst


def _equity_curve(returns: Sequence[tuple[date, float]]) -> list[dict[str, Any]]:
    wealth = 1.0
    output = []
    for day, value in returns:
        wealth *= 1.0 + value
        output.append({"date": day.isoformat(), "value": wealth})
    return output


def _sortable_return(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return result if math.isfinite(result) else -math.inf


def _representative_candidate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [row.get("candidate") for row in folds if isinstance(row.get("candidate"), Mapping)]
    if not rows:
        return None
    counts: dict[str, int] = {}
    for row in rows:
        recipe_id = str(row.get("recipe_id") or "")
        counts[recipe_id] = counts.get(recipe_id, 0) + 1
    winner = sorted(counts, key=lambda key: (-counts[key], key))[0]
    return next(dict(row) for row in reversed(rows) if str(row.get("recipe_id") or "") == winner)


def _run_stress_suite(
    *,
    folds: Sequence[Mapping[str, Any]],
    request: AlphaRuntimeRequest,
    service: StrategyBacktestService,
    strategy_engine: StrategyEngine,
    data_dir: Path,
    base_market,
    panel: pl.DataFrame,
    cancel_check,
) -> tuple[dict[str, Any], int]:
    evaluable = [row for row in folds if isinstance(row.get("candidate"), Mapping)]
    if not evaluable:
        return {
            "double_cost_return": None,
            "delayed_entry_return": None,
            "worst_parameter_return": None,
            "capacity_return": None,
            "capacity_passed": None,
            "concentration_passed": None,
            "stress": {"status": "not_evaluable"},
        }, 0
    policy = BacktestResultPolicy(
        required_stats=frozenset({"total_return", "sharpe", "max_drawdown", "n_trades"}),
        include_monte_carlo=False,
        include_curves=True,
        include_trades=False,
        include_per_symbol_stats=False,
        include_return_distribution=False,
        include_benchmark=False,
        include_strategy_info=False,
    )

    def evaluator_for(scenario_request: AlphaRuntimeRequest) -> MatcherCandidateEvaluator:
        return MatcherCandidateEvaluator(
            service,
            strategy_engine,
            data_dir,
            scenario_request,
            base_market,
            cancel_check,
            result_policy=policy,
        )

    double = evaluator_for(replace(
        request,
        commission_pct=request.commission_pct * 2.0,
        stamp_tax_pct=request.stamp_tax_pct * 2.0,
        slippage_bps=request.slippage_bps * 2.0,
    ))
    delayed = evaluator_for(request)
    capacity = evaluator_for(replace(request, max_positions=min(request.max_positions * 2, 50)))
    parameter_evaluators = [evaluator_for(request) for _ in range(3)]
    scenario_folds: dict[str, list[dict[str, Any]]] = {
        "double_cost": [],
        "delay": [],
        "capacity": [],
        "parameter_low": [],
        "parameter_high": [],
        "parameter_narrow": [],
    }
    for row in evaluable:
        definition = _candidate_definition(row["candidate"])
        labels = tuple(_labels_between(base_market.timestamp_labels, row["test_start"], row["test_end"]))
        outer = _FoldView(row)
        for name, evaluator, candidate_definition in (
            ("double_cost", double, definition),
            ("delay", delayed, _perturb_definition(definition, delay=1)),
            ("capacity", capacity, definition),
            ("parameter_low", parameter_evaluators[0], _perturb_definition(definition, entry_delta=-5, rank_ratio=1.25)),
            ("parameter_high", parameter_evaluators[1], _perturb_definition(definition, entry_delta=5, rank_ratio=0.75)),
            ("parameter_narrow", parameter_evaluators[2], _perturb_definition(definition, exit_delta=5)),
        ):
            evaluation = evaluator.evaluate_candidate_labels(labels, labels, candidate_definition)
            scenario_folds[name].append(_evaluation_row(outer, None, evaluation, name))
    aggregates = {name: _aggregate_fold_metrics(rows) for name, rows in scenario_folds.items()}
    parameter_returns = [
        aggregates[name].get("stitched_oos_return")
        for name in ("parameter_low", "parameter_high", "parameter_narrow")
    ]
    finite_parameter_returns = [float(value) for value in parameter_returns if _is_finite(value)]
    concentration = _concentration_evidence(evaluable, panel)
    count = sum(
        evaluator.backtest_count
        for evaluator in (double, delayed, capacity, *parameter_evaluators)
    )
    capacity_return = aggregates["capacity"].get("stitched_oos_return")
    return {
        "double_cost_return": aggregates["double_cost"].get("stitched_oos_return"),
        "delayed_entry_return": aggregates["delay"].get("stitched_oos_return"),
        "worst_parameter_return": min(finite_parameter_returns) if finite_parameter_returns else None,
        "capacity_return": capacity_return,
        "capacity_passed": bool(_is_finite(capacity_return) and float(capacity_return) > 0),
        "concentration_passed": concentration["passed"],
        "stress": {
            "double_cost": aggregates["double_cost"],
            "delay": aggregates["delay"],
            "capacity": aggregates["capacity"],
            "parameter_returns": parameter_returns,
            "concentration": concentration,
        },
    }, count


class _FoldView:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.outer_index = row["outer_index"]
        self.train_start = row["train_start"]
        self.train_end = row["train_end"]
        self.test_start = row["test_start"]
        self.test_end = row["test_end"]


def _labels_between(labels: Sequence[str], start: str, end: str) -> list[str]:
    return [label[:10] for label in labels if start <= label[:10] <= end]


def _candidate_definition(candidate: Mapping[str, Any]) -> dict[str, Any]:
    features = [str(value) for value in candidate.get("features") or []]
    directions = [int(value) for value in candidate.get("directions") or []]
    weights = [float(value) for value in candidate.get("weights") or []]
    total = sum(abs(value) for value in weights) or 1.0
    return {
        "kind": "factor_rank",
        "scoring": {
            feature: abs(weight) / total
            for feature, weight in zip(features, weights, strict=True)
        },
        "directions": {
            feature: "high" if direction > 0 else "low"
            for feature, direction in zip(features, directions, strict=True)
        },
        "parameters": dict(candidate.get("parameters") or {}),
    }


def _perturb_definition(
    definition: Mapping[str, Any],
    *,
    entry_delta: float = 0.0,
    exit_delta: float = 0.0,
    rank_ratio: float = 1.0,
    delay: int = 0,
) -> dict[str, Any]:
    output = {**definition, "parameters": dict(definition.get("parameters") or {})}
    parameters = output["parameters"]
    parameters["entry_score"] = min(100.0, max(0.0, float(parameters.get("entry_score", 70.0)) + entry_delta))
    parameters["exit_score"] = min(100.0, max(0.0, float(parameters.get("exit_score", 40.0)) + exit_delta))
    parameters["top_rank"] = min(100, max(1, round(int(parameters.get("top_rank", 20)) * rank_ratio)))
    parameters["entry_delay_days"] = delay
    if parameters["exit_score"] > parameters["entry_score"]:
        parameters["exit_score"] = parameters["entry_score"]
    return output


def _concentration_evidence(
    folds: Sequence[Mapping[str, Any]],
    panel: pl.DataFrame,
) -> dict[str, Any]:
    trades = [
        trade
        for fold in folds
        for trade in ((fold.get("metrics") or {}).get("trades") or [])
        if isinstance(trade, Mapping)
    ]
    positive = [trade for trade in trades if float(trade.get("pnl_amount") or 0.0) > 0]
    total = sum(float(trade.get("pnl_amount") or 0.0) for trade in positive)
    if total <= 0:
        return {"passed": False, "reason": "no_positive_pnl_contribution", "trades": len(trades)}

    def share_by(key):
        grouped: dict[str, float] = {}
        for trade in positive:
            label = str(key(trade) or "unknown")
            grouped[label] = grouped.get(label, 0.0) + float(trade.get("pnl_amount") or 0.0)
        shares = sorted((value / total for value in grouped.values()), reverse=True)
        return (shares[0] if shares else 1.0), sum(shares[:5])

    symbol_top, symbol_top5 = share_by(lambda trade: trade.get("symbol"))
    year_top, _ = share_by(lambda trade: str(trade.get("exit_date") or "")[:4])
    industry_top = None
    if "l1_code" in panel.columns:
        lookup = {
            (str(row["symbol"]), str(row["date"])[:10]): str(row["l1_code"] or "unknown")
            for row in panel.select("symbol", "date", "l1_code").iter_rows(named=True)
        }
        industry_top, _ = share_by(
            lambda trade: lookup.get((str(trade.get("symbol")), str(trade.get("entry_date"))[:10]))
        )
    passed = (
        symbol_top <= 0.20
        and symbol_top5 <= 0.50
        and year_top <= 0.40
        and industry_top is not None
        and industry_top <= 0.35
    )
    return {
        "passed": passed,
        "trades": len(trades),
        "positive_contribution": total,
        "top_symbol_share": symbol_top,
        "top5_symbol_share": symbol_top5,
        "top_year_share": year_top,
        "top_industry_share": industry_top,
        "limits": {"symbol": 0.20, "top5": 0.50, "year": 0.40, "industry": 0.35},
    }


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _augment_market_fields(base_market, panel: pl.DataFrame, feature_names: Sequence[str]):
    """Inject standardized provider-derived numeric features into the immutable matrix."""
    names = [name for name in feature_names if name in panel.columns]
    if not names:
        return base_market
    time_index = {label[:10]: index for index, label in enumerate(base_market.timestamp_labels)}
    asset_index = {symbol: index for index, symbol in enumerate(base_market.symbols)}
    fields = dict(base_market.fields)
    scoped = panel.select("date", "symbol", *names).with_columns(
        pl.col("date").cast(pl.Date)
    )
    for name in names:
        values = np.full(base_market.shape, np.nan, dtype=np.float32)
        for row in scoped.select("date", "symbol", name).drop_nulls(name).iter_rows(named=True):
            time_id = time_index.get(str(row["date"])[:10])
            asset_id = asset_index.get(str(row["symbol"]))
            if time_id is not None and asset_id is not None:
                values[time_id, asset_id] = float(row[name])
        values.flags.writeable = False
        fields[name] = values
    return replace(base_market, fields=fields)


def _optional_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result != float(value) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return result


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is None:
        return
    cancelled = cancel_check() if callable(cancel_check) else cancel_check.is_set()
    if cancelled:
        raise AlphaMiningCancelledError("Alpha mining cancelled")
