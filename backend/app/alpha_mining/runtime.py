"""Spawn-only Alpha discovery runtime with train-only engines and central OOS execution."""
# Requirements: AM-S5-001 through AM-S6-012.
# ruff: noqa: RUF001
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
_SHARE_HISTORY_FACTOR_IDS = frozenset(
    {"turnover_rate", "turnover_ratio_5d", "turnover_z_60d"}
)
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
    champion_strategy_id: str | None
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
    source_run_id: str | None = None
    source_suggestion_id: str | None = None
    source_candidate_id: str | None = None
    source_diff: Mapping[str, Any] | None = None
    hypothesis_id: str | None = None
    hypothesis_contract: Mapping[str, Any] | None = None


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
        preserve_market_columns=True,
    )
    if source_panel.is_empty():
        raise ValueError("Alpha date range contains no enriched data")
    # Market attribution is descriptive evidence calculated after the OOS run.
    # Keep it on the same adjusted-close panel and derive each stock's
    # contemporaneous daily return here; never substitute a present-day market
    # snapshot or a future label for historical breadth/regime attribution.
    source_panel = source_panel.with_columns(
        (
            pl.col("close")
            / pl.col("close").shift(1).over("symbol")
            - 1.0
        ).cast(pl.Float32).alias("change_pct")
    )
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
    require_share_history = bool(
        set(request.factor_names) & _SHARE_HISTORY_FACTOR_IDS
    )
    share_qualification = catalog_snapshot.datasets["share_history_pit"]
    if require_share_history and not share_qualification.ready:
        raise ValueError(
            "所选换手率因子缺少点时股本: " + "; ".join(share_qualification.reasons)
        )
    source_panel, universe_audit = catalog.apply_formal_pit_context(
        source_panel,
        require_share_history=require_share_history,
    )
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
    if catalog_snapshot.datasets["industry_pit"].ready:
        source_panel, context_audits["industry_pit"] = catalog.attach_industry_context(source_panel)
    if "event_history" in required_dataset_ids and catalog_snapshot.datasets["event_history"].ready:
        source_panel, context_audits["event_history"] = catalog.attach_event_context(
            source_panel,
            start=request.start,
            end=request.end,
        )
    for dataset_id in required_dataset_ids & {"financial_pit", "industry_pit", "event_history"}:
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
        horizons=(request.forward_horizon,),
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
    engine_progress = {
        engine_id: {
            "engine_id": engine_id,
            "status": "waiting",
            "folds_done": 0,
            "folds_total": len(nested_folds),
            "trials": 0,
            "selected": 0,
            "backtests": 0,
            "errors": 0,
            "message": None,
        }
        for engine_id in request.engine_ids
    }
    total_steps = len(nested_folds) * len(request.engine_ids)
    progress_done = 0
    emit(_alpha_progress(
        phase="validation",
        label="滚动发现与外层样本外验证",
        done=0,
        total=total_steps,
        request=request,
        engines=engine_progress,
    ))
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
                "hypothesis_contract": request.hypothesis_contract,
            },
        )
        if request.champion_strategy_id is not None:
            benchmark_eval = evaluator.evaluate_candidate_labels(
                outer.train_labels,
                outer.test_labels,
                {"kind": "existing_strategy", "strategy_id": request.champion_strategy_id},
            )
            benchmark_folds.append(_evaluation_row(outer, None, benchmark_eval, "champion"))

        for engine_id in request.engine_ids:
            engine = registry.get(engine_id)
            progress_row = engine_progress[engine_id]
            progress_row["status"] = "running"
            progress_row["message"] = f"正在处理外层窗口 {fold_number}/{len(nested_folds)}"
            emit(_alpha_progress(
                phase="validation",
                label=f"{engine.manifest.name} · 外层窗口 {fold_number}/{len(nested_folds)}",
                done=progress_done,
                total=total_steps,
                request=request,
                engines=engine_progress,
                current_engine_id=engine_id,
            ))
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
                discovery_trial_count = len(engine_trials)
                trial_ledger.extend({
                    "outer_index": outer.outer_index,
                    "engine_id": engine_id,
                    "stage": "discovery",
                    **row,
                } for row in engine_trials)
                progress_row["trials"] += discovery_trial_count
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
                progress_row["trials"] += len(selection_evidence)
                if selected is None:
                    engine_folds[engine_id].append({
                        "outer_index": outer.outer_index,
                        "train_start": outer.train_start,
                        "train_end": outer.train_end,
                        "test_start": outer.test_start,
                        "test_end": outer.test_end,
                        "selection_rejection": "inner selection produced no finite candidate",
                    })
                    continue
                progress_row["selected"] += 1
                frozen = engine.materialize(selected, engine_context)
                evaluation = evaluator.evaluate_candidate_labels(
                    outer.train_labels,
                    outer.test_labels,
                    frozen.definition,
                )
                engine_folds[engine_id].append(
                    _evaluation_row(outer, selected, evaluation, "candidate")
                )
                progress_row["backtests"] += 1
            except Exception as exc:
                discovery_trial_count = len(engine_trials)
                trial_ledger.extend({
                    "outer_index": outer.outer_index,
                    "engine_id": engine_id,
                    "stage": "discovery",
                    **row,
                } for row in engine_trials)
                progress_row["trials"] += discovery_trial_count
                progress_row["errors"] += 1
                progress_row["message"] = str(exc)[:160]
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
            finally:
                progress_done += 1
                progress_row["folds_done"] = fold_number
                if (
                    fold_number == len(nested_folds)
                    and progress_row["errors"] >= progress_row["folds_total"]
                ):
                    progress_row["status"] = "failed"
                else:
                    progress_row["status"] = (
                        "completed" if fold_number == len(nested_folds) else "waiting"
                    )
                if not progress_row["errors"]:
                    progress_row["message"] = (
                        "外层验证完成" if fold_number == len(nested_folds)
                        else f"等待外层窗口 {fold_number + 1}/{len(nested_folds)}"
                    )
                emit(_alpha_progress(
                    phase="validation",
                    label=f"完成 {engine.manifest.name} · 窗口 {fold_number}/{len(nested_folds)}",
                    done=progress_done,
                    total=total_steps,
                    request=request,
                    engines=engine_progress,
                    current_engine_id=None,
                ))
    phase_ms["validation"] = round((time.perf_counter() - search_started) * 1000, 1)

    champion_metrics = _aggregate_fold_metrics(benchmark_folds)
    candidates_summary: list[dict[str, Any]] = []
    stress_backtest_count = 0
    stress_total = sum(
        1 for folds in engine_folds.values() if _representative_candidate(folds) is not None
    )
    stress_done = 0
    for engine_id, folds in engine_folds.items():
        metrics = _aggregate_fold_metrics(folds, champion_folds=benchmark_folds)
        frozen_candidate = _representative_candidate(folds)
        if frozen_candidate is None:
            candidates_summary.append({
                "engine_id": engine_id,
                "engine_name": registry.get(engine_id).manifest.name,
                "state": "rejected",
                "metrics": metrics,
                "gates": [],
                "folds": folds,
                "frozen_candidate": None,
                "evidence_reason": "inner_selection_no_finite_candidate",
            })
            continue
        engine_progress[engine_id]["status"] = "stress"
        engine_progress[engine_id]["message"] = "正在执行统一压力测试"
        emit(_alpha_progress(
            phase="stress",
            label=f"{registry.get(engine_id).manifest.name} · 压力测试",
            done=stress_done,
            total=max(stress_total, 1),
            request=request,
            engines=engine_progress,
            current_engine_id=engine_id,
        ))
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
        engine_progress[engine_id]["backtests"] += stress_count
        engine_progress[engine_id]["status"] = "completed"
        engine_progress[engine_id]["message"] = "样本外与压力测试完成"
        stress_done += 1
        emit(_alpha_progress(
            phase="stress",
            label=f"完成 {registry.get(engine_id).manifest.name} · 压力测试",
            done=stress_done,
            total=max(stress_total, 1),
            request=request,
            engines=engine_progress,
        ))
        gates = evaluate_historical_gates(metrics, champion_metrics)
        historical = [gate for gate in gates if gate["id"] != "forward_shadow"]
        failed = any(gate["status"] == "failed" for gate in historical)
        complete = bool(historical) and all(gate["status"] == "passed" for gate in historical)
        state = "rejected" if failed else ("research_candidate" if complete else "outer_evaluated")
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
    discovery_summary = _build_discovery_summary(
        request.engine_ids,
        trial_ledger,
        engine_folds,
        registry,
    )
    market_attribution = {
        candidate["engine_id"]: _build_market_attribution(candidate, panel)
        for candidate in candidates_summary
    }
    candidate_correlations = _build_candidate_correlations(candidates_summary)
    failure_analysis, next_research_suggestions = _build_failure_closure(
        request=request,
        candidates=candidates_summary,
        market_attribution=market_attribution,
        candidate_correlations=candidate_correlations,
        engine_failures=discovery_failures,
        registry=registry,
        catalog_datasets=catalog_snapshot.datasets,
        available_features=feature_names,
    )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    phase_ms["total"] = elapsed_ms
    final_progress = _alpha_progress(
        phase="evidence",
        label="研究计算完成 正在冻结候选证据",
        done=1,
        total=1,
        request=request,
        engines=engine_progress,
    )
    return _sanitize_non_finite({
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
            "source_run_id": request.source_run_id,
            "source_suggestion_id": request.source_suggestion_id,
            "source_candidate_id": request.source_candidate_id,
            "source_diff": dict(request.source_diff or {}),
            "hypothesis_id": request.hypothesis_id,
            "hypothesis_title": (
                request.hypothesis_contract.get("title")
                if request.hypothesis_contract else None
            ),
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
        "discovery_summary": discovery_summary,
        "market_attribution": market_attribution,
        "candidate_correlations": candidate_correlations,
        "failure_analysis": failure_analysis,
        "next_research_suggestions": next_research_suggestions,
        "trial_ledger": trial_ledger,
        "engine_failures": discovery_failures,
        "progress": final_progress,
        "worker_phase_peak_rss_bytes": (
            rss_sampler.phase_peak_rss_bytes() if rss_sampler is not None else None
        ),
    })


def _alpha_progress(
    *,
    phase: str,
    label: str,
    done: int,
    total: int,
    request: AlphaRuntimeRequest,
    engines: Mapping[str, Mapping[str, Any]],
    current_engine_id: str | None = None,
) -> dict[str, Any]:
    rows = [dict(engines[engine_id]) for engine_id in request.engine_ids]
    safe_total = max(total, 1)
    return {
        "phase": phase,
        "label": label,
        "done": done,
        "total": total,
        "percent": round(min(max(done / safe_total * 100.0, 0.0), 100.0), 1),
        "current_engine_id": current_engine_id,
        "trials_used": sum(int(row.get("trials") or 0) for row in rows),
        "trial_limit": (
            request.max_trials_per_engine
            * len(request.engine_ids)
            * max((int(row.get("folds_total") or 0) for row in rows), default=0)
        ),
        "frozen_candidates": sum(int(row.get("selected") or 0) for row in rows),
        "candidate_limit": (
            request.max_candidates_per_engine
            * len(request.engine_ids)
            * max((int(row.get("folds_total") or 0) for row in rows), default=0)
        ),
        "backtests": sum(int(row.get("backtests") or 0) for row in rows),
        "engine_errors": sum(int(row.get("errors") or 0) for row in rows),
        "engines": rows,
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
    champion_value = values.get("champion_strategy_id")
    champion = str(champion_value) if champion_value else None
    if champion is not None:
        champion_strategy = strategy_engine.get(champion)
        if champion_strategy.meta.get("research_only"):
            raise ValueError("Alpha champion cannot be a research-only strategy")
    symbols_value = values.get("symbols")
    symbols = None if symbols_value is None else list(dict.fromkeys(str(v) for v in symbols_value if v))
    return AlphaRuntimeRequest(
        run_id=run_id,
        engine_ids=engine_ids,
        factor_names=factor_names,
        strategy_ids=(champion,) if champion is not None else (),
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
        source_run_id=(str(values["source_run_id"]) if values.get("source_run_id") else None),
        source_suggestion_id=(
            str(values["source_suggestion_id"])
            if values.get("source_suggestion_id") else None
        ),
        source_candidate_id=(
            str(values["source_candidate_id"])
            if values.get("source_candidate_id") else None
        ),
        source_diff=(
            dict(values["source_diff"])
            if isinstance(values.get("source_diff"), Mapping) else None
        ),
        hypothesis_id=(str(values["hypothesis_id"]) if values.get("hypothesis_id") else None),
        hypothesis_contract=(
            dict(values["hypothesis_contract"])
            if isinstance(values.get("hypothesis_contract"), Mapping) else None
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
    coverage_start, coverage_end = _fold_coverage_bounds(folds)
    trade_count = sum(
        len(trades)
        for row in folds
        for metrics in [row.get("metrics")]
        for trades in [metrics.get("trades") if isinstance(metrics, Mapping) else None]
        if isinstance(trades, list)
    )
    if not returns:
        return {
            "stitched_oos_return": None,
            "stitched_oos_sharpe": None,
            "max_drawdown": None,
            "positive_half_year_ratio": None,
            "beat_champion_half_year_ratio": None,
            "recent_1y_return": None,
            "recent_3m_return": None,
            "recent_1y_available": False,
            "recent_3m_available": False,
            "oos_start": coverage_start.isoformat() if coverage_start else None,
            "oos_end": coverage_end.isoformat() if coverage_end else None,
            "oos_calendar_days": (
                (coverage_end - coverage_start).days + 1
                if coverage_start and coverage_end else 0
            ),
            "oos_days": 0,
            "n_trades": trade_count,
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
    end = coverage_end or returns[-1][0]
    start = coverage_start or returns[0][0]
    year_cutoff = end - timedelta(days=365)
    quarter_cutoff = end - timedelta(days=92)
    year_available = start <= year_cutoff
    quarter_available = start <= quarter_cutoff
    return {
        "stitched_oos_return": total_return,
        "stitched_oos_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "positive_half_year_ratio": positive_ratio,
        "beat_champion_half_year_ratio": beat_ratio,
        "recent_1y_return": (
            _compound([value for day, value in returns if day > year_cutoff])
            if year_available else None
        ),
        "recent_3m_return": (
            _compound([value for day, value in returns if day > quarter_cutoff])
            if quarter_available else None
        ),
        "recent_1y_available": year_available,
        "recent_3m_available": quarter_available,
        "oos_start": start.isoformat(),
        "oos_end": end.isoformat(),
        "oos_calendar_days": (end - start).days + 1,
        "half_year_windows": len(half_years),
        "oos_days": len(returns),
        "n_trades": trade_count,
        "equity_curve": _equity_curve(returns),
    }


def _build_discovery_summary(
    engine_ids: Sequence[str],
    trial_ledger: Sequence[Mapping[str, Any]],
    engine_folds: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Any,
) -> list[dict[str, Any]]:
    """Project internal trial rows into stable, user-facing discovery evidence."""
    output: list[dict[str, Any]] = []
    for engine_id in engine_ids:
        rows = [row for row in trial_ledger if row.get("engine_id") == engine_id]
        discovery_rows = [row for row in rows if row.get("stage") == "discovery"]
        selection_rows = [row for row in rows if "penalized_score" in row]
        finite_scores = [
            float(row["penalized_score"])
            for row in selection_rows
            if _is_finite(row.get("penalized_score"))
        ]
        fold_rows = list(engine_folds.get(engine_id, ()))
        selected_rows = [row for row in fold_rows if isinstance(row.get("candidate"), Mapping)]
        recipe_counts: dict[str, int] = {}
        for row in selected_rows:
            recipe_id = str(row.get("recipe_id") or "")
            if recipe_id:
                recipe_counts[recipe_id] = recipe_counts.get(recipe_id, 0) + 1
        representative = _representative_candidate(fold_rows)
        output.append({
            "engine_id": engine_id,
            "engine_name": registry.get(engine_id).manifest.name,
            "discovery_trials": len(discovery_rows),
            "selection_trials": len(selection_rows),
            "finite_selection_trials": len(finite_scores),
            "selected_folds": len(selected_rows),
            "outer_folds": len(fold_rows),
            "selection_stability": (
                len(selected_rows) / len(fold_rows) if fold_rows else None
            ),
            "best_penalized_score": max(finite_scores) if finite_scores else None,
            "recipes_considered": len({
                str(row.get("recipe_id"))
                for row in rows
                if row.get("recipe_id")
            }),
            "selected_recipe_id": (
                representative.get("recipe_id") if representative else None
            ),
            "selected_recipe_fold_count": (
                recipe_counts.get(str(representative.get("recipe_id")), 0)
                if representative else 0
            ),
            "errors": sum(bool(row.get("error")) for row in fold_rows),
        })
    return output


def _curve_daily_returns(curve: Any) -> list[tuple[date, float]]:
    if not isinstance(curve, list):
        return []
    output: list[tuple[date, float]] = []
    previous = 1.0
    for row in curve:
        if not isinstance(row, Mapping):
            continue
        try:
            day = date.fromisoformat(str(row.get("date"))[:10])
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if previous > 0:
            daily_return = value / previous - 1.0
            if math.isfinite(daily_return):
                output.append((day, daily_return))
        previous = value
    return output


def _market_state(market_return: float, breadth: float) -> str:
    if (market_return >= 0.01 and breadth >= 0.55) or breadth >= 0.70:
        return "strong_up"
    if (market_return <= -0.01 and breadth <= 0.45) or breadth <= 0.30:
        return "strong_down"
    return "weak_up" if market_return >= 0 else "weak_down"


def _build_market_attribution(
    candidate: Mapping[str, Any],
    panel: pl.DataFrame,
) -> dict[str, Any]:
    """Explain frozen OOS returns by actual contemporaneous market and PIT contexts."""
    metrics = candidate.get("metrics")
    curve = metrics.get("equity_curve") if isinstance(metrics, Mapping) else None
    candidate_returns = dict(_curve_daily_returns(curve))
    if not candidate_returns:
        return {
            "available": False,
            "reason": "候选没有形成可归因的样本外日收益序列",
            "regimes": [],
            "years": [],
            "industries": {"available": False, "reason": "没有可归因交易", "rows": []},
            "concepts": {
                "available": False,
                "reason": "当前只有概念成员快照。禁止倒填历史归因",
                "rows": [],
            },
        }

    market_column = "change_pct" if "change_pct" in panel.columns else None
    if market_column is None:
        return {
            "available": False,
            "reason": "研究面板缺少可复核的全市场日收益",
            "regimes": [],
            "years": [],
            "industries": {"available": False, "reason": "市场归因前置数据缺失", "rows": []},
            "concepts": {
                "available": False,
                "reason": "当前只有概念成员快照。禁止倒填历史归因",
                "rows": [],
            },
        }

    market_rows = (
        panel.select("date", market_column)
        .drop_nulls()
        .filter(pl.col("date").is_in(list(candidate_returns)))
        .group_by("date")
        .agg(
            pl.col(market_column).mean().alias("market_return"),
            (pl.col(market_column) > 0).mean().alias("breadth"),
            pl.len().alias("stock_count"),
        )
        .sort("date")
    )
    daily: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {
        "strong_up": [], "weak_up": [], "weak_down": [], "strong_down": [],
    }
    for row in market_rows.iter_rows(named=True):
        day = row["date"]
        candidate_return = candidate_returns.get(day)
        if candidate_return is None:
            continue
        market_return = float(row["market_return"])
        breadth = float(row["breadth"])
        state = _market_state(market_return, breadth)
        grouped[state].append(candidate_return)
        daily.append({
            "date": day.isoformat(),
            "state": state,
            "candidate_return": candidate_return,
            "market_return": market_return,
            "breadth": breadth,
            "stock_count": int(row["stock_count"]),
        })

    state_labels = {
        "strong_up": "强势上涨", "weak_up": "温和上涨",
        "weak_down": "温和下跌", "strong_down": "强势下跌",
    }
    regimes = [
        {
            "state": state,
            "label": state_labels[state],
            "days": len(values),
            "return": _compound(values) if values else None,
            "positive_day_ratio": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
            "contribution": sum(values),
        }
        for state, values in grouped.items()
    ]
    years = [
        {"year": year, "days": len(values), "return": _compound(values)}
        for year, values in sorted(_group_returns_by_year(candidate_returns).items())
    ]
    industries = _industry_attribution(candidate, panel)
    return {
        "available": bool(daily),
        "reason": None if daily else "样本外日收益与市场横截面没有重合日期",
        "regimes": regimes,
        "years": years,
        "daily": daily,
        "industries": industries,
        "concepts": {
            "available": False,
            "reason": "当前只有概念成员快照。禁止倒填历史归因",
            "rows": [],
        },
    }


def _group_returns_by_year(returns: Mapping[date, float]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for day, value in sorted(returns.items()):
        output.setdefault(str(day.year), []).append(value)
    return output


def _industry_attribution(
    candidate: Mapping[str, Any],
    panel: pl.DataFrame,
) -> dict[str, Any]:
    industry_column = next(
        (name for name in ("l1_name", "l1_code", "industry_name", "industry_code") if name in panel.columns),
        None,
    )
    if industry_column is None:
        return {
            "available": False,
            "reason": "缺少历史时点行业归属。未进行行业贡献推断",
            "rows": [],
        }
    folds = candidate.get("folds") or []
    trades = [
        trade
        for fold in folds
        if isinstance(fold, Mapping)
        for trade in ((fold.get("metrics") or {}).get("trades") or [])
        if isinstance(trade, Mapping)
    ]
    if not trades:
        return {"available": False, "reason": "候选没有已完成交易", "rows": []}
    lookup = {
        (str(row["symbol"]), str(row["date"])[:10]): str(row[industry_column] or "未分类")
        for row in panel.select("symbol", "date", industry_column).iter_rows(named=True)
    }
    grouped: dict[str, dict[str, Any]] = {}
    for trade in trades:
        key = lookup.get(
            (str(trade.get("symbol")), str(trade.get("entry_date"))[:10]),
            "未分类",
        )
        row = grouped.setdefault(key, {"industry": key, "trades": 0, "pnl": 0.0, "wins": 0})
        pnl = float(trade.get("pnl_amount") or 0.0)
        row["trades"] += 1
        row["pnl"] += pnl
        row["wins"] += int(pnl > 0)
    rows = sorted(grouped.values(), key=lambda row: abs(float(row["pnl"])), reverse=True)
    for row in rows:
        row["win_rate"] = row.pop("wins") / row["trades"] if row["trades"] else None
    return {"available": True, "reason": None, "rows": rows[:20]}


def _build_candidate_correlations(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    returns = {
        str(candidate.get("engine_id")): dict(_curve_daily_returns(
            (candidate.get("metrics") or {}).get("equity_curve")
        ))
        for candidate in candidates
    }
    output: list[dict[str, Any]] = []
    ids = list(returns)
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1:]:
            common = sorted(set(returns[left_id]) & set(returns[right_id]))
            correlation = None
            if len(common) >= 5:
                value = float(np.corrcoef(
                    [returns[left_id][day] for day in common],
                    [returns[right_id][day] for day in common],
                )[0, 1])
                correlation = value if math.isfinite(value) else None
            output.append({
                "left_engine_id": left_id,
                "right_engine_id": right_id,
                "overlap_days": len(common),
                "correlation": correlation,
            })
    return output


_CONTINUING_ALPHA_STATES = frozenset({"validation_candidate", "research_candidate", "shadow", "challenger", "champion"})
_OOS_DECAY_GATES = frozenset({
    "return_vs_champion",
    "sharpe",
    "drawdown",
    "positive_half_years",
    "beat_champion_windows",
    "recent_year",
    "recent_quarter",
    "parameter_perturbation",
})
_EXECUTION_GATES = frozenset({"double_cost", "delay"})
_CONCENTRATION_GATES = frozenset({"capacity", "concentration"})
_FAILURE_GATE_LABELS = {
    "return_vs_champion": "拼接样本外净收益",
    "sharpe": "样本外夏普",
    "drawdown": "最大回撤",
    "positive_half_years": "正收益半年窗口",
    "beat_champion_windows": "半年窗口稳定性",
    "recent_year": "最近一年收益",
    "recent_quarter": "最近三个月收益",
    "double_cost": "双倍成本",
    "delay": "延迟成交",
    "parameter_perturbation": "参数扰动",
    "capacity": "持仓容量",
    "concentration": "收益集中度",
}


def _failure_gate_names(values: Sequence[str]) -> str:
    return "、".join(_FAILURE_GATE_LABELS.get(value, value) for value in sorted(values))


def _build_failure_closure(
    *,
    request: AlphaRuntimeRequest,
    candidates: Sequence[Mapping[str, Any]],
    market_attribution: Mapping[str, Mapping[str, Any]],
    candidate_correlations: Sequence[Mapping[str, Any]],
    engine_failures: Sequence[Mapping[str, Any]],
    registry: Any,
    catalog_datasets: Mapping[str, Any],
    available_features: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze a deterministic failure diagnosis and user-confirmed next-run recipes.

    This is deliberately descriptive, not an auto-tuner.  Every suggestion changes the
    frozen request, remains unstarted, and can therefore only become a new experiment
    after an explicit user action.
    """
    continuing = [row for row in candidates if row.get("state") in _CONTINUING_ALPHA_STATES]
    frozen = [row for row in candidates if isinstance(row.get("frozen_candidate"), Mapping)]
    zero_pass = not continuing
    funnel = {
        "selected_engines": len(request.engine_ids),
        "frozen_candidates": len(frozen),
        "outer_evaluated": sum(
            int(_finite_number((row.get("metrics") or {}).get("oos_days")) or 0) > 0
            for row in candidates
            if isinstance(row.get("metrics"), Mapping)
        ),
        "historical_gate_passed": len(continuing),
    }
    if not zero_pass:
        return ({
            "zero_pass": False,
            "conclusion": f"本轮有{len(continuing)}个候选通过历史硬门槛，失败闭环未触发。",
            "funnel": funnel,
            "best_failed_candidate": None,
            "primary_category_id": None,
            "categories": [],
            "excluded_recipe_ids": [],
        }, [])

    candidate_by_engine = {str(row.get("engine_id")): row for row in candidates}
    failed_gates: dict[str, set[str]] = {}
    pending_gates: dict[str, set[str]] = {}
    for row in candidates:
        engine_id = str(row.get("engine_id") or "")
        gates = row.get("gates") if isinstance(row.get("gates"), list) else []
        failed_gates[engine_id] = {
            str(gate.get("id")) for gate in gates
            if isinstance(gate, Mapping) and gate.get("status") == "failed"
        }
        pending_gates[engine_id] = {
            str(gate.get("id")) for gate in gates
            if isinstance(gate, Mapping) and gate.get("status") == "pending"
        }

    categories: list[dict[str, Any]] = []

    def add_category(
        category_id: str,
        label: str,
        count: int,
        evidence: Sequence[str],
        keep: Sequence[str],
        change: Sequence[str],
        why: str,
        severity: str = "high",
    ) -> None:
        if count <= 0:
            return
        categories.append({
            "id": category_id,
            "label": label,
            "count": count,
            "severity": severity,
            "evidence": list(dict.fromkeys(evidence)),
            "keep": list(keep),
            "change": list(change),
            "why": why,
        })

    invalid = [row for row in candidates if not isinstance(row.get("frozen_candidate"), Mapping)]
    add_category(
        "signal_invalid",
        "训练或内选未形成有效信号",
        len(invalid),
        [f"{row.get('engine_name') or row.get('engine_id')}：没有有限分的冻结方案" for row in invalid],
        ["保持原始研究区间、预测周期与成交成本口径"],
        ["扩大训练内选预算，或更换无法形成信号的发现路径"],
        "当前失败发生在外测之前，不能通过修改回测结果或门槛来补救。",
    )

    oos_failed = [engine_id for engine_id, gates in failed_gates.items() if gates & _OOS_DECAY_GATES]
    add_category(
        "oos_decay",
        "样本外收益或稳定性衰减",
        len(oos_failed),
        [
            f"{candidate_by_engine[engine_id].get('engine_name') or engine_id}：失败门槛 "
            + _failure_gate_names(failed_gates[engine_id] & _OOS_DECAY_GATES)
            for engine_id in oos_failed
        ],
        ["保留点时股票池、外层样本隔离和统一撮合"],
        ["引入不同机制的发现引擎，不修改本轮冻结公式"],
        "训练期规律没有稳定迁移到从未见过的外层窗口，需要改变发现路径而不是回看外测调参。",
    )

    execution_failed = [engine_id for engine_id, gates in failed_gates.items() if gates & _EXECUTION_GATES]
    add_category(
        "execution_cost",
        "交易成本或成交延迟吞噬收益",
        len(execution_failed),
        [
            f"{candidate_by_engine[engine_id].get('engine_name') or engine_id}："
            + _failure_gate_names(failed_gates[engine_id] & _EXECUTION_GATES)
            for engine_id in execution_failed
        ],
        ["保持次日开盘成交和真实费用模型"],
        ["补充流动性与换手特征，重新发现可交易信号"],
        "放宽成本假设会制造虚假收益；下一轮只能提高信号对成本与延迟的容忍度。",
    )

    concentration_failed = [
        engine_id for engine_id, gates in failed_gates.items() if gates & _CONCENTRATION_GATES
    ]
    highly_correlated = [
        row for row in candidate_correlations
        if _finite_number(row.get("correlation")) is not None
        and abs(float(row["correlation"])) >= 0.90
    ]
    concentration_evidence = [
        f"{candidate_by_engine[engine_id].get('engine_name') or engine_id}："
        + _failure_gate_names(failed_gates[engine_id] & _CONCENTRATION_GATES)
        for engine_id in concentration_failed
    ]
    concentration_evidence.extend(
        f"{candidate_by_engine.get(str(row.get('left_engine_id')), {}).get('engine_name') or row.get('left_engine_id')} 与 "
        f"{candidate_by_engine.get(str(row.get('right_engine_id')), {}).get('engine_name') or row.get('right_engine_id')} 的样本外相关性为"
        f"{float(row['correlation']):.2f}"
        for row in highly_correlated
    )
    add_category(
        "capacity_concentration",
        "容量或收益集中度不合格",
        max(len(concentration_failed), len(highly_correlated)),
        concentration_evidence,
        ["保持统一成交、费用和外测窗口"],
        ["剔除高度重复的发现路径，补充不同信息域的引擎"],
        "多个候选若共享相同风险暴露，数量增加也不会形成可分散的Alpha。",
    )

    regime_evidence: list[str] = []
    regime_engines: list[str] = []
    for engine_id, evidence in market_attribution.items():
        regimes = evidence.get("regimes") if isinstance(evidence, Mapping) else None
        if not isinstance(regimes, list):
            continue
        total_days = sum(int(row.get("days") or 0) for row in regimes if isinstance(row, Mapping))
        top = max(
            (row for row in regimes if isinstance(row, Mapping)),
            key=lambda row: int(row.get("days") or 0),
            default=None,
        )
        if total_days < 20 or top is None:
            continue
        top_days = int(top.get("days") or 0)
        if top_days / total_days < 0.75:
            continue
        regime_engines.append(engine_id)
        regime_evidence.append(
            f"{candidate_by_engine.get(engine_id, {}).get('engine_name') or engine_id}："
            f"{top.get('label') or top.get('state')}覆盖{top_days}/{total_days}个样本外交易日"
        )
    add_category(
        "regime_dependency",
        "市场状态覆盖单一或收益依赖特定行情",
        len(regime_engines),
        regime_evidence,
        ["保持原候选的冻结证据，不把当前市场状态外推为长期规律"],
        ["加入市场/行业时序或网络扩散发现路径"],
        "样本外大多落在同一市场状态时，无法证明信号跨行情有效。",
        severity="medium",
    )

    coverage_engines = [
        engine_id for engine_id, gates in pending_gates.items()
        if gates & {"recent_year", "recent_quarter"}
    ]
    add_category(
        "insufficient_coverage",
        "近期或分段证据覆盖不足",
        len(coverage_engines),
        [
            f"{candidate_by_engine[engine_id].get('engine_name') or engine_id}："
            + _failure_gate_names(pending_gates[engine_id] & {"recent_year", "recent_quarter"})
            + "尚未形成完整窗口"
            for engine_id in coverage_engines
        ],
        ["保留缺失值，不把不完整窗口伪装成零收益或通过"],
        ["仅在历史长度足够时升级研究档位并重新形成外层窗口"],
        "覆盖不足是证据缺口，不代表收益为零，也不能靠前端填值。",
        severity="medium",
    )

    fold_errors = sum(
        bool(fold.get("error"))
        for row in candidates
        for fold in (row.get("folds") if isinstance(row.get("folds"), list) else [])
        if isinstance(fold, Mapping)
    )
    error_evidence = [
        f"{row.get('engine_id')} · {row.get('stage')}：{str(row.get('error') or '')[:160]}"
        for row in engine_failures
    ]
    if fold_errors:
        error_evidence.append(f"另有{fold_errors}个外层窗口执行失败")
    add_category(
        "engine_or_data_error",
        "发现引擎或数据链路异常",
        len(engine_failures) + fold_errors,
        error_evidence,
        ["保留已成功引擎和所有已完成外测证据"],
        ["移除失败引擎或修复数据门禁后创建新运行"],
        "执行异常与策略证伪是两类问题，必须先恢复可复核链路再研究收益。",
    )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    categories.sort(key=lambda row: (severity_rank.get(str(row["severity"]), 9), -int(row["count"]), str(row["id"])))
    best = frozen[0] if frozen else None
    best_failed = None
    if best is not None:
        metrics = best.get("metrics") if isinstance(best.get("metrics"), Mapping) else {}
        best_failed = {
            "engine_id": best.get("engine_id"),
            "engine_name": best.get("engine_name"),
            "recipe_id": (best.get("frozen_candidate") or {}).get("recipe_id"),
            "stitched_oos_return": metrics.get("stitched_oos_return"),
            "stitched_oos_sharpe": metrics.get("stitched_oos_sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "failed_gate_ids": sorted(failed_gates.get(str(best.get("engine_id")), set())),
            "pending_gate_ids": sorted(pending_gates.get(str(best.get("engine_id")), set())),
        }

    ready_engine_ids = _ready_followup_engines(
        request=request,
        registry=registry,
        catalog_datasets=catalog_datasets,
        available_features=available_features,
    )
    suggestions = _build_followup_suggestions(
        request=request,
        categories=categories,
        candidates=candidate_by_engine,
        correlations=highly_correlated,
        engine_failures=engine_failures,
        ready_engine_ids=ready_engine_ids,
    )
    return ({
        "zero_pass": True,
        "conclusion": (
            f"本轮{len(candidates)}条发现路径均未通过全部历史硬门槛；"
            f"失败已归入{len(categories)}类，旧方案保持冻结。"
        ),
        "funnel": funnel,
        "best_failed_candidate": best_failed,
        "primary_category_id": categories[0]["id"] if categories else None,
        "categories": categories,
        "excluded_recipe_ids": [
            str(row["frozen_candidate"]["recipe_id"])
            for row in frozen
            if row.get("state") == "rejected" and row["frozen_candidate"].get("recipe_id")
        ],
    }, suggestions)


def _ready_followup_engines(
    *,
    request: AlphaRuntimeRequest,
    registry: Any,
    catalog_datasets: Mapping[str, Any],
    available_features: Sequence[str],
) -> list[str]:
    features = set(available_features)
    output: list[str] = []
    for engine in registry.list():
        manifest = engine.manifest
        if manifest.readiness != "ready":
            continue
        if request.asset_type not in manifest.asset_types:
            continue
        if request.forward_horizon not in manifest.forecast_horizons:
            continue
        if not set(manifest.required_features).issubset(features):
            continue
        context = DataCatalogContext(
            asset_type=request.asset_type,
            start=request.start.isoformat(),
            end=request.end.isoformat(),
            available_features=tuple(available_features),
            datasets=catalog_datasets,
        )
        if not qualify_manifest_datasets(manifest, catalog_datasets).ready:
            continue
        if not engine.preflight(context).ready:
            continue
        output.append(manifest.engine_id)
    return output


def _build_followup_suggestions(
    *,
    request: AlphaRuntimeRequest,
    categories: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    engine_failures: Sequence[Mapping[str, Any]],
    ready_engine_ids: Sequence[str],
) -> list[dict[str, Any]]:
    category_ids = {str(row.get("id")) for row in categories}
    selected = list(request.engine_ids)
    factors = list(request.factor_names)
    ready_alternatives = [engine_id for engine_id in ready_engine_ids if engine_id not in selected]
    suggestions: list[dict[str, Any]] = []
    seen_patches: set[str] = set()

    def add(
        category_id: str,
        title: str,
        why: str,
        keep: Sequence[str],
        patch: Mapping[str, Any],
        changes: Sequence[Mapping[str, Any]],
    ) -> None:
        if category_id not in category_ids or not patch or not changes:
            return
        fingerprint = repr(sorted((key, repr(value)) for key, value in patch.items()))
        if fingerprint in seen_patches:
            return
        seen_patches.add(fingerprint)
        suggestions.append({
            "suggestion_id": f"next-{category_id}-{len(suggestions) + 1}",
            "category_id": category_id,
            "title": title,
            "why": why,
            "keep": list(keep),
            "changes": [dict(row) for row in changes],
            "request_patch": dict(patch),
        })

    common_keep = [
        f"研究区间 {request.start.isoformat()} 至 {request.end.isoformat()}",
        f"未来{request.forward_horizon}日净收益标签",
        "点时股票池、次日开盘成交与原始费用口径",
    ]

    if "capacity_concentration" in category_ids:
        replacement = list(selected)
        removed: str | None = None
        if correlations:
            pair = correlations[0]
            left = str(pair.get("left_engine_id") or "")
            right = str(pair.get("right_engine_id") or "")
            left_return = _sortable_return((candidates.get(left, {}).get("metrics") or {}).get("stitched_oos_return"))
            right_return = _sortable_return((candidates.get(right, {}).get("metrics") or {}).get("stitched_oos_return"))
            removed = right if left_return >= right_return else left
        elif len(replacement) > 1:
            removed = min(
                replacement,
                key=lambda engine_id: _sortable_return(
                    (candidates.get(engine_id, {}).get("metrics") or {}).get("stitched_oos_return")
                ),
            )
        if removed in replacement and len(replacement) > 1:
            replacement.remove(removed)
        preferred = next(
            (engine_id for engine_id in ("market_sector_timing", "network_diffusion", "portfolio_residual") if engine_id in ready_alternatives),
            ready_alternatives[0] if ready_alternatives else None,
        )
        if preferred and preferred not in replacement:
            replacement.append(preferred)
        if replacement != selected:
            add(
                "capacity_concentration",
                "去掉重复暴露，补充不同信息域",
                "优先减少样本外高度相关的路径；若数据允许，再加入机制不同的引擎。",
                common_keep,
                {"engine_ids": replacement},
                [{
                    "field": "engine_ids",
                    "label": "发现引擎",
                    "before": selected,
                    "after": replacement,
                    "reason": "减少重复风险暴露并扩大机制差异",
                }],
            )

    if "regime_dependency" in category_ids:
        preferred = next(
            (engine_id for engine_id in ("market_sector_timing", "network_diffusion") if engine_id in ready_alternatives),
            None,
        )
        if preferred:
            after = [*selected, preferred]
            add(
                "regime_dependency",
                "加入市场状态或扩散路径",
                "当前外测市场状态覆盖单一，下一轮增加能够显式研究状态切换的信息域。",
                common_keep,
                {"engine_ids": after},
                [{"field": "engine_ids", "label": "发现引擎", "before": selected, "after": after, "reason": "补充跨市场状态的发现路径"}],
            )

    if "execution_cost" in category_ids:
        additions = [name for name in ("turnover_rate", "amihud_20d", "log_amount") if name in _FACTOR_IDS and name not in factors]
        if additions:
            after = [*factors, *additions]
            add(
                "execution_cost",
                "补充流动性与换手约束特征",
                "保持真实成本不变，让发现引擎直接识别更可成交、对摩擦更不敏感的信号。",
                common_keep,
                {"factor_names": after},
                [{"field": "factor_names", "label": "研究因子", "before": factors, "after": after, "reason": "显式纳入流动性和换手成本代理"}],
            )

    if "signal_invalid" in category_ids:
        if request.profile == "exploratory":
            add(
                "signal_invalid",
                "扩大训练内选预算",
                "当前引擎在快速档没有形成有限分方案；扩大训练预算，但仍保持外层测试不可见。",
                common_keep,
                {"budget_profile": "balanced", "max_trials_per_engine": 64, "max_candidates_per_engine": 4},
                [
                    {"field": "budget_profile", "label": "研究强度", "before": request.profile, "after": "balanced", "reason": "增加训练内选覆盖"},
                    {"field": "max_trials_per_engine", "label": "每引擎尝试上限", "before": request.max_trials_per_engine, "after": 64, "reason": "只扩大训练预算，不读取外测"},
                ],
            )
        elif ready_alternatives:
            after = [*selected, ready_alternatives[0]]
            add(
                "signal_invalid",
                "加入另一条可运行发现路径",
                "原路径未形成信号，保留其失败证据并增加不同引擎，而不是改写旧方案。",
                common_keep,
                {"engine_ids": after},
                [{"field": "engine_ids", "label": "发现引擎", "before": selected, "after": after, "reason": "扩大独立发现路径"}],
            )

    if "oos_decay" in category_ids and ready_alternatives:
        preferred = ready_alternatives[0]
        after = [*selected, preferred]
        add(
            "oos_decay",
            "增加不同机制的独立发现路径",
            "旧冻结公式保留为失败证据；新一轮只扩大机制覆盖，不针对外测收益微调旧公式。",
            common_keep,
            {"engine_ids": after},
            [{"field": "engine_ids", "label": "发现引擎", "before": selected, "after": after, "reason": "降低单一发现机制的样本外衰减风险"}],
        )

    if "engine_or_data_error" in category_ids:
        failed_ids = {str(row.get("engine_id") or "") for row in engine_failures}
        after = [engine_id for engine_id in selected if engine_id not in failed_ids]
        if not after and ready_alternatives:
            after = [ready_alternatives[0]]
        if after and after != selected:
            add(
                "engine_or_data_error",
                "隔离执行失败的引擎",
                "保留成功路径，移除本轮明确发生执行异常的路径；旧异常日志不删除。",
                common_keep,
                {"engine_ids": after},
                [{"field": "engine_ids", "label": "发现引擎", "before": selected, "after": after, "reason": "先恢复可复核运行链路"}],
            )
    return suggestions


def _fold_coverage_bounds(
    folds: Sequence[Mapping[str, Any]],
) -> tuple[date | None, date | None]:
    """Return the actual outer-test coverage, excluding folds with no backtest curve."""
    starts: list[date] = []
    ends: list[date] = []
    for fold in folds:
        metrics = fold.get("metrics")
        curve = metrics.get("equity_curve") if isinstance(metrics, Mapping) else None
        if not isinstance(curve, list) or not curve:
            continue
        try:
            starts.append(date.fromisoformat(str(fold.get("test_start"))[:10]))
            ends.append(date.fromisoformat(str(fold.get("test_end"))[:10]))
        except ValueError:
            continue
    return (min(starts), max(ends)) if starts and ends else (None, None)


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sanitize_non_finite(value: Any) -> Any:
    """Keep compact worker results strict-JSON serializable without inventing metrics."""
    if isinstance(value, Mapping):
        return {str(key): _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(item) for item in value]
    if isinstance(value, np.generic):
        return _sanitize_non_finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
