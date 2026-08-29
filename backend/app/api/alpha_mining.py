"""Independent Alpha mining API. The legacy ``/api/backtest/mining`` API is untouched."""
# ruff: noqa: RUF001
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sse_starlette.sse import EventSourceResponse

from app.alpha_mining.ai_hypothesis_proposer import AlphaAIHypothesisProposer
from app.alpha_mining.champion import AlphaChampionStore, promotion_lock
from app.alpha_mining.config_store import AlphaConfigStore
from app.alpha_mining.contracts import DataCatalogContext, qualify_manifest_datasets
from app.alpha_mining.data_catalog import AlphaResearchDataCatalog
from app.alpha_mining.evidence import AlphaEvidenceError, AlphaEvidenceStore
from app.alpha_mining.hypotheses import AlphaHypothesisStore
from app.alpha_mining.policy import charter
from app.alpha_mining.publication import AlphaPublicationService
from app.alpha_mining.registry import AlphaEngineRegistryError, load_builtin_registry
from app.alpha_mining.runtime import _validation_config
from app.alpha_mining.shadow import AlphaShadowService
from app.alpha_mining.taxonomy import coverage_matrix
from app.backtest.factor import FACTOR_COLUMNS, FACTOR_METHODOLOGY_VERSION
from app.backtest.mining import generate_nested_folds, required_trading_bars
from app.services.mining_jobs import (
    RUN_STATUSES,
    SUCCESS_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    MiningRunStoreError,
    MiningRunValidationError,
)
from app.services.mining_preflight import enriched_partition_dates

router = APIRouter(prefix="/api/alpha-mining/v1", tags=["alpha-mining"])
_REGISTRY, _ENGINE_LOAD_FAILURES = load_builtin_registry()
_FACTOR_IDS = frozenset(str(item["id"]) for item in FACTOR_COLUMNS)
_FINANCIAL_FACTOR_IDS = frozenset(
    str(item["id"]) for item in FACTOR_COLUMNS if item.get("group") == "财务"
)
_SHARE_HISTORY_FACTOR_IDS = frozenset(
    {"turnover_rate", "turnover_ratio_5d", "turnover_z_60d"}
)
_DEFAULT_FACTORS = [str(item["id"]) for item in FACTOR_COLUMNS]
_SUCCESS = frozenset(SUCCESS_RUN_STATUSES)


class AlphaMiningStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    engine_ids: list[str] = Field(min_length=1, max_length=16)
    factor_names: list[str] = Field(default_factory=lambda: list(_DEFAULT_FACTORS), min_length=1)
    symbols: list[str] | None = None
    asset_type: Literal["stock", "etf"] = "stock"
    start: date | None = None
    end: date | None = None
    budget_profile: Literal["exploratory", "balanced", "strict"] = "balanced"
    forward_horizon: Literal[1, 3, 5, 10, 20, 60] = 5
    champion_strategy_id: str | None = Field(None, min_length=1, max_length=120)
    commission_pct: float = Field(0.0002, ge=0.0, le=0.05, allow_inf_nan=False)
    stamp_tax_pct: float = Field(0.0005, ge=0.0, le=0.05, allow_inf_nan=False)
    slippage_bps: float = Field(5.0, ge=0.0, le=1000.0, allow_inf_nan=False)
    max_positions: int = Field(10, ge=1, le=50)
    max_candidates_per_engine: int = Field(4, ge=1, le=12)
    max_trials_per_engine: int = Field(64, ge=4, le=256)
    source_run_id: str | None = Field(None, min_length=1, max_length=80)
    source_suggestion_id: str | None = Field(None, min_length=1, max_length=120)
    source_candidate_id: str | None = Field(None, min_length=1, max_length=80)
    hypothesis_id: str | None = Field(None, min_length=3, max_length=120)
    force: bool = False

    @field_validator("start", "end", mode="before")
    @classmethod
    def _dates(cls, value: Any) -> Any:
        return date.fromisoformat(value) if isinstance(value, str) else value

    @field_validator("engine_ids", "factor_names")
    @classmethod
    def _unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("IDs must be unique")
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("IDs must contain 1 to 120 characters")
        return values

    @field_validator("factor_names")
    @classmethod
    def _known_factors(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - _FACTOR_IDS)
        if unknown:
            raise ValueError(f"unknown Alpha factors: {unknown}")
        return values

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [value for value in values if value]
        if len(cleaned) > 10_000 or len(cleaned) != len(set(cleaned)):
            raise ValueError("symbols must be unique and contain at most 10000 entries")
        return cleaned or None

    @model_validator(mode="after")
    def _range(self):
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")
        source_kind_count = int(bool(self.source_suggestion_id)) + int(bool(self.source_candidate_id))
        if bool(self.source_run_id) != (source_kind_count == 1):
            raise ValueError("source_run_id and exactly one lineage source must be provided together")
        return self


class AlphaConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool | None = None
    auto_run_enabled: bool | None = None
    auto_run_profile: Literal["balanced", "strict"] | None = None
    shadow_min_trading_days: int | None = Field(None, ge=20, le=500)
    shadow_min_fills: int | None = Field(None, ge=20, le=5000)
    shadow_min_factor_round_trips: int | None = Field(None, ge=10, le=2500)
    shadow_min_rank_ic: float | None = Field(None, ge=-1, le=1, allow_inf_nan=False)


class AlphaShadowStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    baseline_date: date | None = None


class AlphaHypothesisCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_kind: Literal["manual"] = "manual"
    title: str = Field(min_length=3, max_length=120)
    thesis: str = Field(min_length=10, max_length=1000)
    mechanism: str = Field(min_length=10, max_length=1000)
    prediction_object: Literal["forward_net_return", "market_residual_return"] = "forward_net_return"
    asset_type: Literal["stock", "etf"] = "stock"
    forward_horizon: Literal[1, 3, 5, 10, 20, 60] = 5
    information_domains: list[str] = Field(min_length=1, max_length=12)
    test_spec: dict[str, Any]
    falsification: list[str] = Field(min_length=1, max_length=12)
    data_requirements: list[str] = Field(min_length=1, max_length=12)


class AlphaHypothesisRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start: date | None = None
    end: date | None = None
    budget_profile: Literal["exploratory", "balanced", "strict"] = "exploratory"
    commission_pct: float = Field(0.0002, ge=0.0, le=0.05, allow_inf_nan=False)
    stamp_tax_pct: float = Field(0.0005, ge=0.0, le=0.05, allow_inf_nan=False)
    slippage_bps: float = Field(5.0, ge=0.0, le=1000.0, allow_inf_nan=False)
    max_positions: int = Field(10, ge=1, le=50)
    force: bool = False

    @field_validator("start", "end", mode="before")
    @classmethod
    def _dates(cls, value: Any) -> Any:
        return date.fromisoformat(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _range(self):
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")
        return self


class AlphaAIHypothesisProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asset_type: Literal["stock", "etf"] = "stock"
    start: date
    end: date
    count: int = Field(3, ge=1, le=6)
    research_focus: str = Field("", max_length=500)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _dates(cls, value: Any) -> Any:
        return date.fromisoformat(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _range(self):
        if self.start > self.end:
            raise ValueError("start must not be after end")
        return self


@router.get("/charter")
def get_charter() -> dict[str, Any]:
    return charter()


@router.get("/engines")
def get_engines() -> dict[str, Any]:
    engines = _REGISTRY.list()
    return {
        "api_version": "1.0",
        "registry_frozen": _REGISTRY.frozen,
        "items": [engine.manifest.to_dict() for engine in engines],
        "taxonomy": coverage_matrix([engine.manifest for engine in engines]),
        "load_failures": list(_ENGINE_LOAD_FAILURES),
    }


@router.get("/hypotheses")
def list_hypotheses(
    request: Request,
    asset_type: Annotated[Literal["stock", "etf"], Query()] = "stock",
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> dict[str, Any]:
    store = AlphaHypothesisStore(_data_dir(request))
    dates = enriched_partition_dates(_data_dir(request), asset_type, start, end)
    catalog = None
    engine_rows: dict[str, dict[str, Any]] = {}
    if dates:
        catalog = AlphaResearchDataCatalog(_data_dir(request)).snapshot(dates[0], dates[-1], asset_type)
        for horizon in (1, 3, 5, 10, 20, 60):
            for row in _engine_availability(asset_type, dates[0], dates[-1], horizon, catalog):
                engine_rows[f"{horizon}:{row['engine_id']}"] = row
    items = []
    for hypothesis in store.list_all():
        row = dict(hypothesis)
        reasons: list[str] = []
        if row["asset_type"] != asset_type:
            reasons.append("该假设不适用于当前资产类型")
        if not dates or catalog is None:
            reasons.append("所选区间没有可用的enriched交易日")
        else:
            for dataset_id in row.get("data_requirements") or []:
                qualification = catalog.datasets.get(str(dataset_id))
                if qualification is None or not qualification.ready:
                    reasons.extend(
                        list(qualification.reasons)
                        if qualification is not None else [f"缺少数据集: {dataset_id}"]
                    )
            for engine_id in (row.get("test_spec") or {}).get("engine_ids") or []:
                engine_row = engine_rows.get(f"{row['forward_horizon']}:{engine_id}")
                if engine_row is None or not engine_row["ready"]:
                    reasons.extend(
                        list(engine_row.get("reasons") or [])
                        if engine_row is not None else [f"发现引擎不可用: {engine_id}"]
                    )
            try:
                _validate_factor_datasets(
                    list((row.get("test_spec") or {}).get("factor_names") or []),
                    catalog,
                )
            except ValueError as exc:
                reasons.append(str(exc))
        row["readiness"] = {"ready": not reasons, "reasons": list(dict.fromkeys(reasons))}
        items.append(row)
    return {"items": items}


@router.post("/hypotheses")
def create_hypothesis(payload: AlphaHypothesisCreateRequest, request: Request) -> dict[str, Any]:
    try:
        return AlphaHypothesisStore(_data_dir(request)).create(payload.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/hypotheses/ai-proposals")
async def propose_ai_hypotheses(
    payload: AlphaAIHypothesisProposalRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        return await AlphaAIHypothesisProposer(_data_dir(request)).propose(
            asset_type=payload.asset_type,
            start=payload.start,
            end=payload.end,
            count=payload.count,
            research_focus=payload.research_focus,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="DeepSeek假设生成失败，请检查AI服务后重试") from exc


@router.get("/hypotheses/{hypothesis_id}")
def get_hypothesis(hypothesis_id: str, request: Request) -> dict[str, Any]:
    try:
        return AlphaHypothesisStore(_data_dir(request)).get(hypothesis_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alpha假设不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/hypotheses/{hypothesis_id}/runs")
def start_hypothesis_run(
    hypothesis_id: str,
    payload: AlphaHypothesisRunRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        hypothesis = AlphaHypothesisStore(_data_dir(request)).get(hypothesis_id)
        spec = dict(hypothesis["test_spec"])
        profile_budget = {
            "exploratory": (2, 24),
            "balanced": (4, 64),
            "strict": (8, 128),
        }[payload.budget_profile]
        request_payload = {
            "engine_ids": list(spec["engine_ids"]),
            "factor_names": list(spec["factor_names"]),
            "asset_type": hypothesis["asset_type"],
            "start": payload.start,
            "end": payload.end,
            "budget_profile": payload.budget_profile,
            "forward_horizon": hypothesis["forward_horizon"],
            "commission_pct": payload.commission_pct,
            "stamp_tax_pct": payload.stamp_tax_pct,
            "slippage_bps": payload.slippage_bps,
            "max_positions": payload.max_positions,
            "max_candidates_per_engine": profile_budget[0],
            "max_trials_per_engine": profile_budget[1],
            "hypothesis_id": hypothesis_id,
            "force": payload.force,
        }
        if hypothesis.get("source_run_id") and hypothesis.get("source_suggestion_id"):
            request_payload.update({
                "source_run_id": hypothesis["source_run_id"],
                "source_suggestion_id": hypothesis["source_suggestion_id"],
            })
        return start_run(AlphaMiningStartRequest.model_validate(request_payload), request)
    except HTTPException:
        raise
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    return AlphaConfigStore(_data_dir(request)).get()


@router.patch("/config")
def patch_config(payload: AlphaConfigPatch, request: Request) -> dict[str, Any]:
    try:
        return AlphaConfigStore(_data_dir(request)).update(
            payload.model_dump(exclude_none=True)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/availability")
def get_availability(
    request: Request,
    asset_type: Annotated[Literal["stock", "etf"], Query()] = "stock",
    budget_profile: Annotated[Literal["exploratory", "balanced", "strict"], Query()] = "balanced",
    forward_horizon: Annotated[int, Query(ge=1, le=60)] = 5,
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> dict[str, Any]:
    if forward_horizon not in {1, 3, 5, 10, 20, 60}:
        raise HTTPException(status_code=400, detail="forward_horizon must be 1, 3, 5, 10, 20, or 60")
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start must not be after end")
    data_dir = request.app.state.repo.store.data_dir
    all_dates = enriched_partition_dates(data_dir, asset_type)
    scoped = [day for day in all_dates if (start is None or day >= start) and (end is None or day <= end)]
    validation = _validation_config(budget_profile, forward_horizon)
    required = required_trading_bars(validation, 1)
    folds = (
        len(generate_nested_folds([day.isoformat() for day in scoped], validation))
        if len(scoped) >= required else 0
    )
    anchor_start = scoped[0] if scoped else (start or end or date.today())
    anchor_end = scoped[-1] if scoped else (end or start or date.today())
    catalog = AlphaResearchDataCatalog(data_dir).snapshot(
        anchor_start,
        anchor_end,
        asset_type,
    )
    engine_rows = _engine_availability(
        asset_type,
        anchor_start,
        anchor_end,
        forward_horizon,
        catalog,
    )
    return {
        "asset_type": asset_type,
        "budget_profile": budget_profile,
        "trading_bars": len(scoped),
        "required_bars": required,
        "outer_folds": folds,
        "eligible": (
            folds > 0
            and catalog.datasets["historical_universe"].ready
            and any(row["ready"] for row in engine_rows)
        ),
        "available_start": all_dates[0].isoformat() if all_dates else None,
        "available_end": all_dates[-1].isoformat() if all_dates else None,
        "effective_start": scoped[0].isoformat() if scoped else None,
        "effective_end": scoped[-1].isoformat() if scoped else None,
        "catalog": catalog.to_dict(),
        "engines": engine_rows,
    }


@router.get("/catalog")
def get_catalog(
    request: Request,
    asset_type: Annotated[Literal["stock", "etf"], Query()] = "stock",
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> dict[str, Any]:
    dates = enriched_partition_dates(_data_dir(request), asset_type, start, end)
    anchor_start = dates[0] if dates else (start or end or date.today())
    anchor_end = dates[-1] if dates else (end or start or date.today())
    return AlphaResearchDataCatalog(_data_dir(request)).snapshot(
        anchor_start,
        anchor_end,
        asset_type,
    ).to_dict()


@router.get("/runs")
def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    status: Annotated[list[str] | None, Query()] = None,
) -> dict[str, Any]:
    manager = _manager(request)
    statuses = None
    if status:
        unknown = sorted(set(status) - RUN_STATUSES)
        if unknown:
            raise HTTPException(status_code=400, detail=f"unsupported Alpha statuses: {unknown}")
        statuses = status
    try:
        return {"items": [_project_run(manager.store, item) for item in manager.store.list_runs(limit=limit, statuses=statuses)]}
    except MiningRunValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MiningRunStoreError as exc:
        raise HTTPException(status_code=500, detail="failed to read Alpha runs") from exc


@router.post("/runs")
def start_run(payload: AlphaMiningStartRequest, request: Request) -> dict[str, Any]:
    manager = _manager(request)
    try:
        if not AlphaConfigStore(_data_dir(request)).get()["enabled"]:
            raise HTTPException(status_code=409, detail="Alpha挖掘功能开关已关闭")
        champion = AlphaChampionStore(_data_dir(request)).get()["current"]
        champion_value = champion.get("strategy_id")
        champion_strategy_id = str(champion_value) if champion_value else None
        if payload.champion_strategy_id not in (None, champion_strategy_id):
            raise ValueError("请求冠军与动态冠军账本不一致")
        if champion_strategy_id is not None:
            request.app.state.strategy_engine.get(champion_strategy_id)
        worker_request = payload.model_dump(mode="json", exclude={"force"})
        worker_request["champion_strategy_id"] = champion_strategy_id
        _attach_hypothesis_contract(manager, worker_request)
        _attach_failure_lineage(manager, worker_request)
        dates = enriched_partition_dates(
            request.app.state.repo.store.data_dir,
            payload.asset_type,
            payload.start,
            payload.end,
        )
        validation = _validation_config(payload.budget_profile, payload.forward_horizon)
        if len(dates) < required_trading_bars(validation, 1):
            raise ValueError("所选区间不足以形成一个隔离的外层样本外窗口")
        catalog = AlphaResearchDataCatalog(_data_dir(request)).snapshot(
            dates[0],
            dates[-1],
            payload.asset_type,
        )
        _validate_factor_datasets(payload.factor_names, catalog)
        _validate_engines(
            payload.engine_ids,
            payload.asset_type,
            data_dir=_data_dir(request),
            start=dates[0],
            end=dates[-1],
            forward_horizon=payload.forward_horizon,
        )
        fingerprint = _data_fingerprint(request, worker_request)
        manifest = manager.start(
            worker_request,
            fingerprint,
            force=payload.force,
            source="manual",
        )
        return _project_run(manager.store, manifest)
    except HTTPException:
        raise
    except (AlphaEngineRegistryError, MiningRunValidationError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MiningRunStoreError as exc:
        raise HTTPException(status_code=500, detail="failed to persist Alpha run") from exc


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    store = _manager(request).store
    return _project_run(store, _required_manifest(store, run_id))


@router.get("/runs/{run_id}/result")
def get_result(run_id: str, request: Request) -> dict[str, Any]:
    store = _manager(request).store
    manifest = _required_manifest(store, run_id)
    if manifest["status"] not in _SUCCESS:
        raise HTTPException(status_code=409, detail=f"Alpha result unavailable for {manifest['status']}")
    return store.read_summary(run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
    try:
        manager = _manager(request)
        return _project_run(manager.store, manager.cancel(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alpha run not found") from exc


@router.get("/experiments")
def list_experiments(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    return {"items": AlphaEvidenceStore(_data_dir(request)).list_experiments(limit)}


@router.get("/candidates")
def list_candidates(request: Request) -> dict[str, Any]:
    return {
        "items": AlphaEvidenceStore(_data_dir(request)).list_candidates(),
        "leaderboard": AlphaChampionStore(_data_dir(request)).leaderboard(),
    }


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    try:
        store = AlphaEvidenceStore(_data_dir(request))
        return {
            "candidate": store.get_candidate(candidate_id),
            "events": store.events(candidate_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alpha candidate not found") from exc
    except AlphaEvidenceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/champion")
def get_champion(request: Request) -> dict[str, Any]:
    return AlphaChampionStore(_data_dir(request)).leaderboard()


@router.post("/candidates/{candidate_id}/shadow")
def start_shadow(
    candidate_id: str,
    payload: AlphaShadowStartRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        from app.services.paper_trading import get_service

        baseline = payload.baseline_date or request.app.state.repo.latest_enriched_date("stock")
        if baseline is None:
            raise ValueError("缺少前向模拟基准交易日")
        return AlphaShadowService(_data_dir(request)).start(
            candidate_id,
            get_service(request.app.state),
            baseline,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/candidates/{candidate_id}/strict-validation")
def start_strict_validation(candidate_id: str, request: Request) -> dict[str, Any]:
    """Create a new full-history strict run for one immutable discovery path."""
    try:
        evidence = AlphaEvidenceStore(_data_dir(request))
        candidate = evidence.get_candidate(candidate_id)
        if candidate["state"]["state"] != "validation_candidate":
            raise ValueError("只有待严格验证的候选可以创建全历史严格运行")
        source = evidence.read_experiment(str(candidate["run_id"]))
        source_request = dict(source.get("contract", {}).get("request") or {})
        asset_type = str(source_request.get("asset_type") or "stock")
        dates = enriched_partition_dates(_data_dir(request), asset_type)
        if not dates:
            raise ValueError("没有可用于严格验证的完整历史交易日")
        base = {
            field: source_request[field]
            for field in AlphaMiningStartRequest.model_fields
            if field in source_request
        }
        payload = AlphaMiningStartRequest.model_validate({
            **base,
            "engine_ids": [candidate["engine_id"]],
            "asset_type": asset_type,
            "start": dates[0],
            "end": dates[-1],
            "budget_profile": "strict",
            "max_candidates_per_engine": 8,
            "max_trials_per_engine": 128,
            "champion_strategy_id": None,
            "source_run_id": candidate["run_id"],
            "source_suggestion_id": None,
            "source_candidate_id": candidate_id,
            "force": True,
        })
        return start_run(payload, request)
    except HTTPException:
        raise
    except (AlphaEvidenceError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/candidates/{candidate_id}/shadow")
def get_shadow(candidate_id: str, request: Request) -> dict[str, Any]:
    try:
        from app.services.paper_trading import get_service

        return AlphaShadowService(_data_dir(request)).status(
            candidate_id,
            get_service(request.app.state),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/candidates/{candidate_id}/shadow/evaluate")
def evaluate_shadow(candidate_id: str, request: Request) -> dict[str, Any]:
    try:
        from app.services.paper_trading import get_service

        return AlphaShadowService(_data_dir(request)).evaluate_and_advance(
            candidate_id,
            get_service(request.app.state),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/candidates/{candidate_id}/promote")
def promote_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    try:
        champion_store = AlphaChampionStore(_data_dir(request))
        with promotion_lock():
            champion_store.validate_promotion(candidate_id)
            publication = AlphaPublicationService(
                _data_dir(request),
                request.app.state.strategy_engine,
            ).publish(candidate_id)
            champion = champion_store.promote(candidate_id, publication["strategy_id"])
        return {"publication": publication, "champion": champion}
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events")
def stream_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    store = _manager(request).store
    _required_manifest(store, run_id)
    try:
        cursor = max(int(last_event_id or 0), 0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer") from exc

    async def generate() -> AsyncIterator[dict[str, str]]:
        nonlocal cursor
        while not await request.is_disconnected():
            events = await asyncio.to_thread(store.read_events, run_id, after_id=cursor)
            for event in events:
                cursor = int(event["id"])
                yield {
                    "id": str(cursor),
                    "event": str(event["type"]),
                    "data": json.dumps(event["payload"], ensure_ascii=False),
                }
            manifest = await asyncio.to_thread(store.get, run_id)
            if manifest is None or manifest.get("status") in TERMINAL_RUN_STATUSES:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(generate())


def _manager(request: Request):
    manager = getattr(request.app.state, "alpha_mining_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Alpha mining manager is unavailable")
    return manager


def _required_manifest(store, run_id: str) -> dict[str, Any]:
    try:
        manifest = store.get(run_id)
    except MiningRunValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if manifest is None:
        raise HTTPException(status_code=404, detail="Alpha run not found")
    return manifest


def _project_run(store, manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary = store.read_summary(str(manifest["run_id"]))
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "request": manifest["request"],
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "error": manifest.get("error"),
        "progress": summary.get("progress"),
        "research_state": summary.get("research_state"),
    }


def _attach_failure_lineage(manager, worker_request: dict[str, Any]) -> None:
    source_run_id = worker_request.get("source_run_id")
    suggestion_id = worker_request.get("source_suggestion_id")
    source_candidate_id = worker_request.get("source_candidate_id")
    if source_run_id is None and suggestion_id is None and source_candidate_id is None:
        worker_request.pop("source_run_id", None)
        worker_request.pop("source_suggestion_id", None)
        worker_request.pop("source_candidate_id", None)
        return
    source = manager.store.get(str(source_run_id))
    if source is None or source.get("status") not in _SUCCESS:
        raise ValueError("研究来源不存在或尚未完成")
    source_result = manager.store.read_summary(str(source_run_id))
    if source_candidate_id is not None:
        _validate_strict_candidate_lineage(
            manager,
            worker_request,
            candidate_id=str(source_candidate_id),
        )
        return
    suggestions = source_result.get("next_research_suggestions")
    suggestion = next(
        (
            row for row in suggestions
            if isinstance(row, Mapping) and row.get("suggestion_id") == suggestion_id
        ),
        None,
    ) if isinstance(suggestions, list) else None
    if suggestion is None:
        raise ValueError("失败研究建议不存在，不能建立新实验血缘")
    patch = suggestion.get("request_patch")
    if not isinstance(patch, Mapping) or not patch:
        raise ValueError("失败研究建议没有可执行差异")
    for field, expected in patch.items():
        if worker_request.get(field) != expected:
            raise ValueError(f"新实验没有采用建议中的字段差异: {field}")
    source_request = source.get("request")
    if not isinstance(source_request, Mapping):
        raise ValueError("失败研究来源缺少冻结请求")
    ignored = {
        "champion_strategy_id", "source_run_id", "source_suggestion_id",
        "source_candidate_id", "source_diff",
    }
    fields = sorted((set(source_request) | set(worker_request)) - ignored)
    diff = {
        field: {"before": source_request.get(field), "after": worker_request.get(field)}
        for field in fields
        if source_request.get(field) != worker_request.get(field)
    }
    if not diff:
        raise ValueError("新实验与失败研究来源没有任何差异")
    worker_request["source_diff"] = diff


def _attach_hypothesis_contract(manager, worker_request: dict[str, Any]) -> None:
    hypothesis_id = worker_request.get("hypothesis_id")
    if not hypothesis_id:
        worker_request.pop("hypothesis_id", None)
        return
    try:
        hypothesis = AlphaHypothesisStore(manager._data_dir).get(str(hypothesis_id))
    except KeyError as exc:
        raise ValueError("Alpha假设不存在") from exc
    spec = hypothesis["test_spec"]
    expected = {
        "asset_type": hypothesis["asset_type"],
        "forward_horizon": hypothesis["forward_horizon"],
        "engine_ids": spec["engine_ids"],
        "factor_names": spec["factor_names"],
    }
    for field, value in expected.items():
        if worker_request.get(field) != value:
            raise ValueError(f"研究请求偏离冻结Alpha假设: {field}")
    worker_request["hypothesis_contract"] = hypothesis


def _validate_strict_candidate_lineage(
    manager,
    worker_request: dict[str, Any],
    *,
    candidate_id: str,
) -> None:
    evidence = manager.evidence.get_candidate(candidate_id)
    if evidence["state"]["state"] != "validation_candidate":
        raise ValueError("严格验证来源候选不处于待严格验证状态")
    if evidence["run_id"] != worker_request.get("source_run_id"):
        raise ValueError("严格验证来源候选与来源运行不一致")
    source = manager.store.get(str(evidence["run_id"]))
    source_request = source.get("request") if isinstance(source, Mapping) else None
    if not isinstance(source_request, Mapping):
        raise ValueError("严格验证来源缺少冻结请求")
    dates = enriched_partition_dates(
        manager._data_dir,
        str(source_request.get("asset_type") or "stock"),
    )
    expected = {
        "engine_ids": [evidence["engine_id"]],
        "factor_names": source_request.get("factor_names"),
        "budget_profile": "strict",
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "max_candidates_per_engine": 8,
        "max_trials_per_engine": 128,
    }
    for field, value in expected.items():
        if worker_request.get(field) != value:
            raise ValueError(f"严格验证合同字段不符合冻结升级规则: {field}")
    ignored = {
        "champion_strategy_id", "source_run_id", "source_suggestion_id",
        "source_candidate_id", "source_diff",
    }
    fields = sorted((set(source_request) | set(worker_request)) - ignored)
    diff = {
        field: {"before": source_request.get(field), "after": worker_request.get(field)}
        for field in fields
        if source_request.get(field) != worker_request.get(field)
    }
    if not diff:
        raise ValueError("严格验证运行与来源合同没有差异")
    worker_request["source_diff"] = diff


def _validate_engines(
    engine_ids: list[str],
    asset_type: str,
    *,
    data_dir,
    start: date,
    end: date,
    forward_horizon: int,
) -> None:
    catalog = AlphaResearchDataCatalog(data_dir).snapshot(start, end, asset_type)
    rows = {
        row["engine_id"]: row
        for row in _engine_availability(
            asset_type,
            start,
            end,
            forward_horizon,
            catalog,
        )
    }
    for engine_id in engine_ids:
        engine = _REGISTRY.get(engine_id)
        if engine.manifest.readiness != "ready":
            raise ValueError(f"Alpha engine is not ready: {engine_id}")
        if asset_type not in engine.manifest.asset_types:
            raise ValueError(f"Alpha engine {engine_id} does not support {asset_type}")
        row = rows[engine_id]
        if not row["ready"]:
            raise ValueError(
                f"Alpha engine {engine_id} data gate failed: "
                + "; ".join(row["reasons"])
            )


def _validate_factor_datasets(factor_names: list[str], catalog: Any) -> None:
    selected_financial = sorted(set(factor_names) & _FINANCIAL_FACTOR_IDS)
    if selected_financial:
        qualification = catalog.datasets["financial_pit"]
        if not qualification.ready:
            reasons = "; ".join(qualification.reasons) or "公告时点财务数据未就绪"
            raise ValueError(
                "所选财务因子不能进入正式历史研究: "
                + ", ".join(selected_financial)
                + "; "
                + reasons
            )
    selected_share = sorted(set(factor_names) & _SHARE_HISTORY_FACTOR_IDS)
    if selected_share:
        qualification = catalog.datasets["share_history_pit"]
        if not qualification.ready:
            reasons = "; ".join(qualification.reasons) or "点时股本数据未就绪"
            raise ValueError(
                "所选换手率因子不能进入正式历史研究: "
                + ", ".join(selected_share)
                + "; "
                + reasons
            )


def _engine_availability(
    asset_type: str,
    start: date,
    end: date,
    forward_horizon: int,
    catalog,
) -> list[dict[str, Any]]:
    available_features = list(_FACTOR_IDS)
    if catalog.datasets["industry_pit"].ready:
        available_features.extend(["industry_momentum_20d", "industry_breadth_5d"])
    if catalog.datasets["event_history"].ready:
        available_features.extend(["event_count_20d", "event_direction_20d"])
    output = []
    for engine in _REGISTRY.list():
        context = DataCatalogContext(
            asset_type=asset_type,
            start=start.isoformat(),
            end=end.isoformat(),
            available_features=tuple(available_features),
            datasets=catalog.datasets,
        )
        manifest_gate = qualify_manifest_datasets(engine.manifest, catalog.datasets)
        engine_gate = engine.preflight(context)
        reasons = list(dict.fromkeys((*manifest_gate.reasons, *engine_gate.reasons)))
        if forward_horizon not in engine.manifest.forecast_horizons:
            reasons.append(f"不支持{forward_horizon}日预测期限")
        output.append({
            "engine_id": engine.manifest.engine_id,
            "ready": not reasons,
            "reasons": reasons,
            "observations": {**manifest_gate.observations, **engine_gate.observations},
        })
    return output


def _data_dir(request: Request):
    return request.app.state.repo.store.data_dir


def _data_fingerprint(request: Request, worker_request: Mapping[str, Any]) -> dict[str, Any]:
    repo = request.app.state.repo
    asset_type = str(worker_request.get("asset_type") or "stock")
    generation = repo.get_matrix_data_generation(asset_type)
    components = {
        "version": "alpha-data-v1",
        "asset_type": asset_type,
        "generation": generation,
        "latest_enriched_date": (
            repo.latest_enriched_date(asset_type).isoformat()
            if repo.latest_enriched_date(asset_type) is not None else None
        ),
        "factor_methodology_version": FACTOR_METHODOLOGY_VERSION,
        "engines": [
            {
                "engine_id": _REGISTRY.get(engine_id).manifest.engine_id,
                "version": _REGISTRY.get(engine_id).manifest.version,
            }
            for engine_id in worker_request.get("engine_ids") or []
        ],
    }
    raw = json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**components, "digest": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
