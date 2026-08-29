"""Independent Alpha mining API. The legacy ``/api/backtest/mining`` API is untouched."""
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

from app.alpha_mining.champion import AlphaChampionStore
from app.alpha_mining.config_store import AlphaConfigStore
from app.alpha_mining.contracts import DataCatalogContext, qualify_manifest_datasets
from app.alpha_mining.data_catalog import AlphaResearchDataCatalog
from app.alpha_mining.evidence import AlphaEvidenceError, AlphaEvidenceStore
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
        champion_strategy_id = str(champion["strategy_id"])
        if payload.champion_strategy_id not in (None, champion_strategy_id):
            raise ValueError("请求冠军与动态冠军账本不一致")
        request.app.state.strategy_engine.get(champion_strategy_id)
        worker_request = payload.model_dump(mode="json", exclude={"force"})
        worker_request["champion_strategy_id"] = champion_strategy_id
        dates = enriched_partition_dates(
            request.app.state.repo.store.data_dir,
            payload.asset_type,
            payload.start,
            payload.end,
        )
        validation = _validation_config(payload.budget_profile, payload.forward_horizon)
        if len(dates) < required_trading_bars(validation, 1):
            raise ValueError("所选区间不足以形成一个隔离的外层样本外窗口")
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
        publication = AlphaPublicationService(
            _data_dir(request),
            request.app.state.strategy_engine,
        ).publish(candidate_id)
        champion = AlphaChampionStore(_data_dir(request)).promote(
            candidate_id,
            publication["strategy_id"],
        )
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
