"""模拟交易 API。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.paper_trading import (
    PaperTradingStore,
    PaperTradingStoreError,
    run_account,
)
from app.strategy.engine import StrategyDataContext, StrategyEngine
from app.tickflow.capabilities import Cap

router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])


class PaperTradingCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    strategy_id: str = Field(..., min_length=1, max_length=120)
    asset_type: Literal["stock", "etf"] = "stock"
    symbols: list[str] | None = None
    params: dict | None = None
    overrides: dict | None = None
    entry_fill: Literal["close_t", "open_t+1"] = "open_t+1"
    exit_fill: Literal["close_t", "open_t+1", "signal_next_minute"] = "open_t+1"
    commission_pct: float = Field(0.0002, ge=0, le=0.02)
    stamp_tax_pct: float = Field(0.001, ge=0, le=0.02)
    slippage_bps: float = Field(5.0, ge=0, le=1000)
    max_positions: int = Field(10, ge=1, le=100)
    max_exposure_pct: float = Field(1.0, gt=0, le=1)
    initial_capital: float = Field(1_000_000, ge=10_000, le=1_000_000_000)
    position_sizing: Literal["equal", "score_weight"] = "equal"
    holding_days: int = Field(5, ge=1, le=1000)
    minute_fill: bool = False
    regime_filter: dict | None = None
    enforce_t_plus_one: bool = True


def _store(request: Request) -> PaperTradingStore:
    return PaperTradingStore(request.app.state.repo.store.data_dir)


def _raise_store_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, PaperTradingStoreError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/accounts")
def list_accounts(request: Request) -> dict:
    try:
        return {"items": _store(request).list()}
    except Exception as exc:
        _raise_store_error(exc)


@router.post("/accounts")
def create_account(body: PaperTradingCreateRequest, request: Request) -> dict:
    repo = request.app.state.repo
    latest = repo.latest_enriched_date(body.asset_type)
    if latest is None:
        raise HTTPException(status_code=400, detail="缺少完整指标数据, 请先完成日K采集与指标计算")
    if body.entry_fill != "open_t+1":
        raise HTTPException(status_code=400, detail="盘后模拟不能倒填信号日收盘建仓, 仅支持次日开盘")
    if body.exit_fill == "close_t":
        raise HTTPException(status_code=400, detail="盘后收盘信号不能按同一收盘价平仓, 请使用次日开盘")
    try:
        strategy = request.app.state.strategy_engine.get(body.strategy_id)
        StrategyEngine.validate_context(
            strategy,
            StrategyDataContext(asset_type=body.asset_type, timeframe="1d", as_of=latest),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.minute_fill:
        capset = request.app.state.capabilities
        if not capset.has(Cap.KLINE_MINUTE_BATCH):
            raise HTTPException(status_code=403, detail="分钟K成交需要 Pro+ 权限")
    if body.exit_fill == "signal_next_minute" and not body.minute_fill:
        raise HTTPException(status_code=400, detail="触发后下一分钟成交需要先开启分钟K成交")

    config = body.model_dump(exclude={"name"})
    config["strategy_name"] = str(getattr(strategy, "name", body.strategy_id))
    try:
        return _store(request).create(name=body.name, start_date=latest, config=config)
    except Exception as exc:
        _raise_store_error(exc)


@router.get("/accounts/{account_id}")
def get_account(account_id: str, request: Request) -> dict:
    try:
        return _store(request).get(account_id)
    except Exception as exc:
        _raise_store_error(exc)


@router.delete("/accounts/{account_id}")
def delete_account(account_id: str, request: Request) -> dict:
    try:
        deleted = _store(request).delete(account_id)
        return {"ok": True, "id": deleted["id"], "name": deleted["name"]}
    except Exception as exc:
        _raise_store_error(exc)


@router.post("/accounts/{account_id}/run")
def run_paper_account(account_id: str, request: Request) -> dict:
    try:
        # 暂停只关闭每日自动运行, 用户主动同步仍应按最新完整数据重算。
        return run_account(request.app.state, account_id, force=True)
    except Exception as exc:
        _raise_store_error(exc)


@router.post("/accounts/{account_id}/pause")
def pause_account(account_id: str, request: Request) -> dict:
    try:
        return _store(request).set_status(account_id, "paused")
    except Exception as exc:
        _raise_store_error(exc)


@router.post("/accounts/{account_id}/resume")
def resume_account(account_id: str, request: Request) -> dict:
    try:
        return _store(request).set_status(account_id, "active")
    except Exception as exc:
        _raise_store_error(exc)
