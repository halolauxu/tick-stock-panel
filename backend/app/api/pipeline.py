"""盘后管道 API — 异步触发 + 进度跟踪。"""
from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging
from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api.data import invalidate_storage_cache
from app.jobs import daily_pipeline
from app.services.pipeline_jobs import (
    JobCancelledError,
    job_store,
    release_run_slot,
    try_acquire_run_slot,
)

# 长时间任务专用线程池（隔离于 FastAPI 默认线程池，防止阻塞请求处理）
_long_task_executor = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="long-task")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class SupplementalBackfillRequest(BaseModel):
    dataset: Literal["auction", "irm_qa"]
    start_date: date
    end_date: date


@router.post("/run")
async def run_now(request: Request) -> dict:
    """异步触发盘后管道,立即返回 job_id。客户端轮询 /jobs/{id} 拿进度。

    若已有任务在跑,**返回该任务 id 而不是开新任务**(防止并发拉数据撞限流)。
    卡死判定按「进度停滞」而非总时长(慢带宽下长任务不会被误杀), 见 reap_stale。
    """
    repo = request.app.state.repo
    capset = request.app.state.capabilities

    # 检测卡死的 running job (如 reload 后孤儿 task / 网络读无限阻塞)。
    # reap_stale 会在 /run 和 /jobs/{id} 轮询端点都调用,保证卡死后能自愈。
    job_store.reap_stale()

    # 单飞: 复用任何活跃 (pending∨running) 任务, is_new=False 时不再调度新任务
    job_id, is_new = job_store.create()
    if not is_new:
        return {"job_id": job_id, "reused": True}

    # 在 executor 里跑同步任务(pipeline 内部都是阻塞 IO + CPU)
    async def task() -> None:
        # 重任务执行槽: 防僵尸并发(reap 后线程仍活时新任务不得并行写 parquet)
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务在运行(或上一次任务卡死未结束),请稍后再试")
            return
        # 管道运行期间暂停实时行情取数, 防止覆写同一批 parquet 竞态
        qs = getattr(request.app.state, "quote_service", None)
        try:
            job_store.start(job_id)
            loop = asyncio.get_event_loop()

            def progress(stage: str, pct: int, msg: str, stage_pct: int | None = None,
                         skip_log: bool = False) -> None:
                job_store.progress(job_id, stage, pct, msg, stage_pct=stage_pct, skip_log=skip_log)

            def _run() -> dict:
                if qs:
                    with qs.paused():
                        return daily_pipeline.run_now(repo, capset, on_progress=progress)
                return daily_pipeline.run_now(repo, capset, on_progress=progress)

            result = await loop.run_in_executor(_long_task_executor, _run)
            job_store.succeed(job_id, result)
            invalidate_storage_cache()
            repo.refresh_cache()  # 刷新 Polars 缓存
            # 数据与内存视图均完成后再推进模拟账户; 失败与数据任务隔离, 次日可重试。
            try:
                from app.services.paper_trading import run_active_accounts

                summary = await loop.run_in_executor(
                    _long_task_executor,
                    run_active_accounts,
                    request.app.state,
                )
                logger.info("manual pipeline paper trading result: %s", summary)
            except Exception:
                logger.exception("manual pipeline paper trading failed; data job remains succeeded")
        except JobCancelledError:
            # 已被 reap/手动取消终止: job 状态已由 terminate() 写为 failed,
            # 拉取线程在分块回调处自行退出, 这里无需(也无法)再写状态。
            logger.warning("pipeline job %s cancelled", job_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("pipeline failed")
            job_store.fail(job_id, str(e))
            invalidate_storage_cache()
        finally:
            release_run_slot(job_id)

    asyncio.create_task(task())
    return {"job_id": job_id, "reused": False}


@router.post("/tushare-supplemental/backfill")
async def backfill_tushare_supplemental(
    request: Request,
    body: SupplementalBackfillRequest,
) -> dict:
    """Backfill one Tushare supplemental dataset over an explicit date range."""
    from app.api.data import invalidate_data_cache
    from app.market_time import cn_today
    from app.services import tushare_supplemental_sync

    if body.start_date > body.end_date:
        raise HTTPException(status_code=422, detail="开始日期不能晚于结束日期")
    if body.end_date > cn_today():
        raise HTTPException(status_code=422, detail="结束日期不能晚于今天")
    days = (body.end_date - body.start_date).days + 1
    if days > 3660:
        raise HTTPException(status_code=422, detail="单次补采最多 10 年,请分段执行")

    job_store.reap_stale()
    job_id, is_new = job_store.create(long_running=True)
    if not is_new:
        return {"job_id": job_id, "reused": True}

    dataset = body.dataset
    start_date = body.start_date
    end_date = body.end_date
    data_dir = request.app.state.repo.store.data_dir
    stage = f"backfill_{dataset}"
    dataset_label = "集合竞价" if dataset == "auction" else "董秘问答"

    async def task() -> None:
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务在运行(或上一次任务卡死未结束),请稍后再试")
            return
        try:
            job_store.start(job_id)
            job_store.progress(
                job_id,
                stage,
                2,
                f"准备补采 {dataset_label}: {start_date} 至 {end_date}",
                stage_pct=0,
            )

            def on_progress(done: int, total: int, label: str) -> None:
                stage_pct = int(done / max(total, 1) * 100)
                pct = 2 + int(stage_pct * 0.96)
                job_store.progress(
                    job_id,
                    stage,
                    pct,
                    f"{label} · {done}/{total}",
                    stage_pct=stage_pct,
                    skip_log=True,
                )

            def run() -> int:
                fn = (
                    tushare_supplemental_sync.sync_auction_range
                    if dataset == "auction"
                    else tushare_supplemental_sync.sync_irm_qa_range
                )
                return fn(
                    data_dir,
                    start_date=start_date,
                    end_date=end_date,
                    on_progress=on_progress,
                )

            rows = await asyncio.get_event_loop().run_in_executor(_long_task_executor, run)
            invalidate_data_cache(dataset)
            job_store.progress(
                job_id,
                "done",
                100,
                f"{dataset_label}补采完成,本次接口返回 {rows} 行",
                stage_pct=100,
            )
            job_store.succeed(job_id, {
                "dataset": dataset,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                f"{dataset}_rows": rows,
            })
        except JobCancelledError:
            logger.warning("supplemental backfill job %s cancelled", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("supplemental backfill failed")
            job_store.fail(job_id, str(exc))
            invalidate_data_cache(dataset)
        finally:
            release_run_slot(job_id)

    asyncio.create_task(task())
    return {
        "job_id": job_id,
        "reused": False,
        "dataset": dataset,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    # 每次轮询都检查卡死 job — 前端持续轮询, 进度停滞超阈值后必定自愈,
    # 无需用户再次手动点「同步」。
    job_store.reap_stale()
    j = job_store.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return j


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """手动取消一个 running 的 job(协作式: 拉取线程在当前分块完成后自行退出)。"""
    j = job_store.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    if j["status"] not in ("running", "pending"):
        raise HTTPException(status_code=400, detail=f"job status is {j['status']}, cannot cancel")
    job_store.terminate(job_id, "用户手动取消")
    return {"cancelled": job_id}


@router.get("/jobs")
def list_jobs(limit: int = 20) -> dict:
    return {
        "active_id": job_store.active_id(),
        "jobs": job_store.list_recent(limit=limit),
    }
