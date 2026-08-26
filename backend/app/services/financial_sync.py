"""财务数据独立同步服务。

解耦于 K-line 管道, 自有调度 + 自有存储。
能力门控: Cap.FINANCIAL (Expert 套餐)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from app.tickflow.capabilities import Cap, CapabilitySet

logger = logging.getLogger(__name__)

# 每个 API 请求最多 100 个标的
_BATCH_SIZE = 100

# 财务报表 + 历史股本表
FINANCIAL_TABLES = ("metrics", "income", "balance_sheet", "cash_flow", "shares")
FinancialProgressCallback = Callable[[int, int, int, int], None]
_BEIJING_TZ = ZoneInfo("Asia/Shanghai")


# ================================================================
# 同步函数
# ================================================================


def _get_symbols(data_dir: Path) -> list[str]:
    """从 instruments 表获取标的列表。"""
    inst_path = data_dir / "instruments" / "instruments.parquet"
    if not inst_path.exists():
        return []
    try:
        df = pl.read_parquet(inst_path, columns=["symbol"])
        return df["symbol"].to_list()
    except Exception as e:
        logger.warning("读取 instruments 失败: %s", e)
        return []


def _financial_is_custom() -> bool:
    """当前财务数据源是否走 custom (用于绕过 TickFlow Expert 套餐门槛)。"""
    from app.services import preferences

    provider = preferences.get_financial_provider()
    if provider == "tickflow":
        return False
    from app.data_providers import custom as custom_sources

    return custom_sources.provider_has_dataset(provider, "financial")


def _fetch_table(
    table: str,
    symbols: list[str],
    capset: CapabilitySet,
    latest_only: bool = True,
    on_progress: FinancialProgressCallback | None = None,
) -> pl.DataFrame:
    """通过当前财务数据源拉取一张标准化财务表。"""
    is_custom = _financial_is_custom()
    if not is_custom and not capset.has(Cap.FINANCIAL):
        logger.info("sync_%s skipped: no FINANCIAL capability", table)
        return pl.DataFrame()
    if not symbols:
        logger.warning("sync_%s skipped: no symbols", table)
        return pl.DataFrame()

    # 自定义数据源分流
    if is_custom:
        from app.data_providers import custom as custom_sources
        from app.services import preferences

        try:
            provider = custom_sources.get_provider(preferences.get_financial_provider())
            method = provider.get_financials
            parameters = inspect.signature(method).parameters.values()
            supports_progress = any(
                parameter.name == "on_progress" or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if supports_progress:
                df = method(
                    table,
                    symbols,
                    latest_only=latest_only,
                    on_progress=on_progress,
                )
            else:
                # 兼容尚未升级签名的第三方财务插件。旧插件仍可同步, 只在返回后
                # 给出一次完成态进度, 不强制其立即实现新回调。
                df = method(table, symbols, latest_only=latest_only)
                if on_progress is not None:
                    on_progress(len(symbols), len(symbols), len(df), 0)
        except Exception as e:
            logger.warning("sync_%s custom provider failed: %s", table, e)
            return pl.DataFrame()
        if df.is_empty() or "symbol" not in df.columns:
            return pl.DataFrame()
        return df

    from app.tickflow.client import get_client

    tf = get_client()

    # 分批拉取
    api_method = {
        "metrics": tf.financials.metrics,
        "income": tf.financials.income,
        "balance_sheet": tf.financials.balance_sheet,
        "cash_flow": tf.financials.cash_flow,
        "shares": getattr(tf.financials, "shares", None),
    }[table]
    if api_method is None:
        logger.warning("sync_shares skipped: current TickFlow SDK does not support shares")
        return pl.DataFrame()

    all_records: list[dict] = []
    failures = 0
    total_batches = (len(symbols) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(symbols), _BATCH_SIZE):
        chunk = symbols[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        try:
            data = api_method(chunk, latest=latest_only)
            # data 格式: { "600519.SH": [record, ...], ... }
            if isinstance(data, dict):
                for sym, records in data.items():
                    if isinstance(records, list):
                        for rec in records:
                            if isinstance(rec, dict):
                                rec["symbol"] = sym
                                all_records.append(rec)
            logger.debug(
                "sync_%s batch %d/%d: %d records",
                table,
                batch_num,
                total_batches,
                len(data) if isinstance(data, dict) else 0,
            )
        except Exception as e:
            failures += len(chunk)
            logger.warning("sync_%s batch %d/%d failed: %s", table, batch_num, total_batches, e)
        if on_progress is not None:
            on_progress(
                min(i + len(chunk), len(symbols)),
                len(symbols),
                len(all_records),
                failures,
            )

    if not all_records:
        return pl.DataFrame()

    df = pl.DataFrame(all_records)
    if df.is_empty() or "symbol" not in df.columns:
        return pl.DataFrame()
    return df


def _write_table(table: str, df: pl.DataFrame, data_dir: Path) -> int:
    if df.is_empty() or "symbol" not in df.columns:
        return 0

    # 写入 Parquet (全量覆盖)
    out_dir = data_dir / "financials" / table
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part.parquet"
    df.write_parquet(out_file)

    logger.info("sync_%s done: %d records written", table, len(df))
    return len(df)


def _sync_table(
    table: str,
    symbols: list[str],
    data_dir: Path,
    capset: CapabilitySet,
    latest_only: bool = True,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步单张财务表。返回写入的行数。"""
    if on_progress is None:
        frame = _fetch_table(table, symbols, capset, latest_only=latest_only)
    else:
        frame = _fetch_table(
            table,
            symbols,
            capset,
            latest_only=latest_only,
            on_progress=on_progress,
        )
    return _write_table(
        table,
        frame,
        data_dir,
    )


def _merge_report_history(*frames: pl.DataFrame) -> pl.DataFrame:
    valid = [
        frame
        for frame in frames
        if not frame.is_empty() and {"symbol", "period_end"} <= set(frame.columns)
    ]
    if not valid:
        return pl.DataFrame()
    merged = pl.concat(valid, how="diagonal_relaxed").filter(
        pl.col("symbol").is_not_null() & pl.col("period_end").is_not_null()
    )
    # 同一 (symbol, period_end) 多条时保留 announce_date 最新一条 (业绩修正以最新公告为准)。
    if "announce_date" in merged.columns:
        merged = merged.sort(["symbol", "period_end", "announce_date"], nulls_last=True)
    return merged.unique(subset=["symbol", "period_end"], keep="last").sort(
        ["symbol", "period_end"]
    )


def _sync_history_table_for_symbols(
    table: str,
    symbols: list[str],
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """历史累积同步: 保留已有各期记录, 仅拉最新期 + 为新标的补全量历史。

    与 shares 同一模式。若改为 latest_only 全量覆盖, 历史各期会在每次同步时
    被冲掉, 财务因子将永远只有单期快照, 任何回测都是未来函数。
    """
    existing = get_financial_df(data_dir, table)
    if existing.is_empty() or not {"symbol", "period_end"} <= set(existing.columns):
        return _sync_table(
            table,
            symbols,
            data_dir,
            capset,
            latest_only=False,
            on_progress=on_progress,
        )

    existing_symbols = set(existing["symbol"].drop_nulls().to_list())
    missing_symbols = [symbol for symbol in symbols if symbol not in existing_symbols]
    current_symbols = [symbol for symbol in symbols if symbol in existing_symbols]

    # 缺失标的补历史、已有标的拉最新是两个 provider 调用。把两个阶段的回调
    # 合并成同一条单调递增进度, 避免第二阶段从 0 开始导致页面“倒退”。
    processed_offset = 0
    row_offset = 0
    failure_offset = 0

    def fetch_phase(phase_symbols: list[str], *, latest_only: bool) -> pl.DataFrame:
        nonlocal processed_offset, row_offset, failure_offset
        if not phase_symbols:
            return pl.DataFrame()
        phase_rows = 0
        phase_failures = 0
        progress_seen = False

        def phase_progress(done: int, _total: int, rows: int, failures: int) -> None:
            nonlocal phase_rows, phase_failures, progress_seen
            progress_seen = True
            phase_rows = rows
            phase_failures = failures
            if on_progress is not None:
                on_progress(
                    processed_offset + done,
                    len(symbols),
                    row_offset + rows,
                    failure_offset + failures,
                )

        if on_progress is None:
            frame = _fetch_table(
                table,
                phase_symbols,
                capset,
                latest_only=latest_only,
            )
        else:
            frame = _fetch_table(
                table,
                phase_symbols,
                capset,
                latest_only=latest_only,
                on_progress=phase_progress,
            )
        processed_offset += len(phase_symbols)
        row_offset += phase_rows if progress_seen else len(frame)
        failure_offset += phase_failures
        return frame

    missing_history = fetch_phase(missing_symbols, latest_only=False)
    latest = fetch_phase(current_symbols, latest_only=True)
    merged = _merge_report_history(existing, missing_history, latest)
    return _write_table(table, merged, data_dir)


def sync_metrics(
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步核心财务指标 (metrics), 历史各期累积保留。"""
    symbols = _get_symbols(data_dir)
    return _sync_history_table_for_symbols(
        "metrics", symbols, data_dir, capset, on_progress=on_progress
    )


def sync_income(
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步利润表, 历史各期累积保留。"""
    symbols = _get_symbols(data_dir)
    return _sync_history_table_for_symbols(
        "income", symbols, data_dir, capset, on_progress=on_progress
    )


def sync_balance_sheet(
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步资产负债表, 历史各期累积保留。"""
    symbols = _get_symbols(data_dir)
    return _sync_history_table_for_symbols(
        "balance_sheet", symbols, data_dir, capset, on_progress=on_progress
    )


def sync_cash_flow(
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步现金流量表, 历史各期累积保留。"""
    symbols = _get_symbols(data_dir)
    return _sync_history_table_for_symbols(
        "cash_flow", symbols, data_dir, capset, on_progress=on_progress
    )


def sync_shares(
    data_dir: Path,
    capset: CapabilitySet,
    on_progress: FinancialProgressCallback | None = None,
) -> int:
    """同步历史股本表。"""
    symbols = _get_symbols(data_dir)
    return _sync_history_table_for_symbols(
        "shares", symbols, data_dir, capset, on_progress=on_progress
    )


def sync_all(data_dir: Path, capset: CapabilitySet) -> dict[str, int]:
    """同步所有财务表。返回 {table: rows}。"""
    if not capset.has(Cap.FINANCIAL) and not _financial_is_custom():
        logger.info("sync_all financials skipped: no FINANCIAL capability")
        return {}

    symbols = _get_symbols(data_dir)
    results: dict[str, int] = {}
    for table in FINANCIAL_TABLES:
        results[table] = _sync_history_table_for_symbols(table, symbols, data_dir, capset)

    # 同步完成后注册 DuckDB 视图
    _refresh_financials_views(data_dir)

    return results


# ================================================================
# DuckDB 视图
# ================================================================


def _refresh_financials_views(data_dir: Path) -> None:
    """刷新财务表 DuckDB 视图 (在 DataStore.db 上注册)。"""
    d = data_dir.as_posix()
    views = {
        "financials_metrics": f"{d}/financials/metrics/*.parquet",
        "financials_income": f"{d}/financials/income/*.parquet",
        "financials_balance_sheet": f"{d}/financials/balance_sheet/*.parquet",
        "financials_cash_flow": f"{d}/financials/cash_flow/*.parquet",
        "financials_shares": f"{d}/financials/shares/*.parquet",
    }
    for name, _path in views.items():
        out = data_dir / "financials" / name.replace("financials_", "") / "part.parquet"
        if not out.exists():
            continue
        # 视图注册需要由 DataStore 完成,这里只做日志
        logger.debug("financial parquet ready: %s (%d rows)", name, out.stat().st_size)


def get_financial_df(data_dir: Path, table: str) -> pl.DataFrame:
    """读取本地财务 Parquet。"""
    path = data_dir / "financials" / table / "part.parquet"
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception as e:
        logger.warning("读取 financials/%s 失败: %s", table, e)
        return pl.DataFrame()


# ================================================================
# 调度器
# ================================================================


class FinancialScheduler:
    """独立调度器: 每周同步 metrics, 财务表支持手动同步。"""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._data_dir: Path | None = None
        self._capset: CapabilitySet | None = None
        self._lock = threading.Lock()
        self._last_sync: dict[str, str] = {}  # {table: iso_timestamp}
        # 手动同步(run_now)是否正在进行。前端据此显示"同步中"并防重复点击。
        self._is_syncing = False
        # 当前同步范围与正在处理的表。不能只暴露全局 bool: 否则前端会把五张表
        # 全部渲染成“下载中”, 页面刷新后也无法知道真正运行的是哪一张。
        self._sync_scope: str | None = None  # single / all
        self._active_table: str | None = None
        self._sync_progress: dict[str, int] | None = None

    def start(self, data_dir: Path, capset: CapabilitySet, *, auto_schedule: bool = False) -> None:
        """初始化调度器, 并按需启动周期同步后台任务。

        auto_schedule=False (默认): 仅初始化 (设置数据目录/能力 + 恢复 last_sync),
            供 /api/financials/sync/* 手动同步使用, 不启动自动调度。
        auto_schedule=True: 额外启动每周一次的 metrics 自动同步 (启动后 60s 首跑)。
        """
        # 先记录 data_dir/capset, 即使当前无 FINANCIAL 也保留引用:
        # 用户稍后在「设置」页升级到 Expert Key 时, update_capabilities() 会把新 capset
        # 推进来,trigger()/run_now() 才能用上 FINANCIAL。否则 _capset 永远是 None,
        # 即便 app.state.capabilities 已更新, 调度器仍报 "no FINANCIAL capability"。
        self._data_dir = data_dir
        self._capset = capset
        if not capset.has(Cap.FINANCIAL) and not _financial_is_custom():
            logger.info("FinancialScheduler skipped: no FINANCIAL capability")
            return
        # 从持久化恢复上次同步时间: 重启后前端仍能显示真实最后同步时间,而非"尚未同步"
        try:
            from app.services import preferences

            restored = dict(preferences.get_financial_sync_times())
            # 老用户迁移兜底: 若某表在 preferences 无记录但 parquet 已存在(升级前同步过),
            # 用 parquet 文件的修改时间作为同步时间并补写持久化。
            for table in FINANCIAL_TABLES:
                if table in restored:
                    continue
                parquet = data_dir / "financials" / table / "part.parquet"
                if parquet.exists():
                    mtime = datetime.fromtimestamp(parquet.stat().st_mtime, tz=UTC).isoformat()
                    restored[table] = mtime
                    preferences.set_financial_sync_time(table, mtime)
                    logger.info(
                        "FinancialScheduler backfilled last_sync for %s from parquet mtime", table
                    )
            self._last_sync = restored
            if self._last_sync:
                logger.info(
                    "FinancialScheduler restored last_sync: %s", list(self._last_sync.keys())
                )
        except Exception as e:
            logger.warning("restore financial_sync_times failed: %s", e)

        if not auto_schedule:
            # 仅初始化 (手动同步用), 不启动周期任务。
            logger.info("FinancialScheduler initialized (auto-schedule disabled; manual sync only)")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("FinancialScheduler started (auto-schedule enabled)")

    def _record_sync(self, table: str) -> None:
        """记录一张表的同步完成时间: 更新内存 + 持久化到 preferences.json。

        持久化确保即使重启,前端 /status 仍返回真实的最后同步时间,
        不会错误地显示"尚未同步"。
        """
        ts = datetime.now(UTC).isoformat()
        self._last_sync[table] = ts
        try:
            from app.services import preferences

            preferences.set_financial_sync_time(table, ts)
        except Exception as e:
            logger.warning("persist financial_sync_time(%s) failed: %s", e)

    def _is_fresh_today(self, table: str, *, now: datetime | None = None) -> bool:
        """已有落盘数据且今天成功同步过时,避免按钮重复扫描全市场。"""
        if not self._data_dir:
            return False
        parquet = self._data_dir / "financials" / table / "part.parquet"
        if not parquet.exists() or parquet.stat().st_size == 0:
            return False
        raw = self._last_sync.get(table)
        if not raw:
            return False
        try:
            synced_at = datetime.fromisoformat(raw)
            if synced_at.tzinfo is None:
                synced_at = synced_at.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return False
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return synced_at.astimezone(_BEIJING_TZ).date() == current.astimezone(_BEIJING_TZ).date()

    def _tables_due_today(self, table: str | None) -> list[str]:
        requested = [table] if table else list(FINANCIAL_TABLES)
        return [name for name in requested if not self._is_fresh_today(name)]

    def update_capabilities(self, capset: CapabilitySet) -> None:
        """刷新调度器持有的能力集。

        用户在「设置」页新增/清除 API Key 后, settings API 会重新探测能力并更新
        app.state.capabilities; 必须同步推给本调度器, 否则 trigger()/run_now() 仍读
        启动时的旧 capset, 即便 app.state 已含 FINANCIAL, 调度器仍报
        "no FINANCIAL capability" 而拒绝同步 (表现为前端「全部同步」按钮闪一下无动作)。
        """
        prev = self._capset
        self._capset = capset
        had = bool(prev) and prev.has(Cap.FINANCIAL)
        now = capset.has(Cap.FINANCIAL)
        if had != now:
            logger.info("FinancialScheduler capabilities updated: FINANCIAL %s -> %s", had, now)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("FinancialScheduler stopped")

    async def _run_loop(self) -> None:
        """每周执行一次 metrics 同步。"""
        try:
            while self._running:
                # 首次启动等 60s, 之后每 7 天执行一次
                await asyncio.sleep(60)
                if not self._running:
                    break

                # 每周: 只同步 metrics
                try:
                    rows = sync_metrics(self._data_dir, self._capset)
                    self._record_sync("metrics")
                    logger.info("FinancialScheduler: metrics synced, %d rows", rows)
                except Exception as e:
                    logger.warning("FinancialScheduler: metrics sync failed: %s", e)

                # 等待下一次 (7天)
                for _ in range(7 * 24 * 60):  # 每分钟检查一次 _running
                    if not self._running:
                        break
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            pass

    def _run_body(self, table: str | None) -> dict[str, int]:
        """同步逻辑本体(不加锁,假设调用方已持有 _is_syncing)。

        table=None 同步全部财务表;否则只同步指定表。
        每张表完成立即更新 last_sync,让前端轮询 /status 能看到进度递增。
        """
        if table:
            self._set_active_table(table)
            fn = {
                "metrics": sync_metrics,
                "income": sync_income,
                "balance_sheet": sync_balance_sheet,
                "cash_flow": sync_cash_flow,
                "shares": sync_shares,
            }.get(table)
            if not fn:
                return {}
            rows = fn(
                self._data_dir,
                self._capset,
                on_progress=self._progress_callback(table),
            )
            self._record_sync(table)
            return {table: rows}
        # 全部同步只处理今天尚未成功同步的表,避免“全部同步”重复扫描已完成表。
        symbols = _get_symbols(self._data_dir)
        result: dict[str, int] = {}
        for t in self._tables_due_today(None):
            self._set_active_table(t)
            result[t] = _sync_history_table_for_symbols(
                t,
                symbols,
                self._data_dir,
                self._capset,
                on_progress=self._progress_callback(t),
            )
            self._record_sync(t)
        _refresh_financials_views(self._data_dir)
        return result

    def _begin_sync(self, table: str | None) -> None:
        """在已持有 ``_lock`` 时记录一次同步的服务端真值。"""
        self._is_syncing = True
        self._sync_scope = "single" if table else "all"
        self._active_table = table or FINANCIAL_TABLES[0]
        self._sync_progress = {
            "symbols_done": 0,
            "symbols_total": 0,
            "rows_received": 0,
            "failures": 0,
        }

    def _set_active_table(self, table: str) -> None:
        """切换当前表; 供全量同步逐表推进时更新前端状态。"""
        with self._lock:
            if self._active_table != table:
                self._sync_progress = {
                    "symbols_done": 0,
                    "symbols_total": 0,
                    "rows_received": 0,
                    "failures": 0,
                }
            self._active_table = table

    def _progress_callback(self, table: str) -> FinancialProgressCallback:
        """创建仅更新当前表的线程安全进度回调。"""

        def update(done: int, total: int, rows: int, failures: int) -> None:
            with self._lock:
                if not self._is_syncing or self._active_table != table:
                    return
                self._sync_progress = {
                    "symbols_done": max(0, int(done)),
                    "symbols_total": max(0, int(total)),
                    "rows_received": max(0, int(rows)),
                    "failures": max(0, int(failures)),
                }

        return update

    def _finish_sync(self) -> None:
        """在已持有 ``_lock`` 时清理同步状态。"""
        self._is_syncing = False
        self._sync_scope = None
        self._active_table = None
        self._sync_progress = None

    def run_now(self, table: str | None = None) -> dict[str, int]:
        """同步执行一次同步(阻塞调用线程)。

        ⚠ 全量同步需数分钟,务必在后台线程调用,不要直接在 HTTP 请求线程里阻塞,
        否则请求会长时间 pending 直至被浏览器/代理超时掐断(表现为"点击无反应")。
        HTTP 接口应调用 trigger() 立即返回,再让前端轮询 /status.syncing 看进度。

        用 _is_syncing 标志防并发:若已有同步在进行,本次直接跳过,
        避免重复请求拖慢服务端 / 触发上游限流。
        """
        if not self._capset or (not self._capset.has(Cap.FINANCIAL) and not _financial_is_custom()):
            return {}
        with self._lock:
            if self._is_syncing:
                logger.info("financial sync skipped: already running")
                return {"_skipped": 1}
            if not self._tables_due_today(table):
                logger.info("financial sync skipped: already up to date: table=%s", table or "all")
                return {"_skipped": 1, "_up_to_date": 1}
            self._begin_sync(table)
        try:
            return self._run_body(table)
        finally:
            with self._lock:
                self._finish_sync()

    def trigger(self, table: str | None = None) -> dict[str, int]:
        """触发一次同步(非阻塞,立即返回)。

        在后台线程执行同步体,HTTP 请求无需等待。
        返回 {"started": True/False}:
          - False = 能力不足或已有同步在进行(被防并发跳过)
          - True  = 已在后台开始,前端应轮询 /status.syncing 观察进度

        ⚠ _is_syncing 在此处置 True(持锁),确保 trigger 返回时前端轮询
        /status 已能看到 syncing=True,无竞态窗口;同时防止快速重复点击
        启动多个后台线程。后台线程复用 _run_body 执行真正的同步逻辑。
        """
        if not self._capset or (not self._capset.has(Cap.FINANCIAL) and not _financial_is_custom()):
            return {"started": False, "reason": "no FINANCIAL capability"}
        with self._lock:
            if self._is_syncing:
                logger.info("financial sync trigger skipped: already running")
                return {"started": False, "reason": "already running"}
            if not self._tables_due_today(table):
                logger.info(
                    "financial sync trigger skipped: already up to date: table=%s",
                    table or "all",
                )
                return {
                    "started": False,
                    "reason": "already up to date",
                    "tables": [table] if table else list(FINANCIAL_TABLES),
                }
            # 持锁置位:保证 trigger 返回前 syncing 已为 True
            self._begin_sync(table)

        def _bg() -> None:
            try:
                self._run_body(table)
            except Exception as e:
                logger.exception("background financial sync failed: %s", e)
            finally:
                with self._lock:
                    self._finish_sync()

        t = threading.Thread(target=_bg, name="financial-sync", daemon=True)
        t.start()
        logger.info("financial sync triggered in background: table=%s", table or "all")
        return {"started": True}

    @property
    def is_syncing(self) -> bool:
        """手动同步是否正在进行(供 /status 返回,前端据此显示"同步中")。"""
        with self._lock:
            return self._is_syncing

    @property
    def sync_state(self) -> dict[str, object]:
        """返回一次加锁读取的同步快照, 避免 API 拼出不一致的状态。"""
        with self._lock:
            return {
                "syncing": self._is_syncing,
                "sync_scope": self._sync_scope,
                "syncing_table": self._active_table,
                "sync_progress": dict(self._sync_progress) if self._sync_progress else None,
            }

    @property
    def last_sync(self) -> dict[str, str]:
        return dict(self._last_sync)


# 全局单例
financial_scheduler = FinancialScheduler()
