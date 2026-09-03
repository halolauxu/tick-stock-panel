"""Incremental Tushare post-market datasets.

Opening/closing auction data is partitioned by trade date. IR Q&A is fetched by
publish time (rather than question date) so an answer published today is not
missed when the original question was asked earlier. Both datasets use a short
lookback and idempotent merge to cover holidays, late publication and retries
without re-downloading their full history.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import TushareProvider

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], None]


def sync_auction(
    data_dir: Path,
    *,
    end_date: date,
    lookback_days: int = 3,
    on_progress: ProgressCb | None = None,
) -> int:
    """Sync opening and closing auction rows for a short calendar lookback."""
    start_date = end_date - timedelta(days=max(1, lookback_days) - 1)
    return sync_auction_range(
        data_dir,
        start_date=start_date,
        end_date=end_date,
        on_progress=on_progress,
    )


def sync_auction_range(
    data_dir: Path,
    *,
    start_date: date,
    end_date: date,
    on_progress: ProgressCb | None = None,
) -> int:
    """Backfill auction history one date/session at a time.

    A daily partition is merged immediately after each response, so a retry is
    idempotent and a long range never accumulates all rows in memory.
    """
    days = _date_range(start_date, end_date)
    work = [(day, session) for day in days for session in ("open", "close")]
    provider = TushareProvider()
    fetched = 0
    try:
        for index, (day, session) in enumerate(work, start=1):
            frame = provider.get_auction(day, session)
            fetched += frame.height
            _merge_partitions(
                data_dir / "tushare_supplemental" / "auction",
                frame,
                unique_by=["symbol", "date", "session"],
                sort_by=["date", "symbol", "session"],
            )
            if on_progress is not None:
                label = "开盘" if session == "open" else "收盘"
                on_progress(index, len(work), f"{day.isoformat()} {label}竞价")
    finally:
        provider.close()
    logger.info(
        "tushare auction sync: [%s ~ %s] fetched=%d",
        start_date,
        end_date,
        fetched,
    )
    return fetched


def sync_irm_qa(
    data_dir: Path,
    *,
    end_date: date,
    lookback_days: int = 3,
    on_progress: ProgressCb | None = None,
) -> int:
    """Sync newly published SH/SZ investor-relations Q&A."""
    start_date = end_date - timedelta(days=max(1, lookback_days) - 1)
    return sync_irm_qa_range(
        data_dir,
        start_date=start_date,
        end_date=end_date,
        on_progress=on_progress,
    )


def sync_irm_qa_range(
    data_dir: Path,
    *,
    start_date: date,
    end_date: date,
    on_progress: ProgressCb | None = None,
) -> int:
    """Backfill IR Q&A by publish date and exchange.

    Tushare limits each IR response to 3,000 rows. Fetching one publish date at
    a time makes truncation explicit (the provider raises at the limit) while
    retaining late answers whose original question date is older.
    """
    days = _date_range(start_date, end_date)
    exchanges = ("sh", "sz")
    work = [(day, exchange) for day in days for exchange in exchanges]
    provider = TushareProvider()
    fetched = 0
    try:
        for index, (day, exchange) in enumerate(work, start=1):
            frame = provider.get_irm_qa(
                exchange,
                pub_start=day,
                pub_end=day,
            )
            fetched += frame.height
            _merge_partitions(
                data_dir / "tushare_supplemental" / "irm_qa",
                frame,
                unique_by=["symbol", "pub_time", "question"],
                sort_by=["date", "pub_time", "symbol"],
            )
            if on_progress is not None:
                label = "上证E互动" if exchange == "sh" else "深证互动易"
                on_progress(index, len(work), f"{day.isoformat()} {label}")
    finally:
        provider.close()
    logger.info(
        "tushare irm qa sync: [%s ~ %s] fetched=%d",
        start_date,
        end_date,
        fetched,
    )
    return fetched


def sync_forecast(
    data_dir: Path,
    *,
    end_date: date,
    lookback_days: int = 3,
    on_progress: ProgressCb | None = None,
) -> int:
    """Sync newly announced earnings forecasts and publish a coverage receipt."""
    start_date = end_date - timedelta(days=max(1, lookback_days) - 1)
    days = _date_range(start_date, end_date)
    provider = TushareProvider()
    fetched = 0
    try:
        for index, day in enumerate(days, start=1):
            frame = provider.get_forecast(day)
            fetched += frame.height
            _merge_forecast_year(data_dir, frame, day.year)
            if on_progress is not None:
                on_progress(index, len(days), day.isoformat())
    finally:
        provider.close()
    receipt = {
        "schema_version": "tushare-forecast-sync-v1",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rows_returned": fetched,
    }
    _atomic_json(
        data_dir / "event_data" / "forecast" / "sync_status.json",
        receipt,
    )
    logger.info(
        "tushare forecast sync: [%s ~ %s] fetched=%d",
        start_date,
        end_date,
        fetched,
    )
    return fetched


def _date_range(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    return [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]


def _merge_partitions(
    root: Path,
    frame: pl.DataFrame,
    *,
    unique_by: list[str],
    sort_by: list[str],
) -> None:
    if frame.is_empty():
        return
    for day_frame in frame.partition_by("date", maintain_order=True):
        day = str(day_frame["date"][0])
        out = root / f"date={day}" / "part.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged = day_frame
        if out.exists():
            existing = pl.read_parquet(out)
            merged = pl.concat([existing, day_frame], how="diagonal_relaxed")
        merged = merged.unique(subset=unique_by, keep="last").sort(sort_by)
        temporary = out.with_name(out.name + ".tmp")
        merged.write_parquet(temporary)
        temporary.replace(out)


def _merge_forecast_year(data_dir: Path, frame: pl.DataFrame, year: int) -> None:
    if frame.is_empty():
        return
    out = data_dir / "event_data" / "forecast" / f"year={year}" / "part.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = frame
    if out.exists():
        merged = pl.concat([pl.read_parquet(out), frame], how="diagonal_relaxed")
    merged = merged.unique(
        subset=[
            "symbol",
            "ann_date",
            "period_end",
            "type",
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
        ],
        keep="last",
    ).sort(["ann_date", "symbol", "period_end", "type"])
    temporary = out.with_name(out.name + ".tmp")
    merged.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(out)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
