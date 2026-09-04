from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import kline as kline_api
from app.services import kline_sync
from app.services import pipeline_jobs


class _Repo:
    def __init__(self, data_dir, dates: list[date]) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)
        self._dates = dates

    def execute_all(self, _sql: str, _params: list[str]):
        return [(value,) for value in self._dates]


def test_recent_year_window_fetches_only_missing_trading_span(monkeypatch, tmp_path):
    dates = [date(2025, 8, 27), date(2025, 8, 28), date(2026, 8, 26)]
    repo = _Repo(tmp_path, dates)
    monkeypatch.setattr(
        kline_sync,
        "minute_coverage_summary",
        lambda _data_dir: {
            "dates": [
                {"date": "2025-08-27", "complete": False},
                {"date": "2025-08-28", "complete": True},
                {"date": "2026-08-26", "complete": True},
            ],
        },
    )

    window = kline_sync._minute_missing_window(
        repo, date(2025, 8, 27), date(2026, 8, 27),
    )

    assert window == (
        datetime(2025, 8, 27),
        datetime(2025, 8, 28),
    )


def test_recent_year_window_is_idempotent_when_complete(monkeypatch, tmp_path):
    dates = [date(2025, 8, 27), date(2026, 8, 26)]
    repo = _Repo(tmp_path, dates)
    monkeypatch.setattr(
        kline_sync,
        "minute_coverage_summary",
        lambda _data_dir: {
            "dates": [
                {"date": "2025-08-27", "complete": True},
                {"date": "2026-08-26", "complete": True},
            ],
        },
    )

    assert kline_sync._minute_missing_window(
        repo, date(2025, 8, 27), date(2026, 8, 27),
    ) is None


def test_recent_year_window_falls_back_to_full_calendar_range(monkeypatch, tmp_path):
    repo = _Repo(tmp_path, [])
    monkeypatch.setattr(kline_sync, "minute_coverage_summary", lambda _data_dir: None)

    window = kline_sync._minute_missing_window(
        repo, date(2025, 8, 27), date(2026, 8, 27),
    )

    assert window == (
        datetime(2025, 8, 27),
        datetime(2026, 8, 28),
    )


def test_targeted_minute_retry_never_expands_end_date_to_today(monkeypatch, tmp_path):
    """补偿任务按原交易日执行，不能在次日盘中混入尚未收盘的分钟线。"""
    repo = _Repo(tmp_path, [date(2026, 9, 3)])
    captured = {}

    monkeypatch.setattr(kline_sync.preferences, "get_minute_data_provider", lambda: "mock")
    monkeypatch.setattr(kline_sync, "_resolve_minute_provider", lambda _name: (object(), False, None))
    monkeypatch.setattr(kline_sync, "_cleanup_null_datetime_minute", lambda _repo: None)
    monkeypatch.setattr(kline_sync, "_migrate_symbol_to_date_partition", lambda _repo: None)
    monkeypatch.setattr(
        kline_sync,
        "_minute_missing_window",
        lambda _repo, start, end: (
            captured.update({"target_start": start, "target_end": end})
            or (datetime(2026, 9, 3), datetime(2026, 9, 4))
        ),
    )
    monkeypatch.setattr(
        kline_sync,
        "resolve_limit",
        lambda *_args, **_kwargs: SimpleNamespace(batch=100, rpm=60),
    )
    monkeypatch.setattr(kline_sync.preferences, "get_minute_sync_segment_days", lambda: 20)

    def fake_sync(_symbols, *, start_time, end_time, **_kwargs):
        captured.update({"fetch_start": start_time, "fetch_end": end_time})
        return None

    monkeypatch.setattr(kline_sync, "sync_minute_batch", fake_sync)

    written = kline_sync.sync_and_persist_minute(
        ["600000.SH"],
        repo,
        SimpleNamespace(has=lambda _cap: False),
        target_start_date=date(2026, 9, 3),
        target_end_date=date(2026, 9, 3),
    )

    assert written == 0
    assert captured["target_start"] == date(2026, 9, 3)
    assert captured["target_end"] == date(2026, 9, 3)
    assert captured["fetch_end"] == datetime(2026, 9, 4)


def test_manual_minute_sync_rejects_unrelated_active_job(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(repo=object(), capabilities=object()),
        ),
        json=lambda: None,
    )

    async def request_json():
        return {"recent_year": True}

    request.json = request_json
    monkeypatch.setattr(kline_api, "_minute_allowed", lambda _capset: True)
    monkeypatch.setattr(
        pipeline_jobs.job_store,
        "create",
        lambda **_kwargs: ("already-running", False),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(kline_api.sync_minute(request))

    assert exc_info.value.status_code == 409
    assert "已有数据同步任务" in str(exc_info.value.detail)


def test_minute_progress_explains_segment_and_symbol_denominators():
    message = kline_sync.format_minute_progress(
        56,
        61_083,
        5_553,
        "2025-08-27~2025-09-24",
    )

    assert message == (
        "日期分段 1/11 · 当前分段标的 56/5553 只 · "
        "日期范围 2025-08-27~2025-09-24"
    )

    assert kline_sync.format_minute_progress(
        5_554,
        61_083,
        5_553,
        "2025-09-25~2025-10-22",
    ).startswith("日期分段 2/11 · 当前分段标的 1/5553 只")


def test_minute_progress_labels_non_symbol_work_as_provider_requests():
    assert kline_sync.format_minute_progress(
        3,
        56,
        5_553,
        "2026-08-24~2026-08-27",
    ) == "数据源请求 3/56 · 日期范围 2026-08-24~2026-08-27"
