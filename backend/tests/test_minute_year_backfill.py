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
