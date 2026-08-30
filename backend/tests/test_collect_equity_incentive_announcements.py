# ruff: noqa: RUF001
from __future__ import annotations

import importlib.util
import io
import json
from datetime import date, datetime
from http.client import IncompleteRead
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT = (
    Path(__file__).resolve().parents[2] / "research" / "collect_equity_incentive_announcements.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_equity_incentive_announcements", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(announcement_id: str) -> dict:
    timestamp = int(
        datetime(2019, 1, 2, 18, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1000
    )
    return {
        "secCode": "300001",
        "secName": "测试公司",
        "orgId": "org-1",
        "announcementId": announcement_id,
        "announcementTitle": "<em>限制性股票激励计划</em>（草案）",
        "announcementTime": timestamp,
        "adjunctUrl": "finalpage/test.pdf",
        "columnId": "column",
        "announcementType": "type",
    }


def test_fetch_year_reads_all_pages_once() -> None:
    calls = []

    def fetch(payload):
        calls.append(payload["pageNum"])
        page = int(payload["pageNum"])
        rows = [_row(f"A{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 2:
            rows = rows[:1]
        return {
            "totalAnnouncement": 31,
            "totalpages": 2,
            "announcements": rows,
        }

    rows = collector.fetch_year(fetch, 2019)

    assert len(rows) == 31
    assert calls == ["1", "2"]


def test_fetch_year_accepts_cninfo_zero_based_last_page_count() -> None:
    calls = []

    def fetch(payload):
        calls.append(payload["pageNum"])
        page = int(payload["pageNum"])
        rows = [_row(f"B{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 2:
            rows = rows[:1]
        return {
            "totalAnnouncement": 31,
            "totalpages": 1,
            "announcements": rows,
        }

    rows = collector.fetch_year(fetch, 2019)

    assert len(rows) == 31
    assert calls == ["1", "2"]


def test_normalize_uses_beijing_date_strips_html_and_deduplicates() -> None:
    row = _row("A1")

    frame = collector.normalize([row, row], 2019)

    assert frame.height == 1
    assert frame["ann_date"][0] == date(2019, 1, 2)
    assert frame["symbol"][0] == "300001.SZ"
    assert frame["title"][0] == "限制性股票激励计划（草案）"


def test_client_retries_incomplete_chunk_without_changing_payload(monkeypatch) -> None:
    calls = []

    def urlopen(request, *, timeout):
        calls.append((request.data, timeout))
        if len(calls) == 1:
            raise IncompleteRead(b'{"totalAnnouncement":')
        payload = {
            "totalAnnouncement": 0,
            "totalpages": 0,
            "announcements": [],
        }
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(collector.urllib.request, "urlopen", urlopen)
    client = collector.CNInfoClient(
        min_interval=0,
        max_attempts=2,
        retry_base_seconds=0,
    )

    result = client.fetch(collector._payload(2019, 1))

    assert result["totalAnnouncement"] == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
