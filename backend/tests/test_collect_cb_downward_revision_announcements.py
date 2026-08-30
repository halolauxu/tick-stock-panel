from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_cb_downward_revision_announcements.py"
    )
    spec = importlib.util.spec_from_file_location("collect_cb_downward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(identifier: str, title: str) -> dict:
    timestamp = int(datetime(2020, 1, 2, tzinfo=UTC).timestamp() * 1000)
    return {
        "announcementId": identifier,
        "announcementTime": timestamp,
        "secCode": "600001",
        "secName": "样本公司",
        "announcementTitle": title,
    }


def test_normalize_classifies_proposal_and_excludes_no_revision() -> None:
    frame = collector.normalize(
        [
            _row("a", "董事会提议向下修正可转换公司债券转股价格的公告"),
            _row("b", "关于向下修正可转换公司债券转股价格的公告"),
            _row("c", "关于暂不向下修正可转换公司债券转股价格的公告"),
        ],
        2020,
    )

    assert frame["phase"].to_list() == ["implemented", "proposal"]
    assert frame["announcement_id"].to_list() == ["b", "a"]


def test_fetch_year_accepts_cninfo_off_by_one_page_count() -> None:
    rows = [_row(str(index), "关于向下修正转股价格的公告") for index in range(31)]

    def fetch(payload: dict[str, str]) -> dict:
        page = int(payload["pageNum"])
        start = (page - 1) * collector.PAGE_SIZE
        end = start + collector.PAGE_SIZE
        return {
            "announcements": rows[start:end],
            "totalAnnouncement": len(rows),
            "totalpages": 1,
        }

    assert len(collector.fetch_year(fetch, 2020)) == 31
