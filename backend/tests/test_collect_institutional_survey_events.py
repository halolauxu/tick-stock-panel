from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_institutional_survey_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_institutional_survey_events", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(object_code: str, *, object_type: str = "001") -> dict:
    return {
        "SECUCODE": "000001.SZ",
        "NOTICE_DATE": "2020-01-23 00:00:00",
        "RECEIVE_START_DATE": "2020-01-10 00:00:00",
        "RECEIVE_END_DATE": "2020-01-10 00:00:00",
        "RECEIVE_OBJECT_TYPE": object_type,
        "RECEIVE_OBJECT": f"机构{object_code}",
        "OBJECT_CODE": object_code,
        "SUM": 3,
        "ORG_TYPE": "证券公司",
        "URL": "AN1",
    }


def test_fetch_month_reads_exact_pages() -> None:
    calls = []

    def fetch(params):
        calls.append(params["pageNumber"])
        page = int(params["pageNumber"])
        rows = [_row(f"{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 2:
            rows = rows[:1]
        return {
            "result": {"count": 501, "pages": 2, "data": rows},
            "success": True,
        }

    rows = collector.fetch_month(fetch, 2020, 1)

    assert len(rows) == 501
    assert calls == ["1", "2"]


def test_normalize_counts_unique_institutions_and_excludes_noninstitution() -> None:
    rows = [_row("A"), _row("A"), _row("B"), _row("person", object_type="002")]

    frame = collector.normalize(rows, 2020, 1)

    assert frame.height == 1
    assert frame["notice_date"][0] == date(2020, 1, 23)
    assert frame["institution_count"][0] == 2
    assert frame["institution_detail_rows"][0] == 3
    assert frame["provider_sum_max"][0] == 3
