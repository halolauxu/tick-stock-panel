from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "collect_restructuring_announcements.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_restructuring_announcements", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(art_code: str, stock_code: str = "600001") -> dict:
    return {
        "art_code": art_code,
        "notice_date": "2020-01-02 00:00:00",
        "title": "测试公司:关于筹划重大资产重组的停牌公告",
        "codes": [
            {
                "ann_type": "A,SHA",
                "stock_code": stock_code,
                "short_name": "测试公司",
            }
        ],
        "columns": [{"column_code": "x", "column_name": "重组进展公告"}],
    }


def test_fetch_month_reads_every_page_once() -> None:
    calls = []

    def fetch(params):
        calls.append(params["page_index"])
        page = int(params["page_index"])
        rows = [_row(f"A{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 2:
            rows = rows[:1]
        return {"success": 1, "data": {"total_hits": 101, "list": rows}}

    rows = collector.fetch_month(fetch, 2020, 1)

    assert len(rows) == 101
    assert calls == ["1", "2"]


def test_normalize_expands_a_share_codes_and_deduplicates() -> None:
    row = _row("A1")
    row["codes"].extend(
        [
            {"ann_type": "A,CYB", "stock_code": "300001", "short_name": "测试二"},
            {"ann_type": "B", "stock_code": "900001", "short_name": "B股"},
        ]
    )

    frame = collector.normalize([row, row], 2020)

    assert frame.height == 2
    assert set(frame["symbol"].to_list()) == {"600001.SH", "300001.SZ"}
    assert frame["ann_date"].to_list() == [date(2020, 1, 2), date(2020, 1, 2)]
