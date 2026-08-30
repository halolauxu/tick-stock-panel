from __future__ import annotations

import importlib.util
import stat
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_analyst_report_history.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_analyst_report_history", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(report_id: str) -> dict:
    return {
        "infoCode": report_id,
        "publishDate": "2020-01-31 00:00:00.000",
        "stockCode": "002156",
        "market": "SHENZHEN",
        "stockName": "测试股份",
        "orgCode": "80000031",
        "orgSName": "测试证券",
        "researcher": "研究员",
        "author": ["1.研究员"],
        "emRatingName": "买入",
        "lastEmRatingName": "增持",
        "indvAimPriceT": "15.0",
        "indvAimPriceL": "13.0",
        "reportType": 2,
        "indvIsNew": "001",
        "title": "测试报告",
    }


def test_fetch_year_reads_exact_pages() -> None:
    calls = []

    def fetch(params):
        calls.append(params["pageNumber"])
        page = int(params["pageNumber"])
        rows = [_row(f"R{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 3:
            rows = rows[:1]
        return {"hits": 201, "TotalPage": 3, "data": rows}

    rows = collector.fetch_year(fetch, 2020)

    assert len(rows) == 201
    assert calls == ["1", "2", "3"]


def test_normalize_maps_symbol_types_and_deduplicates_report_id() -> None:
    frame = collector.normalize([_row("R1"), _row("R1")], 2020)

    assert frame.height == 1
    assert frame["publish_date"][0] == date(2020, 1, 31)
    assert frame["symbol"][0] == "002156.SZ"
    assert frame["target_price_high"][0] == 15.0
    assert frame["target_price_low"][0] == 13.0


def test_collect_year_writes_readable_partition(tmp_path) -> None:
    def fetch(_params):
        return {"hits": 1, "TotalPage": 1, "data": [_row("R1")]}

    result = collector.collect_year(fetch, tmp_path, 2020)
    path = Path(result["path"])

    assert result["reports"] == 1
    assert result["target_price_reports"] == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
