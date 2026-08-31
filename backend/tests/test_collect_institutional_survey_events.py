from __future__ import annotations

import importlib.util
import json
import stat
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "collect_institutional_survey_events.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_institutional_survey_events", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_warmup_year_is_allowed_but_earlier_history_is_rejected() -> None:
    collector.validate_period(2013, 1)

    with pytest.raises(ValueError, match="2013-2026"):
        collector.validate_period(2012, 12)


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
        if page == 3:
            rows = rows[:1]
        return {
            "result": {"count": 101, "pages": 3, "data": rows},
            "success": True,
        }

    rows = collector.fetch_month(fetch, 2020, 1)

    assert len(rows) == 101
    assert calls == ["1", "2", "3"]


def test_request_uses_verified_wide_report_page_size() -> None:
    params = collector._params(2020, 1, 1)

    assert params["pageSize"] == "50"
    assert collector.SOURCE_PAGE_LIMIT == 100
    assert params["sortColumns"] == collector.SORT_COLUMNS
    assert params["sortTypes"] == collector.SORT_TYPES_ASC
    assert params["sortColumns"].split(",") == [
        "NOTICE_DATE",
        "SECUCODE",
        "URL",
        "OBJECT_CODE",
        "RECEIVE_START_DATE",
        "RECEIVE_END_DATE",
        "RECEIVE_OBJECT_TYPE",
        "RECEIVE_OBJECT",
        "ORG_TYPE",
        "SUM",
    ]
    assert params["source"] == "WEB"
    assert params["client"] == "WEB"


def test_client_disables_persistent_chunked_transport(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"success": True, "result": {"data": []}})

    monkeypatch.setattr(collector.urllib.request, "urlopen", fake_urlopen)
    client = collector.EastmoneySurveyClient(min_interval=0)

    payload = client.fetch(collector._params(2020, 1, 1))

    headers = {key.lower(): value for key, value in captured["request"].headers.items()}
    assert payload["success"] is True
    assert headers["connection"] == "close"
    assert headers["accept-encoding"] == "identity"


def test_client_surfaces_provider_error_context(monkeypatch) -> None:
    def fake_urlopen(_request, *, timeout):
        del timeout
        return _Response({"success": False, "code": 9501, "message": "bad filter"})

    monkeypatch.setattr(collector.urllib.request, "urlopen", fake_urlopen)
    client = collector.EastmoneySurveyClient(min_interval=0)

    with pytest.raises(ValueError, match=r"code=9501.*bad filter.*NOTICE_DATE"):
        client.fetch(collector._params(2020, 1, 1))


def test_client_normalizes_provider_empty_result(monkeypatch) -> None:
    def fake_urlopen(_request, *, timeout):
        del timeout
        return _Response({"success": False, "code": 9201, "message": "返回数据为空"})

    monkeypatch.setattr(collector.urllib.request, "urlopen", fake_urlopen)
    client = collector.EastmoneySurveyClient(min_interval=0)

    payload = client.fetch(collector._params(2013, 3, 1))

    assert payload["success"] is True
    assert payload["result"] == {"count": 0, "pages": 0, "data": []}


def test_fetch_month_reuses_valid_page_cache(tmp_path) -> None:
    calls = []

    def fetch(params):
        calls.append(params["pageNumber"])
        page = int(params["pageNumber"])
        rows = [_row(f"{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 3:
            rows = rows[:1]
        return {
            "result": {"count": 101, "pages": 3, "data": rows},
            "success": True,
        }

    first = collector.fetch_month(fetch, 2020, 1, cache_dir=tmp_path)
    calls.clear()
    second = collector.fetch_month(fetch, 2020, 1, cache_dir=tmp_path)

    assert len(first) == len(second) == 101
    assert calls == ["1"]


def test_fetch_month_rejects_cache_from_old_sort_contract(tmp_path) -> None:
    calls = []

    def fetch(params):
        calls.append(params["pageNumber"])
        return {
            "result": {"count": 1, "pages": 1, "data": [_row("fresh")]},
            "success": True,
        }

    cache_path = tmp_path / "stable-ascending" / "page=0001.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "year": 2020,
                "month": 1,
                "page": 1,
                "count": 1,
                "pages": 1,
                "rows": [_row("stale")],
            }
        )
    )

    rows = collector.fetch_month(fetch, 2020, 1, cache_dir=tmp_path)

    assert calls == ["1"]
    assert rows[0]["OBJECT_CODE"] == "fresh"
    payload = json.loads(cache_path.read_text())
    assert payload["cache_schema_version"] == collector.PAGE_CACHE_SCHEMA_VERSION
    assert payload["sort_columns"] == collector.SORT_COLUMNS


def test_fetch_month_splits_source_deep_pagination_into_daily_shards(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(collector, "SOURCE_PAGE_LIMIT", 2)

    def fetch(params):
        filter_value = params["filter"]
        page = int(params["pageNumber"])
        if "2020-01-01')(NOTICE_DATE<='2020-01-31" in filter_value:
            return {
                "result": {"count": 101, "pages": 3, "data": [_row("month")] * 50},
                "success": True,
            }
        if "2020-01-01')(NOTICE_DATE<='2020-01-01" in filter_value:
            rows = [_row(f"day1-{page}-{index}") for index in range(50)]
            if page == 2:
                rows = rows[:1]
            return {
                "result": {"count": 51, "pages": 2, "data": rows},
                "success": True,
            }
        if "2020-01-02')(NOTICE_DATE<='2020-01-02" in filter_value:
            return {
                "result": {
                    "count": 50,
                    "pages": 1,
                    "data": [_row(f"day2-{index}") for index in range(50)],
                },
                "success": True,
            }
        return {"result": {"count": 0, "pages": 0, "data": []}, "success": True}

    rows = collector.fetch_month(fetch, 2020, 1, cache_dir=tmp_path)

    assert len(rows) == 101
    assert (tmp_path / "day=2020-01-01" / "stable-ascending" / "page=0002.json").is_file()


def test_daily_shard_over_page_limit_uses_bidirectional_stable_paging(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(collector, "SOURCE_PAGE_LIMIT", 2)

    canonical = [_row(f"row-{index:03d}") for index in range(101)]

    def fetch(params):
        filter_value = params["filter"]
        page = int(params["pageNumber"])
        if "2020-01-01')(NOTICE_DATE<='2020-01-31" in filter_value:
            return {
                "result": {"count": 101, "pages": 3, "data": canonical[:50]},
                "success": True,
            }
        if "2020-01-01')(NOTICE_DATE<='2020-01-01" not in filter_value:
            return {"result": {"count": 0, "pages": 0, "data": []}, "success": True}
        descending = params["sortTypes"].startswith("-1")
        ordered = list(reversed(canonical)) if descending else canonical
        start = (page - 1) * collector.PAGE_SIZE
        return {
            "result": {
                "count": 101,
                "pages": 3,
                "data": ordered[start : start + collector.PAGE_SIZE],
            },
            "success": True,
        }

    rows = collector.fetch_month(fetch, 2020, 1, cache_dir=tmp_path)

    assert len(rows) == 101
    assert {row["OBJECT_CODE"] for row in rows} == {row["OBJECT_CODE"] for row in canonical}
    day_cache = tmp_path / "day=2020-01-01"
    assert (day_cache / "ascending" / "page=0002.json").is_file()
    assert (day_cache / "descending" / "page=0002.json").is_file()


def test_normalize_counts_unique_institutions_and_excludes_noninstitution() -> None:
    rows = [_row("A"), _row("A"), _row("B"), _row("person", object_type="002")]

    frame = collector.normalize(rows, 2020, 1)

    assert frame.height == 1
    assert frame["notice_date"][0] == date(2020, 1, 23)
    assert frame["institution_count"][0] == 2
    assert frame["institution_detail_rows"][0] == 3
    assert frame["provider_sum_max"][0] == 3


def test_collect_month_writes_world_readable_atomic_partition(tmp_path) -> None:
    def fetch(_params):
        return {
            "result": {"count": 1, "pages": 1, "data": [_row("A")]},
            "success": True,
        }

    result = collector.collect_month(fetch, tmp_path, 2020, 1)
    path = Path(result["path"])

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_collect_month_persists_auditable_empty_partition(tmp_path) -> None:
    def fetch(_params):
        return {
            "result": {"count": 0, "pages": 0, "data": []},
            "success": True,
        }

    result = collector.collect_month(fetch, tmp_path, 2013, 6)
    frame = collector.pl.read_parquet(result["path"])

    assert result["raw_rows"] == result["events"] == result["symbols"] == 0
    assert result["first_notice_date"] is None
    assert frame.schema == collector.EVENT_SCHEMA
