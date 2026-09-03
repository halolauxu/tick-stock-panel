from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import duckdb

SCRIPT = (
    Path(__file__).resolve().parents[2] / "research" / "collect_p0_short_horizon_event_documents.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_short_horizon_event_documents", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selects_matching_first_forecast_document() -> None:
    study = _load_module()
    rows = [
        {
            "announcement_id": "1",
            "announce_time": None,
            "title": "关于召开股东大会的通知",
            "pdf_url": "https://example.invalid/1.pdf",
            "announced_size_kb": 10.0,
        },
        {
            "announcement_id": "2",
            "announce_time": None,
            "title": "2019年度业绩预增公告",
            "pdf_url": "https://example.invalid/2.pdf",
            "announced_size_kb": 20.0,
        },
        {
            "announcement_id": "3",
            "announce_time": None,
            "title": "2019年度业绩预告修正公告",
            "pdf_url": "https://example.invalid/3.pdf",
            "announced_size_kb": 30.0,
        },
    ]

    selected = study.select_forecast_document(rows, date(2019, 12, 31))

    assert selected is not None
    assert selected["announcement_id"] == "2"

    embedded = study.select_forecast_document(
        [
            {
                "announcement_id": "4",
                "announce_time": None,
                "title": "2019年年度报告摘要",
                "pdf_url": "https://example.invalid/4.pdf",
                "announced_size_kb": 20.0,
            },
            {
                "announcement_id": "5",
                "announce_time": None,
                "title": "2019年年度报告",
                "pdf_url": "https://example.invalid/5.pdf",
                "announced_size_kb": 100.0,
            },
        ],
        date(2020, 3, 31),
    )
    assert embedded is not None
    assert embedded["announcement_id"] == "5"


def test_extracts_reason_section_without_later_boilerplate() -> None:
    study = _load_module()
    text = """
    一、本期业绩预计情况
    公司预计净利润同比增长。
    二、业绩变动原因说明
    公司主营产品销量增长,订单交付增加,营业收入同比提升。
    三、其他相关说明
    本次业绩预告未经审计,公司收到政府补助情况请见其他公告。
    """

    reason = study.extract_reason_section(text)

    assert "主营产品销量增长" in reason
    assert "政府补助" not in reason
    assert study.classify_reason(reason) == "OPERATING"


def test_event_key_is_stable_for_null_numeric_fields() -> None:
    study = _load_module()
    event = {
        "symbol": "000001.SZ",
        "ann_date": date(2020, 1, 1),
        "period_end": date(2019, 12, 31),
        "type": "预增",
        "p_change_min": None,
        "p_change_max": 100.0,
        "net_profit_min": None,
        "net_profit_max": None,
    }

    assert study.event_key(event) == study.event_key(dict(event))


def test_materialize_joins_event_and_document_keys_without_ambiguity(tmp_path) -> None:
    study = _load_module()
    root = tmp_path / "evidence"
    root.mkdir()
    connection = duckdb.connect(str(root / "event_documents.duckdb"))
    study._initialize(connection)
    connection.execute(
        "INSERT INTO event_document_targets VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)",
        ["event-1", "A.SZ", date(2020, 1, 1), date(2019, 12, 31), "预增", "NO_MATCH", None],
    )

    summary = study._materialize(connection, root)

    assert summary["targets"] == 1
    assert summary["status"] == {"NO_MATCH": 1}
    connection.close()


def test_no_match_target_is_not_queried_again(tmp_path) -> None:
    study = _load_module()
    connection = duckdb.connect(str(tmp_path / "events.duckdb"))
    study._initialize(connection)
    row = {
        "symbol": "000001.SZ",
        "ann_date": date(2020, 1, 1),
        "period_end": date(2019, 12, 31),
        "type": "预增",
        "p_change_min": None,
        "p_change_max": None,
        "net_profit_min": None,
        "net_profit_max": None,
    }
    connection.execute(
        "INSERT INTO event_document_targets VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)",
        [
            study.event_key(row),
            row["symbol"],
            row["ann_date"],
            row["period_end"],
            row["type"],
            f"NO_MATCH_{study.SELECTION_VERSION}",
            None,
        ],
    )

    class ClientThatMustNotRun:
        def announcements(self, *_args, **_kwargs):
            raise AssertionError("cached NO_MATCH must not be queried")

    assert study._discover(connection, ClientThatMustNotRun(), row) is None
    connection.close()
