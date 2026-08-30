from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "collect_major_contract_events.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_major_contract_events", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(event_id: str, ratio: float | None = 55.0) -> dict:
    return {
        "DIM_SCODE": event_id,
        "DIM_RDATE": "2019-01-02 00:00:00",
        "SECURITYCODE": "600001",
        "SECURITYSHORTNAME": "测试公司",
        "CONTRACTNAME": "重大销售合同",
        "CONTRACTTYPE": "001001",
        "CONTRACTTYPENAME": "销售合同",
        "SIGNATORY": "测试公司",
        "SIGNATORYRELNAME": "公司本身",
        "COUNTERPARTY": "客户",
        "COUNTERPARTYRELNAME": "无关联关系",
        "SIGNDATE": "2019-01-01 00:00:00",
        "AMOUNTS": 550_000_000,
        "SNDYYSR": 1_000_000_000,
        "ZSNDYYSRBL": ratio,
        "ISABOLISHED": None,
        "CONTENTS": "合同已签署",
        "SIGNEFFECT": "占公司上年度营业收入的55%",
        "RCHANGE1DC": 9.99,
        "RCHANGE20DC": 99.99,
    }


def test_fetch_year_reads_every_page_and_checks_total() -> None:
    calls = []

    def fetch(params):
        calls.append(params["pageNumber"])
        page = int(params["pageNumber"])
        rows = [_row(f"E{page}-{index}") for index in range(collector.PAGE_SIZE)]
        if page == 2:
            rows = rows[:1]
        return {
            "success": True,
            "result": {"count": 501, "pages": 2, "data": rows},
        }

    rows = collector.fetch_year(fetch, 2019)

    assert len(rows) == 501
    assert calls == ["1", "2"]


def test_normalize_parses_ratio_and_never_persists_provider_returns() -> None:
    explicit = _row("E1")
    text_ratio = _row("E2", ratio=None)
    text_ratio["SNDYYSR"] = None
    text_ratio["SIGNEFFECT"] = "合同额约占2018年度经审计营业收入的47.12%。"

    frame = collector.normalize([explicit, explicit, text_ratio], 2019)

    assert frame.height == 2
    assert frame["ann_date"].to_list() == [date(2019, 1, 2), date(2019, 1, 2)]
    by_source = {
        row["source_security_id"]: (
            row["revenue_ratio_pct"],
            row["ratio_source"],
        )
        for row in frame.iter_rows(named=True)
    }
    assert by_source == {
        "E1": (55.0, "provider_previous_revenue_ratio"),
        "E2": (47.12, "announcement_effect_text"),
    }
    assert "RCHANGE1DC" not in frame.columns
    assert "RCHANGE20DC" not in frame.columns


def test_same_security_multiple_contracts_are_not_collapsed() -> None:
    first = _row("SECURITY-INNER-ID")
    second = _row("SECURITY-INNER-ID")
    second["CONTRACTNAME"] = "另一份重大合同"

    frame = collector.normalize([first, second], 2019)

    assert frame.height == 2
    assert frame["event_id"].n_unique() == 2
