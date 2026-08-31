from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "audit_p0_index_inclusion_notices.py"
    )
    spec = importlib.util.spec_from_file_location("p0_index_notices", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _official(notice_id: int):
    notice = next(item for item in study.NOTICES if item.notice_id == notice_id)
    return {
        "publishDate": notice.announcement_date.isoformat(),
        "title": "official",
        "content": (
            f"沪深300指数更换{notice.expected_csi300_additions}只股票;"
            f"中证500指数更换{notice.expected_csi500_additions}只样本。"
        ),
    }


def _additions(*, corrupt_cycle: date | None = None) -> pl.DataFrame:
    rows = []
    for notice in study.NOTICES:
        for index_code, count in (
            ("000300.SH", notice.expected_csi300_additions),
            ("000905.SH", notice.expected_csi500_additions),
        ):
            if notice.cycle_month == corrupt_cycle and index_code == "000300.SH":
                count += 1
            rows.extend(
                {
                    "cycle_month": notice.cycle_month,
                    "index_code": index_code,
                    "symbol": f"{number:06d}.SH",
                }
                for number in range(count)
            )
    return pl.DataFrame(rows)


def test_extract_official_counts_normalizes_html_and_spaces() -> None:
    result = study.extract_official_counts(
        "<p>沪深 300 指数更换 21 只股票;中证 500 指数更换 50 只样本股。</p>"
    )

    assert result == {"000300.SH": 21, "000905.SH": 50}


def test_audit_rejects_entire_cycle_when_one_index_count_differs() -> None:
    corrupt = date(2015, 12, 1)

    matched, payload = study.audit_notices(
        _additions(corrupt_cycle=corrupt), fetcher=_official
    )

    assert payload["status"] == "NOTICE_MATCH_SUFFICIENT"
    assert payload["matched_cycles"] == len(study.NOTICES) - 1
    assert corrupt not in set(matched["cycle_month"].to_list())


def test_audit_refuses_price_fields() -> None:
    additions = _additions().with_columns(pl.lit(1.0).alias("close"))

    try:
        study.audit_notices(additions, fetcher=_official)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("price field should be rejected")
