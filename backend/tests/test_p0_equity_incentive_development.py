# ruff: noqa: RUF001
from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "run_p0_equity_incentive_development.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_equity_incentive", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_frozen_title_taxonomy_keeps_only_first_full_plan() -> None:
    assert study.classify_title("2020年限制性股票激励计划（草案）") == (
        study.CATEGORY,
        "restricted_stock",
    )
    assert study.classify_title("股票期权与限制性股票激励计划（草案）") == (
        study.CATEGORY,
        "mixed_option_restricted",
    )
    assert study.classify_title("股票期权激励计划（草案）摘要") is None
    assert study.classify_title("限制性股票激励计划（草案修订稿）") is None
    assert study.classify_title("监事会关于激励计划（草案）的核查意见") is None
    assert study.classify_title("子公司员工股权激励计划（草案）") is None


def _announcement(announcement_id: str, ann_date: date) -> dict:
    return {
        "announcement_id": announcement_id,
        "ann_date": ann_date,
        "symbol": "600001.SH",
        "company_name": "测试公司",
        "title": "限制性股票激励计划（草案）",
        "org_id": "org",
        "adjunct_url": "test.pdf",
        "column_id": "column",
        "announcement_type": "type",
    }


def test_same_symbol_cooldown_keeps_first_after_full_year() -> None:
    frame = pl.DataFrame(
        [
            _announcement("A1", date(2018, 1, 1)),
            _announcement("A2", date(2018, 6, 1)),
            _announcement("A3", date(2019, 1, 1)),
            _announcement("A4", date(2020, 1, 1)),
        ]
    )

    result = study.categorize_events(frame)

    assert result["ann_date"].to_list() == [
        date(2018, 1, 1),
        date(2019, 1, 1),
        date(2020, 1, 1),
    ]
