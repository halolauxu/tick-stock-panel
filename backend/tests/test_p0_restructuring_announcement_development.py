from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_restructuring_announcement_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_restructuring_announcements", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_frozen_title_taxonomy_prioritizes_termination_and_excludes_updates() -> None:
    assert (
        study.classify_title("公司关于筹划重大资产重组的停牌公告") == "initial_major_restructuring"
    )
    assert (
        study.classify_title("公司发行股份购买资产暨关联交易预案") == "initial_major_restructuring"
    )
    assert study.classify_title("公司关于终止重大资产重组事项的公告") == "termination_control"
    assert study.classify_title("公司要约收购报告书") == "formal_tender_offer"
    assert study.classify_title("公司控股股东协议转让暨实际控制人变更公告") == "control_transfer"
    assert study.classify_title("公司控股股东协议转让部分股份暨权益变动公告") is None
    assert study.classify_title("公司重大资产重组进展公告") is None
    assert study.classify_title("公司要约收购报告书摘要") is None
    assert study.classify_title("公司控股股东减持进展公告") is None


def _announcement(symbol: str, ann_date: date, title: str, art_code: str) -> dict:
    return {
        "art_code": art_code,
        "ann_date": ann_date,
        "symbol": symbol,
        "company_name": "测试",
        "title": title,
        "column_name": "测试",
        "column_code": "x",
    }


def test_same_category_cooldown_keeps_first_event_after_full_year() -> None:
    frame = pl.DataFrame(
        [
            _announcement("000001.SZ", date(2018, 1, 1), "筹划重大资产重组停牌公告", "A1"),
            _announcement("000001.SZ", date(2018, 6, 1), "筹划重大资产重组停牌公告", "A2"),
            _announcement("000001.SZ", date(2019, 1, 1), "筹划重大资产重组停牌公告", "A3"),
            _announcement("000001.SZ", date(2020, 1, 1), "筹划重大资产重组停牌公告", "A4"),
        ]
    )

    result = study.categorize_events(frame)

    assert result["ann_date"].to_list() == [
        date(2018, 1, 1),
        date(2019, 1, 1),
        date(2020, 1, 1),
    ]
    assert result["holding_trading_days"].to_list() == [5, 5, 5]


def test_same_day_duplicate_titles_count_once_per_category() -> None:
    frame = pl.DataFrame(
        [
            _announcement("600001.SH", date(2020, 1, 2), "发行股份购买资产预案", "A1"),
            _announcement("600001.SH", date(2020, 1, 2), "发行股份购买资产暨关联交易预案", "A2"),
        ]
    )

    result = study.categorize_events(frame)

    assert result.height == 1
    assert result["category"][0] == "initial_major_restructuring"
