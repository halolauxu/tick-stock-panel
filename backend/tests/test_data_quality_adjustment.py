"""除权污染检测: 未复权价格在除权日产生的假跌幅必须能被识别并量化。"""
from __future__ import annotations

from datetime import date

import polars as pl

from app.data_quality import (
    AdjustmentAudit,
    audit_price_adjustment,
)


def _panel(rows: list[tuple[str, date, float]]) -> pl.DataFrame:
    """构造最小行情面板; close 与 raw_close 相等表示未应用复权。"""
    return pl.DataFrame(
        {
            "symbol": [row[0] for row in rows],
            "date": [row[1] for row in rows],
            "close": [row[2] for row in rows],
            "raw_close": [row[2] for row in rows],
        }
    )


def test_main_board_drop_beyond_limit_is_flagged() -> None:
    """主板跌停是 10%; 单日跌 15% 只可能来自除权, 必须被标记。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 10.0),
        ("600000.SH", date(2026, 6, 11), 8.5),
    ])

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert audit.suspect_rows == 1
    assert audit.suspect_symbols == 1


def test_drop_within_main_board_limit_is_not_flagged() -> None:
    """跌幅在板块限制内属于正常交易, 不能误报。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 10.0),
        ("600000.SH", date(2026, 6, 11), 9.2),
    ])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0


def test_chinext_uses_twenty_percent_limit() -> None:
    """创业板跌停是 20%; 跌 15% 属正常, 不能按主板口径误报。"""
    panel = _panel([
        ("300001.SZ", date(2026, 6, 10), 10.0),
        ("300001.SZ", date(2026, 6, 11), 8.5),
    ])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0


def test_star_market_drop_beyond_twenty_percent_is_flagged() -> None:
    """科创板跌停是 20%; 跌 25% 超出限制, 必须被标记。"""
    panel = _panel([
        ("688001.SH", date(2026, 6, 10), 10.0),
        ("688001.SH", date(2026, 6, 11), 7.5),
    ])

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert audit.suspect_rows == 1


def test_beijing_exchange_uses_thirty_percent_limit() -> None:
    """北交所跌停是 30%; 跌 25% 属正常。"""
    panel = _panel([
        ("830001.BJ", date(2026, 6, 10), 10.0),
        ("830001.BJ", date(2026, 6, 11), 7.5),
    ])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0


def test_reports_adjustment_never_applied_when_close_equals_raw_close() -> None:
    """close 恒等于 raw_close 说明复权数据从未生效, 必须显式报告。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 10.0),
        ("600000.SH", date(2026, 6, 11), 9.8),
    ])

    audit = audit_price_adjustment(panel)

    assert audit.adjustment_applied is False


def test_detects_adjustment_applied_when_close_differs() -> None:
    """close 与 raw_close 不同说明复权已生效。"""
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "date": [date(2026, 6, 10), date(2026, 6, 11)],
            "close": [9.5, 9.31],
            "raw_close": [10.0, 9.8],
        }
    )

    audit = audit_price_adjustment(panel)

    assert audit.adjustment_applied is True


def test_suspect_rows_group_by_month_for_dividend_season_evidence() -> None:
    """按月汇总, 用于判断异常是否集中在分红除权季。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 10.0),
        ("600000.SH", date(2026, 6, 11), 8.5),
        ("600001.SH", date(2026, 7, 10), 10.0),
        ("600001.SH", date(2026, 7, 11), 8.0),
    ])

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert audit.monthly_counts == {"2026-06": 1, "2026-07": 1}


def test_empty_panel_returns_zero_audit() -> None:
    """空面板不应抛错。"""
    audit = audit_price_adjustment(pl.DataFrame())

    assert isinstance(audit, AdjustmentAudit)
    assert audit.suspect_rows == 0


def test_gap_across_symbols_does_not_leak() -> None:
    """跨标的不能串联计算收益率。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 100.0),
        ("600001.SH", date(2026, 6, 11), 5.0),
    ])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0


def test_samples_expose_offending_rows_for_manual_verification() -> None:
    """必须给出可人工核对的样本行, 而不是只报一个总数。"""
    panel = _panel([
        ("600000.SH", date(2026, 6, 10), 10.0),
        ("600000.SH", date(2026, 6, 11), 8.5),
    ])

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert len(audit.samples) == 1
    sample = audit.samples[0]
    assert sample["symbol"] == "600000.SH"
    assert sample["date"] == date(2026, 6, 11)
    assert sample["return_pct"] < -0.10
