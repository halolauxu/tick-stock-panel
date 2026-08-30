"""次新股上市初期不受涨跌幅限制, 不能计为数据污染。

A股新股上市首日不设涨跌幅限制, 创业板与科创板前 5 个交易日同样放开。
这些暴跌是合法交易, 与除权断层是两回事。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.data_quality import audit_price_adjustment


def _listing(symbol: str, closes: list[float]) -> pl.DataFrame:
    """构造一只从数据首日开始交易的标的。"""
    start = date(2026, 1, 5)
    dates = [start + timedelta(days=index) for index in range(len(closes))]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "date": dates,
            "close": closes,
            "raw_close": closes,
        }
    )


def test_new_listing_crash_is_excluded_by_default() -> None:
    """上市第二天暴跌 28% 属于次新股常态, 默认不计入污染。"""
    panel = _listing("301667.SZ", [100.0, 72.0])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0


def test_seasoned_stock_crash_is_still_flagged() -> None:
    """上市已久的标的出现同样跌幅时仍必须报出。"""
    closes = [100.0] * 40 + [72.0]
    panel = _listing("600000.SH", closes)

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 1


def test_new_listing_window_is_configurable() -> None:
    """窗口设为 0 时不做次新股豁免, 保留原始口径。"""
    panel = _listing("301667.SZ", [100.0, 72.0])

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert audit.suspect_rows == 1


def test_excluded_new_listing_rows_are_reported_separately() -> None:
    """被豁免的次新股行数必须单独报告, 不能静默丢弃。"""
    panel = _listing("301667.SZ", [100.0, 72.0])

    audit = audit_price_adjustment(panel)

    assert audit.new_listing_rows == 1
