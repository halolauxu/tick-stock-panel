"""审计必须以复权价判定断层。

``raw_close`` 按设计保留除权跳空(涨跌停基准价需要它), 策略信号读的是复权后的
``close``。用 raw_close 判定会让审计在复权修好之后仍然永远失败。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from app.data_quality import audit_price_adjustment


def test_adjusted_close_without_gap_is_clean_even_if_raw_close_gaps() -> None:
    """复权已修正断层时, 原始价的跳空不应再被计为污染。"""
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "date": [date(2026, 6, 10), date(2026, 6, 11)],
            # 复权后连续: 6.85 → 7.00, 正常波动
            "close": [6.85, 7.00],
            # 原始价除权跳空 -30%, 这是正确行为, 不是缺陷
            "raw_close": [10.0, 7.0],
        }
    )

    audit = audit_price_adjustment(panel)

    assert audit.adjustment_applied is True
    assert audit.suspect_rows == 0


def test_gap_remaining_in_adjusted_close_is_still_flagged() -> None:
    """复权价仍有无法解释的断层, 说明该标的的除权因子缺失, 必须报出。"""
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "date": [date(2026, 6, 10), date(2026, 6, 11)],
            "close": [10.0, 7.0],
            "raw_close": [10.0, 7.0],
        }
    )

    audit = audit_price_adjustment(panel, new_listing_bars=0)

    assert audit.suspect_rows == 1
