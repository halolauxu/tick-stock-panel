"""停牌复牌的价格断层不是数据污染。

停牌期间信息继续累积, 复牌首日的跳空不受单日涨跌幅约束(重大资产重组等情形
交易所本就放开限制)。这类断层必须与除权断层区分开。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from app.data_quality import audit_price_adjustment


def _two_symbol_panel(rows: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [row[0] for row in rows],
            "date": [row[1] for row in rows],
            "close": [row[2] for row in rows],
            "raw_close": [row[2] for row in rows],
        }
    )


def _calendar(symbol: str, days: list[int], price: float) -> list[tuple[str, date, float]]:
    return [(symbol, date(2026, 6, day), price) for day in days]


def test_gap_after_suspension_is_not_counted_as_contamination() -> None:
    """停牌一天后复牌暴跌, 属于合法断层。"""
    # 参照标的每天都有数据, 确立全市场交易日历为 1..14 日。
    reference = _calendar("600001.SH", list(range(1, 15)), 20.0)
    # 目标标的在 12 日停牌, 13 日复牌暴跌 18%。
    target = [
        *_calendar("600000.SH", list(range(1, 12)), 10.0),
        ("600000.SH", date(2026, 6, 13), 8.2),
        ("600000.SH", date(2026, 6, 14), 8.3),
    ]
    panel = _two_symbol_panel([*reference, *target])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 0
    assert audit.suspension_rows == 1


def test_gap_on_consecutive_trading_days_is_still_flagged() -> None:
    """没有停牌的连续交易日出现同样跌幅时仍必须报出。"""
    reference = _calendar("600001.SH", list(range(1, 15)), 20.0)
    target = [
        *_calendar("600000.SH", list(range(1, 13)), 10.0),
        ("600000.SH", date(2026, 6, 13), 8.2),
        ("600000.SH", date(2026, 6, 14), 8.3),
    ]
    panel = _two_symbol_panel([*reference, *target])

    audit = audit_price_adjustment(panel)

    assert audit.suspect_rows == 1
    assert audit.suspension_rows == 0
