"""除权污染的下游影响: 假跌幅会制造假新低并误触发止损。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.data_quality import audit_signal_contamination


def _series(symbol: str, closes: list[float], *, start: date = date(2026, 1, 5)) -> pl.DataFrame:
    """构造单标的连续价格序列; open 略低于 close 以满足收阳条件。"""
    dates = [start + timedelta(days=index) for index in range(len(closes))]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "date": dates,
            "open": [value * 0.99 for value in closes],
            "close": closes,
            "raw_close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )


def test_ex_dividend_gap_that_makes_a_new_low_is_counted() -> None:
    """除权断层把价格砸到区间新低时, 必须计入假新低。"""
    panel = _series("600000.SH", [10.0] * 25 + [7.0])

    impact = audit_signal_contamination(panel, lookback=20, min_samples=20)

    assert impact.suspect_rows == 1
    assert impact.fake_new_lows == 1


def test_gap_that_does_not_reach_a_new_low_is_not_counted_as_fake_low() -> None:
    """跌幅异常但没有跌破区间低点时, 不计入假新低。"""
    # 首日 5.0 压低了区间低点; 末日 20.0 → 13.0 跌 35% 属异常, 但仍高于该低点。
    closes = [5.0] + [20.0] * 24 + [13.0]
    panel = _series("600000.SH", closes)

    impact = audit_signal_contamination(panel, min_samples=20)

    assert impact.suspect_rows == 1
    assert impact.fake_new_lows == 0


def test_counts_rows_breaching_the_stop_loss_threshold() -> None:
    """除权跌幅超过止损线的行会导致持仓被误止损。"""
    panel = _series("600000.SH", [10.0] * 25 + [7.0])

    impact = audit_signal_contamination(panel, stop_loss=-0.06)

    assert impact.stop_loss_hits == 1


def test_normal_limit_down_is_not_treated_as_contamination() -> None:
    """真实跌停在限制内, 不能算作除权污染。"""
    panel = _series("600000.SH", [10.0] * 25 + [9.2])

    impact = audit_signal_contamination(panel)

    assert impact.suspect_rows == 0
    assert impact.fake_new_lows == 0
    assert impact.stop_loss_hits == 0


def test_empty_panel_is_safe() -> None:
    impact = audit_signal_contamination(pl.DataFrame())

    assert impact.suspect_rows == 0
    assert impact.fake_new_lows == 0


def test_new_listing_crash_is_not_a_false_stop_loss() -> None:
    """次新股上市初期的暴跌是真实行情, 止损被触发不算误伤。"""
    panel = _series("301667.SZ", [100.0, 72.0])

    impact = audit_signal_contamination(panel, stop_loss=-0.06)

    assert impact.suspect_rows == 0
    assert impact.stop_loss_hits == 0
