"""分钟 K 的统一质量契约。

股票一分钟线只接受日内常规交易时段：09:30-11:30、13:00-15:00。
数据源可能返回北交所 15:01-15:30 的盘后大宗交易确认记录；这些记录不属于
日内行情，必须在进入缓存、落盘和完整度统计前过滤。
"""
from __future__ import annotations

from datetime import time

import polars as pl

REGULAR_MINUTE_BARS = 241
MORNING_START = time(9, 30)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(15, 0)


def regular_session_expr(column: str = "datetime") -> pl.Expr:
    """返回 A 股日内常规交易时段过滤表达式（边界均包含）。"""
    minute_time = pl.col(column).dt.time()
    return (
        ((minute_time >= MORNING_START) & (minute_time <= MORNING_END))
        | ((minute_time >= AFTERNOON_START) & (minute_time <= AFTERNOON_END))
    )


def filter_regular_session(frame: pl.DataFrame) -> pl.DataFrame:
    """删除空时间和日内常规交易时段外记录；不修改其他字段或策略数据。"""
    if frame.is_empty() or "datetime" not in frame.columns:
        return frame
    return frame.filter(
        pl.col("datetime").is_not_null() & regular_session_expr("datetime")
    )


def minute_quality_payload(frame: pl.DataFrame) -> dict[str, int]:
    """深度检查一个或多个分钟 K 分区，返回可序列化的质量计数。"""
    if frame.is_empty():
        return {
            "rows": 0,
            "symbols": 0,
            "full_symbols": 0,
            "partial_symbols": 0,
            "extra_symbols": 0,
            "null_datetime": 0,
            "null_ohlc": 0,
            "invalid_ohlc": 0,
            "duplicate_symbol_datetime": 0,
            "out_of_regular_session": 0,
            "min_bars": 0,
            "max_bars": 0,
        }

    required = {"symbol", "datetime", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"分钟K缺少字段: {', '.join(missing)}")

    null_datetime = int(frame.select(pl.col("datetime").is_null().sum()).item() or 0)
    null_ohlc = int(
        frame.select(
            pl.any_horizontal(
                *[pl.col(column).is_null() for column in ("open", "high", "low", "close")]
            ).sum()
        ).item()
        or 0
    )
    invalid_ohlc = int(
        frame.select(
            (
                (pl.col("open") <= 0)
                | (pl.col("high") <= 0)
                | (pl.col("low") <= 0)
                | (pl.col("close") <= 0)
                | (pl.col("high") < pl.max_horizontal("open", "low", "close"))
                | (pl.col("low") > pl.min_horizontal("open", "high", "close"))
            ).fill_null(False).sum()
        ).item()
        or 0
    )
    duplicate_keys = int(
        frame.select(pl.struct("symbol", "datetime").is_duplicated().sum()).item() or 0
    )
    out_of_session = int(
        frame.select(
            (
                pl.col("datetime").is_not_null()
                & ~regular_session_expr("datetime")
            ).sum()
        ).item()
        or 0
    )
    bars = frame.filter(pl.col("datetime").is_not_null()).group_by("symbol").len()
    if bars.is_empty():
        min_bars = max_bars = full_symbols = partial_symbols = extra_symbols = 0
    else:
        lengths = bars.get_column("len")
        min_bars = int(lengths.min() or 0)
        max_bars = int(lengths.max() or 0)
        full_symbols = int((lengths == REGULAR_MINUTE_BARS).sum())
        partial_symbols = int((lengths < REGULAR_MINUTE_BARS).sum())
        extra_symbols = int((lengths > REGULAR_MINUTE_BARS).sum())

    return {
        "rows": int(frame.height),
        "symbols": int(bars.height),
        "full_symbols": full_symbols,
        "partial_symbols": partial_symbols,
        "extra_symbols": extra_symbols,
        "null_datetime": null_datetime,
        "null_ohlc": null_ohlc,
        "invalid_ohlc": invalid_ohlc,
        "duplicate_symbol_datetime": duplicate_keys,
        "out_of_regular_session": out_of_session,
        "min_bars": min_bars,
        "max_bars": max_bars,
    }
