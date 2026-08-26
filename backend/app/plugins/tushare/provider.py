"""Tushare A-share historical-minute provider.

The user's independently purchased ``stk_mins`` entitlement covers historical
A-share minute bars.  This plugin therefore declares only ``minute`` and only
accepts ``asset_type='stock'``.  ETF/index minute requests deliberately raise so
the existing routing layer can fall back to TickFlow instead of silently
claiming unsupported data.

Tushare documents ``vol`` as shares and ``amount`` as CNY.  The panel's minute
volume contract is hands, so the provider performs the explicit ``vol / 100``
boundary conversion.  ``trade_time`` is a Beijing wall-clock string and is
stored as a naive ``Datetime('us')``, matching the existing minute repository
contract.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from app import secrets_store
from app.data_providers.base import AssetType
from app.market_time import CN_TZ
from app.plugins.tushare.client import TushareClient, TushareError

logger = logging.getLogger(__name__)

API_KEY_ENV = "TUSHARE_TOKEN"
SECRETS_FIELD = "tushare_api_key"
_DATASETS = ("minute",)
_MINUTE_CANONICAL = [
    "symbol", "datetime", "open", "high", "low", "close", "volume", "amount",
]


def get_api_key() -> str:
    return secrets_store.get_env_backed_secret(SECRETS_FIELD, API_KEY_ENV)


def availability() -> tuple[bool, str]:
    if get_api_key():
        return True, "ok"
    return False, f"未配置 {API_KEY_ENV}(可在设置页数据源卡片中直接填写)"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    """Validate both the token and the purchased ``stk_mins`` entitlement."""
    client = None
    try:
        client = TushareClient(api_key, timeout=15.0, min_interval_s=0)
        rows = client.stock_minutes("600000.SH", freq="1min")
        if not rows:
            return False, "Token 可访问 Tushare, 但 stk_mins 未返回数据; 请确认已开通 A股历史分钟权限"
        return True, "ok"
    except TushareError as exc:
        return False, f"Token 或 stk_mins 权限验证失败: {exc}"
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


@dataclass
class _TushareConfig:
    name: str = "tushare"
    display_name: str = "Tushare (A股历史分钟)"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _beijing_wall_clock(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(CN_TZ).replace(tzinfo=None)


def _normalize_freq(value: str) -> str:
    raw = str(value or "1m").strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit()) or "1"
    if digits not in {"1", "5", "15", "30", "60"}:
        raise TushareError(f"Tushare 不支持分钟周期: {value}")
    return f"{digits}min"


class TushareProvider:
    name = "tushare"
    builtin = True

    def __init__(self) -> None:
        self.config = _TushareConfig()
        self._client: TushareClient | None = None

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    def _get_client(self) -> TushareClient:
        if self._client is None:
            self._client = TushareClient(get_api_key())
        return self._client

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        if asset_type != "stock":
            raise TushareError(f"当前权限仅覆盖 A股历史分钟, 不支持 asset_type={asset_type}")

        tushare_freq = _normalize_freq(freq)
        start = _beijing_wall_clock(start_time)
        end = _beijing_wall_clock(end_time)
        frames: list[pl.DataFrame] = []
        total = len(symbols)

        for index, symbol in enumerate(symbols, start=1):
            rows = self._get_client().stock_minutes(
                symbol,
                freq=tushare_freq,
                start_time=start,
                end_time=end,
            )
            frame = self._minute_df(rows)
            if not frame.is_empty():
                frames.append(frame)
            if on_chunk_done is not None:
                on_chunk_done(index, total)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").unique(
            subset=["symbol", "datetime"], keep="last",
        ).sort(["symbol", "datetime"])

    @staticmethod
    def _minute_df(rows: list[dict]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        frame = pl.DataFrame(rows)
        required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TushareError(f"stk_mins 响应缺少字段: {', '.join(missing)}")

        frame = frame.rename({
            "ts_code": "symbol",
            "trade_time": "datetime",
            "vol": "volume",
        }).with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("datetime").cast(pl.Utf8).str.to_datetime(
                "%Y-%m-%d %H:%M:%S", strict=False, time_unit="us",
            ),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ("open", "high", "low", "close", "volume", "amount")
            ],
        ).with_columns(
            (pl.col("volume") / 100).alias("volume"),
        ).select(_MINUTE_CANONICAL).drop_nulls(_MINUTE_CANONICAL)
        return frame

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        if dataset != "minute":
            raise ValueError(f"Tushare 插件不支持数据集: {dataset}")
        frame = self.get_minute(symbols or ["600000.SH"], None, None)
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts() if not frame.is_empty() else [],
        }
