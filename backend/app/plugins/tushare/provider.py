"""Tushare A-share minute, financial and post-market supplemental provider.

The user's independently purchased ``stk_mins`` entitlement covers historical
A-share minute bars, while the standard (non-VIP) financial APIs cover per-stock
statements and indicators. ETF/index minute requests deliberately raise so the
existing routing layer can fall back to TickFlow instead of silently claiming
unsupported data.

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
from datetime import date, datetime, timedelta

import polars as pl

from app import secrets_store
from app.data_providers.base import AssetType
from app.market_time import CN_TZ
from app.plugins.tushare.client import TushareClient, TushareError
from app.services.minute_quality import minute_quality_payload, sanitize_minute_rows

logger = logging.getLogger(__name__)

API_KEY_ENV = "TUSHARE_TOKEN"
SECRETS_FIELD = "tushare_api_key"
_DATASETS = ("adj_factor", "minute", "financial")
_MINUTE_CANONICAL = [
    "symbol",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
_AUCTION_CANONICAL = [
    "symbol", "date", "session", "open", "high", "low", "close",
    "volume_shares", "amount", "vwap",
]
_IRM_QA_CANONICAL = [
    "symbol", "name", "date", "trade_date", "question", "answer",
    "pub_time", "industry", "exchange",
]
_FORECAST_CANONICAL = [
    "symbol",
    "ann_date",
    "period_end",
    "type",
    "p_change_min",
    "p_change_max",
    "net_profit_min",
    "net_profit_max",
    "last_parent_net",
    "first_ann_date",
    "summary",
    "change_reason",
    "collection_source",
]
_FINANCIAL_NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "metrics": (
        "eps_basic",
        "eps_diluted",
        "bps",
        "ocfps",
        "roe",
        "roe_diluted",
        "roa",
        "gross_margin",
        "net_margin",
        "debt_to_asset_ratio",
        "revenue_yoy",
        "net_income_yoy",
        "operating_cash_to_revenue",
        "inventory_turnover",
    ),
    "income": (
        "revenue",
        "operating_cost",
        "operating_profit",
        "selling_expense",
        "admin_expense",
        "rd_expense",
        "financial_expense",
        "non_operating_income",
        "non_operating_expense",
        "total_profit",
        "income_tax",
        "net_income",
        "net_income_attributable",
        "basic_eps",
        "diluted_eps",
    ),
    "balance_sheet": (
        "total_assets",
        "total_current_assets",
        "total_non_current_assets",
        "cash_and_equivalents",
        "accounts_receivable",
        "inventory",
        "fixed_assets",
        "intangible_assets",
        "goodwill",
        "total_liabilities",
        "total_current_liabilities",
        "total_non_current_liabilities",
        "short_term_borrowing",
        "long_term_borrowing",
        "accounts_payable",
        "total_equity",
        "equity_attributable",
        "retained_earnings",
        "minority_interest",
    ),
    "cash_flow": (
        "net_operating_cash_flow",
        "net_investing_cash_flow",
        "net_financing_cash_flow",
        "capex",
        "net_cash_change",
    ),
    "shares": ("total_shares", "float_shares"),
}


def get_api_key() -> str:
    return secrets_store.get_env_backed_secret(SECRETS_FIELD, API_KEY_ENV)


def availability() -> tuple[bool, str]:
    if get_api_key():
        return True, "ok"
    return False, f"未配置 {API_KEY_ENV}(可在设置页数据源卡片中直接填写)"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    """Validate the token plus the minute and standard-financial entitlements."""
    client = None
    try:
        client = TushareClient(api_key, timeout=15.0, min_interval_s=0)
        rows = client.stock_minutes("600000.SH", freq="1min")
        if not rows:
            return (
                False,
                "Token 可访问 Tushare, 但 stk_mins 未返回数据; 请确认已开通 A股历史分钟权限",
            )
        rows = client.financial_records("metrics", "600000.SH")
        if not rows:
            return False, "Token 可访问 Tushare, 但 fina_indicator 未返回数据; 请确认财务接口权限"
        return True, "ok"
    except TushareError as exc:
        return False, f"Token 或 Tushare 接口权限验证失败: {exc}"
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


@dataclass
class _TushareConfig:
    name: str = "tushare"
    display_name: str = "Tushare (除权/分钟/财务/盘后特色数据)"
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

    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """Return event ratios derived from Tushare's cumulative factors.

        The enriched pipeline expects one ``pre/post`` ratio per ex-date, while
        Tushare returns a cumulative daily factor.  Differencing the cumulative
        series by symbol preserves that contract and avoids persisting millions
        of unchanged daily rows.
        """
        if not symbols:
            return pl.DataFrame()
        if asset_type != "stock":
            raise TushareError(f"当前除权因子仅支持 A股, 不支持 asset_type={asset_type}")

        if len(symbols) > 20 and start_time is not None and end_time is not None:
            return self._get_adj_factors_by_date(
                symbols,
                start_time,
                end_time,
                on_chunk_done,
            )

        frames: list[pl.DataFrame] = []
        failures = 0
        total = len(symbols)
        for index, symbol in enumerate(symbols, start=1):
            try:
                rows = self._get_client().adjustment_factors(
                    symbol,
                    start_time=start_time,
                    end_time=end_time,
                )
            except TushareError as exc:
                message = str(exc)
                if any(marker in message for marker in ("没有接口", "没有权限", "未配置")):
                    raise
                failures += 1
                logger.warning("tushare adj-factor symbol failed: symbol=%s: %s", symbol, exc)
                rows = []
            frame = self._adj_factor_df(rows)
            if not frame.is_empty():
                frames.append(frame)
            if on_chunk_done is not None:
                on_chunk_done(index, total)
            if index % 100 == 0 or index == total:
                logger.info(
                    "tushare adj-factor progress: symbols=%d/%d events=%d failures=%d",
                    index,
                    total,
                    sum(item.height for item in frames),
                    failures,
                )
        if not frames:
            return pl.DataFrame()
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset=["symbol", "trade_date"], keep="last")
            .sort(["symbol", "trade_date"])
        )

    def _get_adj_factors_by_date(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        on_chunk_done: Callable[[int, int], None] | None,
    ) -> pl.DataFrame:
        """Use bounded date queries for full-universe incremental refreshes."""
        start_date = start_time.date()
        end_date = end_time.date()
        if end_date < start_date:
            return pl.DataFrame()
        # A prior observation is required to turn Tushare's cumulative value
        # into an event ratio.  Twenty-one calendar days covers long A-share
        # exchange holidays while keeping the daily refresh bounded.
        query_start = start_date - timedelta(days=21)
        dates = [
            query_start + timedelta(days=offset)
            for offset in range((end_date - query_start).days + 1)
        ]
        requested = set(symbols)
        rows: list[dict] = []
        failures = 0
        for index, trade_date in enumerate(dates, start=1):
            try:
                fetched = self._get_client().adjustment_factors_by_date(trade_date)
            except TushareError as exc:
                message = str(exc)
                if any(marker in message for marker in ("没有接口", "没有权限", "未配置")):
                    raise
                failures += 1
                logger.warning("tushare adj-factor date failed: date=%s: %s", trade_date, exc)
                fetched = []
            rows.extend(
                row for row in fetched if str(row.get("ts_code") or "") in requested
            )
            if on_chunk_done is not None:
                on_chunk_done(index, len(dates))
        frame = self._adj_factor_df(rows)
        if frame.is_empty():
            return frame
        logger.info(
            "tushare adj-factor date refresh: dates=%d events=%d failures=%d",
            len(dates),
            frame.height,
            failures,
        )
        return frame.filter(
            (pl.col("trade_date") >= start_date)
            & (pl.col("trade_date") <= end_date)
        )

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

        frames: list[pl.DataFrame] = []
        self.stream_minute(
            symbols,
            start_time=start_time,
            end_time=end_time,
            asset_type=asset_type,
            freq=freq,
            on_batch=frames.append,
            on_chunk_done=on_chunk_done,
        )

        if not frames:
            return pl.DataFrame()
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(
                subset=["symbol", "datetime"],
                keep="last",
            )
            .sort(["symbol", "datetime"])
        )

    def stream_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_batch: Callable[[pl.DataFrame], None] | None = None,
        on_chunk_done: Callable[[int, int], None] | None = None,
        batch_symbols: int = 100,
    ) -> None:
        """Bounded-memory minute fetch used by full-market persistence.

        Tushare ``stk_mins`` is queried per symbol.  Emitting at most
        ``batch_symbols`` frames prevents a full-market range from being held in
        one Python list before persistence.  The caller additionally splits the
        requested time range so every per-symbol request stays below Tushare's
        8,000-row response limit.
        """
        if not symbols:
            return
        if asset_type != "stock":
            raise TushareError(f"当前权限仅覆盖 A股历史分钟, 不支持 asset_type={asset_type}")

        tushare_freq = _normalize_freq(freq)
        start = _beijing_wall_clock(start_time)
        end = _beijing_wall_clock(end_time)
        batch_symbols = max(1, int(batch_symbols))
        frames: list[pl.DataFrame] = []
        total = len(symbols)

        def flush() -> None:
            if not frames:
                return
            batch = (
                pl.concat(frames, how="diagonal_relaxed")
                .unique(subset=["symbol", "datetime"], keep="last")
                .sort(["symbol", "datetime"])
            )
            frames.clear()
            if on_batch is not None and not batch.is_empty():
                on_batch(batch)

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
            if len(frames) >= batch_symbols:
                flush()
            if on_chunk_done is not None:
                on_chunk_done(index, total)
        flush()

    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,
        on_progress: Callable[[int, int, int, int], None] | None = None,
    ) -> pl.DataFrame:
        """Fetch and normalize one of the panel's five canonical financial tables.

        Tushare's standard endpoints are per-stock.  They are intentionally used
        instead of the ``*_vip`` batch endpoints because the latter require a
        separate 5000-point entitlement.  Statement values are already CNY;
        daily_basic share capital is in 10k shares and is converted to shares.
        """
        if table not in _FINANCIAL_NUMERIC_FIELDS:
            raise TushareError(f"Tushare 不支持财务表: {table}")
        if not symbols:
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        failures = 0
        row_count = 0
        total = len(symbols)
        for index, symbol in enumerate(symbols, start=1):
            try:
                rows = self._get_client().financial_records(table, symbol)
            except TushareError as exc:
                message = str(exc)
                # Configuration/entitlement failures affect every symbol and
                # must abort immediately. Transient single-symbol failures are
                # isolated so the rest of a multi-hour full-market sync survives.
                if any(marker in message for marker in ("没有接口", "没有权限", "未配置")):
                    raise
                failures += 1
                logger.warning(
                    "tushare financial symbol failed: table=%s symbol=%s: %s",
                    table,
                    symbol,
                    exc,
                )
                rows = []
            frame = self._financial_df(table, rows, latest_only=latest_only)
            if not frame.is_empty():
                frames.append(frame)
                row_count += frame.height
            if on_progress is not None:
                on_progress(index, total, row_count, failures)
            if index % 100 == 0 or index == total:
                logger.info(
                    "tushare financial progress: table=%s symbols=%d/%d rows=%d failures=%d",
                    table,
                    index,
                    total,
                    row_count,
                    failures,
                )
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort(
            ["symbol", "period_end"],
            descending=[False, True],
        )

    def get_auction(self, trade_date: date, session: str) -> pl.DataFrame:
        """Return one day's market-wide opening or closing auction rows."""
        rows = self._get_client().auction_records(session, trade_date)
        if len(rows) >= 10_000:
            raise TushareError(
                f"{session} 集合竞价返回 {len(rows)} 行,达到接口上限,拒绝保存可能被截断的数据"
            )
        records: list[dict] = []
        for source in rows:
            symbol = _canonical_symbol(source.get("ts_code"))
            day = _canonical_date(source.get("trade_date"))
            if not symbol or not day:
                continue
            records.append({
                "symbol": symbol,
                "date": day,
                "session": session,
                "open": _number(source, "open"),
                "high": _number(source, "high"),
                "low": _number(source, "low"),
                "close": _number(source, "close"),
                "volume_shares": _number(source, "vol"),
                "amount": _number(source, "amount"),
                "vwap": _number(source, "vwap"),
            })
        if not records:
            return pl.DataFrame()
        return (
            pl.DataFrame(
                records,
                schema_overrides={
                    "symbol": pl.Utf8,
                    "date": pl.Utf8,
                    "session": pl.Utf8,
                    **dict.fromkeys(_AUCTION_CANONICAL[3:], pl.Float64),
                },
                strict=False,
                infer_schema_length=None,
            )
            .select(_AUCTION_CANONICAL)
            .unique(subset=["symbol", "date", "session"], keep="last")
            .sort(["date", "symbol", "session"])
        )

    def get_irm_qa(
        self,
        exchange: str,
        *,
        pub_start: date,
        pub_end: date,
    ) -> pl.DataFrame:
        """Return newly published Shanghai or Shenzhen IR Q&A rows."""
        normalized_exchange = exchange.strip().lower()
        rows = self._get_client().irm_qa_records(
            normalized_exchange,
            pub_start=pub_start,
            pub_end=pub_end,
        )
        if len(rows) >= 3_000:
            raise TushareError(
                f"{normalized_exchange.upper()} 董秘问答返回 {len(rows)} 行,达到接口上限,"
                "拒绝保存可能被截断的数据"
            )
        records: list[dict] = []
        for source in rows:
            symbol = _canonical_symbol(source.get("ts_code"), normalized_exchange)
            trade_date = _canonical_date(source.get("trade_date"))
            pub_time = str(source.get("pub_time") or "").strip()
            published_date = _canonical_datetime_date(pub_time) or trade_date
            question = str(source.get("q") or "").strip()
            answer = str(source.get("a") or "").strip()
            if not symbol or not published_date or not question:
                continue
            records.append({
                "symbol": symbol,
                "name": str(source.get("name") or "").strip(),
                "date": published_date,
                "trade_date": trade_date,
                "question": question,
                "answer": answer,
                "pub_time": pub_time,
                "industry": str(source.get("industry") or "").strip(),
                "exchange": normalized_exchange.upper(),
            })
        if not records:
            return pl.DataFrame()
        return (
            pl.DataFrame(
                records,
                schema_overrides=dict.fromkeys(_IRM_QA_CANONICAL, pl.Utf8),
                strict=False,
                infer_schema_length=None,
            )
            .select(_IRM_QA_CANONICAL)
            .unique(subset=["symbol", "pub_time", "question"], keep="last")
            .sort(["date", "pub_time", "symbol"])
        )

    def get_forecast(self, announcement_date: date) -> pl.DataFrame:
        """Return one announcement day's point-in-time earnings forecasts."""
        rows = self._get_client().forecast_records(announcement_date)
        if len(rows) >= 5_000:
            raise TushareError(
                f"业绩预告 {announcement_date} 返回 {len(rows)} 行,达到接口上限,"
                "拒绝保存可能被截断的数据"
            )
        records: list[dict] = []
        numeric = {
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            "last_parent_net",
        }
        for source in rows:
            symbol = _canonical_symbol(source.get("ts_code"))
            ann_date = _canonical_date(source.get("ann_date"))
            if not symbol or ann_date != announcement_date.isoformat():
                continue
            record = {
                "symbol": symbol,
                "ann_date": ann_date,
                "period_end": _canonical_date(source.get("end_date")),
                "type": str(source.get("type") or "").strip(),
                "first_ann_date": _canonical_date(source.get("first_ann_date")),
                "summary": str(source.get("summary") or "").strip(),
                "change_reason": str(source.get("change_reason") or "").strip(),
                "collection_source": "forecast_by_announcement_day",
            }
            record.update({field: _number(source, field) for field in numeric})
            records.append(record)
        if not records:
            return pl.DataFrame()
        return (
            pl.DataFrame(records, infer_schema_length=None, strict=False)
            .with_columns(
                pl.col("ann_date").str.to_date(strict=False),
                pl.col("period_end").str.to_date(strict=False),
                pl.col("first_ann_date").str.to_date(strict=False),
            )
            .select(_FORECAST_CANONICAL)
            .unique(
                subset=[
                    "symbol",
                    "ann_date",
                    "period_end",
                    "type",
                    "p_change_min",
                    "p_change_max",
                    "net_profit_min",
                    "net_profit_max",
                ],
                keep="last",
            )
            .sort(["ann_date", "symbol", "period_end", "type"])
        )

    @staticmethod
    def _financial_df(
        table: str,
        rows: list[dict],
        *,
        latest_only: bool,
    ) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()

        normalized: dict[tuple[str, str], tuple[tuple[int, str], dict]] = {}
        for source in rows:
            symbol = str(source.get("ts_code") or "").strip()
            period = _canonical_date(
                source.get("trade_date") if table == "shares" else source.get("end_date")
            )
            if not symbol or not period:
                continue
            announce = _canonical_date(
                source.get("trade_date")
                if table == "shares"
                else (source.get("f_ann_date") or source.get("ann_date"))
            )
            record = {
                "symbol": symbol,
                "period_end": period,
                "announce_date": announce,
                **_map_financial_values(table, source),
            }
            # Tushare may return both the superseded and current version for the
            # same report.  update_flag=1 is the authoritative current record.
            priority = (
                1 if str(source.get("update_flag") or "") == "1" else 0,
                announce or "",
            )
            key = (symbol, period)
            if key not in normalized or priority >= normalized[key][0]:
                normalized[key] = (priority, record)

        records = [item[1] for item in normalized.values()]
        if not records:
            return pl.DataFrame()

        if table == "shares":
            records = _compress_share_history(records)
        records.sort(key=lambda row: (row["symbol"], row["period_end"]), reverse=True)
        if latest_only:
            latest: dict[str, dict] = {}
            for record in records:
                symbol = record["symbol"]
                if symbol not in latest or record["period_end"] > latest[symbol]["period_end"]:
                    latest[symbol] = record
            records = list(latest.values())

        numeric = _FINANCIAL_NUMERIC_FIELDS[table]
        # Some long Tushare histories have >100 records where the first 100
        # values of a field are integers/nulls and a later value is float. The
        # default inference window then selects an incompatible builder and the
        # whole multi-hour table sync is lost. Pin the canonical schema up front.
        frame = pl.DataFrame(
            records,
            schema_overrides={
                "symbol": pl.Utf8,
                "period_end": pl.Utf8,
                "announce_date": pl.Utf8,
                **dict.fromkeys(numeric, pl.Float64),
            },
            strict=False,
            infer_schema_length=None,
        )
        frame = frame.with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("period_end").cast(pl.Utf8),
            pl.col("announce_date").cast(pl.Utf8),
            *[pl.col(column).cast(pl.Float64, strict=False) for column in numeric],
        )
        return frame.select(["symbol", "period_end", "announce_date", *numeric])

    @staticmethod
    def _minute_df(rows: list[dict]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        frame = pl.DataFrame(rows)
        required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TushareError(f"stk_mins 响应缺少字段: {', '.join(missing)}")

        frame = (
            frame.rename(
                {
                    "ts_code": "symbol",
                    "trade_time": "datetime",
                    "vol": "volume",
                }
            )
            .with_columns(
                pl.col("symbol").cast(pl.Utf8),
                pl.col("datetime")
                .cast(pl.Utf8)
                .str.to_datetime(
                    "%Y-%m-%d %H:%M:%S",
                    strict=False,
                    time_unit="us",
                ),
                *[
                    pl.col(column).cast(pl.Float64, strict=False)
                    for column in ("open", "high", "low", "close", "volume", "amount")
                ],
            )
            .with_columns(
                (pl.col("volume") / 100).alias("volume"),
            )
            .select(_MINUTE_CANONICAL)
            .drop_nulls(_MINUTE_CANONICAL)
        )
        frame = sanitize_minute_rows(frame)
        quality = minute_quality_payload(frame)
        if quality["null_ohlc"] or quality["invalid_ohlc"]:
            raise TushareError(
                "stk_mins 返回非法价格数据: "
                f"空OHLC {quality['null_ohlc']} 行,"
                f"非法OHLC {quality['invalid_ohlc']} 行"
            )
        return frame

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        if dataset not in _DATASETS:
            raise ValueError(f"Tushare 插件不支持数据集: {dataset}")
        if dataset == "minute":
            frame = self.get_minute(symbols or ["600000.SH"], None, None)
        elif dataset == "adj_factor":
            frame = self.get_adj_factors(symbols or ["600000.SH"], None, None)
        else:
            frame = self.get_financials("metrics", symbols or ["600000.SH"])
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts() if not frame.is_empty() else [],
        }

    @staticmethod
    def _adj_factor_df(rows: list[dict]) -> pl.DataFrame:
        """Convert cumulative daily factors into sparse ex-date event ratios."""
        if not rows:
            return pl.DataFrame()
        required = {"ts_code", "trade_date", "adj_factor"}
        frame = pl.DataFrame(rows, infer_schema_length=None)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TushareError(f"adj_factor 响应缺少字段: {', '.join(missing)}")
        return (
            frame.select(
                pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
                pl.col("trade_date")
                .cast(pl.Utf8)
                .str.to_date("%Y%m%d", strict=False)
                .alias("trade_date"),
                pl.col("adj_factor").cast(pl.Float64, strict=False).alias("_cumulative"),
            )
            .drop_nulls(["symbol", "trade_date", "_cumulative"])
            .filter(pl.col("_cumulative") > 0)
            .unique(subset=["symbol", "trade_date"], keep="last")
            .sort(["symbol", "trade_date"])
            .with_columns(
                (
                    pl.col("_cumulative")
                    / pl.col("_cumulative").shift(1).over("symbol")
                ).alias("ex_factor")
            )
            .filter(
                pl.col("ex_factor").is_finite()
                & (pl.col("ex_factor") > 0)
                & ((pl.col("ex_factor") - 1.0).abs() > 1e-10)
            )
            .with_columns(
                pl.lit("stock").alias("asset_type"),
                pl.lit("tushare").alias("source"),
            )
            .select(["symbol", "asset_type", "source", "trade_date", "ex_factor"])
        )


def _canonical_date(value: object) -> str | None:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _canonical_datetime_date(value: object) -> str | None:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw[:10] if ch.isdigit())
    return _canonical_date(digits)


def _canonical_symbol(value: object, exchange: str | None = None) -> str | None:
    raw = str(value or "").strip().upper()
    if not raw:
        return None
    if "." in raw:
        return raw
    code = "".join(ch for ch in raw if ch.isdigit())
    if len(code) != 6:
        return None
    suffix = (exchange or "").strip().upper()
    if suffix not in {"SH", "SZ", "BJ"}:
        if code.startswith(("6", "68")):
            suffix = "SH"
        elif code.startswith(("0", "2", "3")):
            suffix = "SZ"
        elif code.startswith(("4", "8", "9")):
            suffix = "BJ"
        else:
            return None
    return f"{code}.{suffix}"


def _number(source: dict, *fields: str, scale: float = 1.0) -> float | None:
    for source_field in fields:
        value = source.get(source_field)
        if value is None or value == "":
            continue
        try:
            return float(value) * scale
        except (TypeError, ValueError):
            continue
    return None


def _map_financial_values(table: str, source: dict) -> dict[str, float | None]:
    if table == "metrics":
        return {
            "eps_basic": _number(source, "eps"),
            "eps_diluted": _number(source, "dt_eps"),
            "bps": _number(source, "bps"),
            "ocfps": _number(source, "ocfps"),
            "roe": _number(source, "roe"),
            # Tushare's roe_dt means ROE after non-recurring items, not diluted
            # ROE. Keep the canonical field empty rather than mislabeling it.
            "roe_diluted": None,
            "roa": _number(source, "roa"),
            "gross_margin": _number(source, "grossprofit_margin"),
            "net_margin": _number(source, "netprofit_margin"),
            "debt_to_asset_ratio": _number(source, "debt_to_assets"),
            "revenue_yoy": _number(source, "or_yoy"),
            "net_income_yoy": _number(source, "netprofit_yoy"),
            # ocf_to_or is a ratio (0.0859 = 8.59%); panel percentages are
            # stored in percentage points.
            "operating_cash_to_revenue": _number(source, "ocf_to_or", scale=100),
            "inventory_turnover": _number(source, "inv_turn"),
        }
    if table == "income":
        return {
            "revenue": _number(source, "revenue"),
            "operating_cost": _number(source, "oper_cost"),
            "operating_profit": _number(source, "operate_profit"),
            "selling_expense": _number(source, "sell_exp"),
            "admin_expense": _number(source, "admin_exp"),
            "rd_expense": _number(source, "rd_exp"),
            "financial_expense": _number(source, "fin_exp"),
            "non_operating_income": _number(source, "non_oper_income"),
            "non_operating_expense": _number(source, "non_oper_exp"),
            "total_profit": _number(source, "total_profit"),
            "income_tax": _number(source, "income_tax"),
            "net_income": _number(source, "n_income"),
            "net_income_attributable": _number(source, "n_income_attr_p"),
            "basic_eps": _number(source, "basic_eps"),
            "diluted_eps": _number(source, "diluted_eps"),
        }
    if table == "balance_sheet":
        return {
            "total_assets": _number(source, "total_assets"),
            "total_current_assets": _number(source, "total_cur_assets"),
            "total_non_current_assets": _number(source, "total_nca"),
            "cash_and_equivalents": _number(source, "money_cap"),
            "accounts_receivable": _number(source, "accounts_receiv"),
            "inventory": _number(source, "inventories"),
            "fixed_assets": _number(source, "fix_assets", "fix_assets_total"),
            "intangible_assets": _number(source, "intan_assets"),
            "goodwill": _number(source, "goodwill"),
            "total_liabilities": _number(source, "total_liab"),
            "total_current_liabilities": _number(source, "total_cur_liab"),
            "total_non_current_liabilities": _number(source, "total_ncl"),
            "short_term_borrowing": _number(source, "st_borr"),
            "long_term_borrowing": _number(source, "lt_borr"),
            "accounts_payable": _number(source, "acct_payable"),
            "total_equity": _number(source, "total_hldr_eqy_inc_min_int"),
            "equity_attributable": _number(source, "total_hldr_eqy_exc_min_int"),
            "retained_earnings": _number(source, "undistr_porfit"),
            "minority_interest": _number(source, "minority_int"),
        }
    if table == "cash_flow":
        return {
            "net_operating_cash_flow": _number(source, "n_cashflow_act"),
            "net_investing_cash_flow": _number(source, "n_cashflow_inv_act"),
            "net_financing_cash_flow": _number(source, "n_cash_flows_fnc_act"),
            "capex": _number(source, "c_pay_acq_const_fiolta"),
            "net_cash_change": _number(source, "n_incr_cash_cash_equ"),
        }
    if table == "shares":
        return {
            "total_shares": _number(source, "total_share", scale=10_000),
            "float_shares": _number(source, "float_share", scale=10_000),
        }
    raise TushareError(f"Tushare 不支持财务表: {table}")


def _compress_share_history(records: list[dict]) -> list[dict]:
    """Keep share-capital change points plus the latest daily snapshot."""
    by_symbol: dict[str, list[dict]] = {}
    for record in records:
        by_symbol.setdefault(record["symbol"], []).append(record)
    compact: list[dict] = []
    for rows in by_symbol.values():
        rows.sort(key=lambda row: row["period_end"])
        previous: tuple[float | None, float | None] | None = None
        for index, record in enumerate(rows):
            current = (record.get("total_shares"), record.get("float_shares"))
            if current != previous or index == len(rows) - 1:
                compact.append(record)
            previous = current
    return compact
