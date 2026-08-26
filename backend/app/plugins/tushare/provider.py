"""Tushare A-share historical-minute and financial provider.

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
from datetime import datetime

import polars as pl

from app import secrets_store
from app.data_providers.base import AssetType
from app.market_time import CN_TZ
from app.plugins.tushare.client import TushareClient, TushareError

logger = logging.getLogger(__name__)

API_KEY_ENV = "TUSHARE_TOKEN"
SECRETS_FIELD = "tushare_api_key"
_DATASETS = ("minute", "financial")
_MINUTE_CANONICAL = [
    "symbol", "datetime", "open", "high", "low", "close", "volume", "amount",
]
_FINANCIAL_NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "metrics": (
        "eps_basic", "eps_diluted", "bps", "ocfps", "roe", "roe_diluted",
        "roa", "gross_margin", "net_margin", "debt_to_asset_ratio",
        "revenue_yoy", "net_income_yoy", "operating_cash_to_revenue",
        "inventory_turnover",
    ),
    "income": (
        "revenue", "operating_cost", "operating_profit", "selling_expense",
        "admin_expense", "rd_expense", "financial_expense",
        "non_operating_income", "non_operating_expense", "total_profit",
        "income_tax", "net_income", "net_income_attributable", "basic_eps",
        "diluted_eps",
    ),
    "balance_sheet": (
        "total_assets", "total_current_assets", "total_non_current_assets",
        "cash_and_equivalents", "accounts_receivable", "inventory",
        "fixed_assets", "intangible_assets", "goodwill", "total_liabilities",
        "total_current_liabilities", "total_non_current_liabilities",
        "short_term_borrowing", "long_term_borrowing", "accounts_payable",
        "total_equity", "equity_attributable", "retained_earnings",
        "minority_interest",
    ),
    "cash_flow": (
        "net_operating_cash_flow", "net_investing_cash_flow",
        "net_financing_cash_flow", "capex", "net_cash_change",
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
            return False, "Token 可访问 Tushare, 但 stk_mins 未返回数据; 请确认已开通 A股历史分钟权限"
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
    display_name: str = "Tushare (A股分钟与财务)"
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

    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,
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
        for symbol in symbols:
            rows = self._get_client().financial_records(table, symbol)
            frame = self._financial_df(table, rows, latest_only=latest_only)
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort(
            ["symbol", "period_end"], descending=[False, True],
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

        frame = pl.DataFrame(records)
        numeric = _FINANCIAL_NUMERIC_FIELDS[table]
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
        if dataset not in _DATASETS:
            raise ValueError(f"Tushare 插件不支持数据集: {dataset}")
        if dataset == "minute":
            frame = self.get_minute(symbols or ["600000.SH"], None, None)
        else:
            frame = self.get_financials("metrics", symbols or ["600000.SH"])
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts() if not frame.is_empty() else [],
        }


def _canonical_date(value: object) -> str | None:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


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
