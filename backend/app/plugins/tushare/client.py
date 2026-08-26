"""Minimal HTTP client for the Tushare Pro API.

Authentication, response-envelope parsing and request pacing stay here so the
provider only handles normalization.  The token is never included in logs or
raised error messages.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.tushare.pro"
MINUTE_FIELDS = (
    "ts_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)

AUCTION_FIELDS = (
    "ts_code",
    "trade_date",
    "close",
    "open",
    "high",
    "low",
    "vol",
    "amount",
    "vwap",
)

IRM_QA_FIELDS: dict[str, tuple[str, ...]] = {
    "sh": ("ts_code", "name", "trade_date", "q", "a", "pub_time"),
    "sz": ("ts_code", "name", "trade_date", "q", "a", "pub_time", "industry"),
}

MAIN_BUSINESS_FIELDS = (
    "ts_code",
    "end_date",
    "bz_item",
    "bz_sales",
    "bz_profit",
    "bz_cost",
    "curr_type",
    "update_flag",
)

FINANCIAL_FIELDS: dict[str, tuple[str, ...]] = {
    "metrics": (
        "ts_code", "ann_date", "end_date", "update_flag",
        "eps", "dt_eps", "bps", "ocfps", "roe", "roe_dt", "roa",
        "grossprofit_margin", "netprofit_margin", "debt_to_assets",
        "or_yoy", "netprofit_yoy", "ocf_to_or", "inv_turn",
    ),
    "income": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "comp_type", "update_flag", "revenue", "oper_cost",
        "operate_profit", "sell_exp", "admin_exp", "rd_exp", "fin_exp",
        "non_oper_income", "non_oper_exp", "total_profit", "income_tax",
        "n_income", "n_income_attr_p", "basic_eps", "diluted_eps",
    ),
    "balance_sheet": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "comp_type", "update_flag", "total_assets", "total_cur_assets",
        "total_nca", "money_cap", "accounts_receiv", "inventories",
        "fix_assets", "fix_assets_total", "intan_assets", "goodwill",
        "total_liab", "total_cur_liab", "total_ncl", "st_borr", "lt_borr",
        "acct_payable", "total_hldr_eqy_inc_min_int",
        "total_hldr_eqy_exc_min_int", "undistr_porfit", "minority_int",
    ),
    "cash_flow": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "comp_type", "update_flag", "n_cashflow_act", "n_cashflow_inv_act",
        "n_cash_flows_fnc_act", "c_pay_acq_const_fiolta",
        "n_incr_cash_cash_equ",
    ),
    "shares": (
        "ts_code", "trade_date", "total_share", "float_share",
    ),
}

_FINANCIAL_APIS = {
    "metrics": "fina_indicator",
    "income": "income",
    "balance_sheet": "balancesheet",
    "cash_flow": "cashflow",
    "shares": "daily_basic",
}


class TushareError(RuntimeError):
    """Tushare configuration, transport or API-contract error."""


class TushareClient:
    """Thread-safe Tushare HTTP client with a conservative shared request pace."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        min_interval_s: float = 0.32,
        rate_limit_pause_s: float = 61.0,
        max_rate_limit_retries: int = 2,
    ) -> None:
        token = token.strip()
        if not token:
            raise TushareError("未配置 TUSHARE_TOKEN")
        self._token = token
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._rate_limit_pause_s = max(0.0, float(rate_limit_pause_s))
        self._max_rate_limit_retries = max(0, int(max_rate_limit_retries))
        self._pace_lock = threading.Lock()
        self._last_request_at = 0.0
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    def _wait_for_turn(self) -> None:
        with self._pace_lock:
            remaining = self._min_interval_s - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def query(self, api_name: str, params: dict, fields: tuple[str, ...]) -> list[dict]:
        """Call one Tushare endpoint and transpose ``fields`` + ``items`` into rows."""
        body = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
            "fields": ",".join(fields),
        }
        payload: dict = {}
        for attempt in range(self._max_rate_limit_retries + 1):
            self._wait_for_turn()
            try:
                response = self._http.post("/", json=body)
            except httpx.HTTPError as exc:
                raise TushareError(f"Tushare 网络请求失败: {exc}") from exc
            if response.status_code != 200:
                raise TushareError(f"Tushare HTTP {response.status_code}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise TushareError("Tushare 响应不是 JSON") from exc
            if not isinstance(payload, dict):
                raise TushareError("Tushare 响应结构无效")

            code = payload.get("code")
            if code in (0, "0"):
                break
            message = str(payload.get("msg") or "未知错误").strip()
            if "频率超限" in message and attempt < self._max_rate_limit_retries:
                logger.warning(
                    "Tushare %s rate limited; retrying after %.0fs (%d/%d)",
                    api_name,
                    self._rate_limit_pause_s,
                    attempt + 1,
                    self._max_rate_limit_retries,
                )
                time.sleep(self._rate_limit_pause_s)
                continue
            raise TushareError(f"Tushare API 错误 code={code}: {message[:500]}")

        data = payload.get("data") or {}
        if not data:
            return []
        response_fields = data.get("fields")
        items = data.get("items")
        if not isinstance(response_fields, list) or not isinstance(items, list):
            raise TushareError("Tushare data 缺少 fields/items")

        rows: list[dict] = []
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != len(response_fields):
                raise TushareError("Tushare fields/items 列数不一致")
            rows.append(dict(zip(response_fields, item, strict=True)))
        return rows

    def stock_minutes(
        self,
        symbol: str,
        *,
        freq: str = "1min",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {"ts_code": symbol, "freq": freq}
        if start_time is not None:
            params["start_date"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if end_time is not None:
            params["end_date"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
        return self.query("stk_mins", params, MINUTE_FIELDS)

    def financial_records(self, table: str, symbol: str) -> list[dict]:
        """Fetch one stock's standard (non-VIP) financial dataset."""
        try:
            api_name = _FINANCIAL_APIS[table]
            fields = FINANCIAL_FIELDS[table]
        except KeyError as exc:
            raise TushareError(f"Tushare 不支持财务表: {table}") from exc
        return self.query(api_name, {"ts_code": symbol}, fields)

    def main_business_records(self, symbol: str, *, kind: str = "P") -> list[dict]:
        """Fetch product/region/industry main-business composition for one stock."""
        normalized = kind.strip().upper()
        if normalized not in {"P", "D", "I"}:
            raise TushareError(f"Tushare 主营构成 type 不支持: {kind}")
        return self.query(
            "fina_mainbz",
            {"ts_code": symbol, "type": normalized},
            MAIN_BUSINESS_FIELDS,
        )

    def auction_records(self, session: str, trade_date: date) -> list[dict]:
        """Fetch one market-wide opening or closing auction snapshot."""
        api_name = {
            "open": "stk_auction_o",
            "close": "stk_auction_c",
        }.get(session)
        if api_name is None:
            raise TushareError(f"Tushare 不支持集合竞价时段: {session}")
        return self.query(
            api_name,
            {"trade_date": trade_date.strftime("%Y%m%d")},
            AUCTION_FIELDS,
        )

    def irm_qa_records(
        self,
        exchange: str,
        *,
        pub_start: date,
        pub_end: date,
    ) -> list[dict]:
        """Fetch newly published exchange IR questions and answers."""
        normalized = exchange.strip().lower()
        if normalized not in IRM_QA_FIELDS:
            raise TushareError(f"Tushare 不支持董秘问答交易所: {exchange}")
        return self.query(
            f"irm_qa_{normalized}",
            {
                # irm_qa_* 的 pub_start/pub_end 是发布时间，而不是 YYYYMMDD
                # 日期参数。使用完整日边界，避免服务端将无效格式按空结果处理。
                "pub_start": pub_start.strftime("%Y-%m-%d 00:00:00"),
                "pub_end": pub_end.strftime("%Y-%m-%d 23:59:59"),
            },
            IRM_QA_FIELDS[normalized],
        )
