"""Minimal HTTP client for the Tushare Pro API.

Authentication, response-envelope parsing and request pacing stay here so the
provider only handles normalization.  The token is never included in logs or
raised error messages.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

import httpx

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
        min_interval_s: float = 0.15,
    ) -> None:
        token = token.strip()
        if not token:
            raise TushareError("未配置 TUSHARE_TOKEN")
        self._token = token
        self._min_interval_s = max(0.0, float(min_interval_s))
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
        if code not in (0, "0"):
            message = str(payload.get("msg") or "未知错误").strip()
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
