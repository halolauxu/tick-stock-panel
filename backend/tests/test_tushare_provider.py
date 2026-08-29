"""Tushare stk_mins plugin contract tests (no real token or network required)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.plugins.tushare import client as tc
from app.plugins.tushare import provider as tp
from app.plugins.tushare.client import TushareClient, TushareError
from app.plugins.tushare.provider import TushareProvider


def _sample_items() -> list[list]:
    return [
        ["000017.SZ", "2026-08-25 09:31:00", 8.10, 8.20, 8.08, 8.18, 120000, 981000.0],
        ["000017.SZ", "2026-08-25 09:30:00", 8.00, 8.12, 8.00, 8.10, 80000, 646000.0],
    ]


class _Response:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self):
        return self._payload


class _HTTP:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, json: dict):
        self.calls.append((path, json))
        return _Response(self.payload)

    def close(self):
        pass


class _HTTPSequence(_HTTP):
    def __init__(self, payloads: list[dict]) -> None:
        super().__init__({})
        self.payloads = list(payloads)

    def post(self, path: str, json: dict):
        self.calls.append((path, json))
        return _Response(self.payloads.pop(0))


def _patch_http(monkeypatch, payload: dict) -> _HTTP:
    fake = _HTTP(payload)
    monkeypatch.setattr(tc.httpx, "Client", lambda **kwargs: fake)
    return fake


def test_client_posts_stk_mins_contract_and_transposes_rows(monkeypatch):
    fields = list(tc.MINUTE_FIELDS)
    fake = _patch_http(
        monkeypatch, {"code": 0, "msg": None, "data": {"fields": fields, "items": _sample_items()}}
    )
    client = TushareClient("secret-token", min_interval_s=0)

    rows = client.stock_minutes(
        "000017.SZ",
        start_time=datetime(2026, 8, 25, 9, 25),
        end_time=datetime(2026, 8, 25, 15, 5),
    )

    assert rows[0]["ts_code"] == "000017.SZ"
    _, body = fake.calls[0]
    assert body["api_name"] == "stk_mins"
    assert body["params"] == {
        "ts_code": "000017.SZ",
        "freq": "1min",
        "start_date": "2026-08-25 09:25:00",
        "end_date": "2026-08-25 15:05:00",
    }
    assert body["fields"] == ",".join(tc.MINUTE_FIELDS)


def test_client_posts_adj_factor_contract(monkeypatch):
    fields = list(tc.ADJ_FACTOR_FIELDS)
    fake = _patch_http(
        monkeypatch,
        {
            "code": 0,
            "msg": None,
            "data": {
                "fields": fields,
                "items": [["600000.SH", "20260827", 17.3774]],
            },
        },
    )
    client = TushareClient("secret-token", min_interval_s=0)

    rows = client.adjustment_factors(
        "600000.SH",
        start_time=datetime(2019, 8, 28),
        end_time=datetime(2026, 8, 27),
    )

    assert rows == [
        {"ts_code": "600000.SH", "trade_date": "20260827", "adj_factor": 17.3774}
    ]
    _, body = fake.calls[0]
    assert body["api_name"] == "adj_factor"
    assert body["params"] == {
        "ts_code": "600000.SH",
        "start_date": "20190828",
        "end_date": "20260827",
    }
    assert body["fields"] == ",".join(tc.ADJ_FACTOR_FIELDS)


def test_client_posts_adj_factor_date_contract(monkeypatch):
    fields = list(tc.ADJ_FACTOR_FIELDS)
    fake = _patch_http(
        monkeypatch,
        {
            "code": 0,
            "msg": None,
            "data": {"fields": fields, "items": [["600000.SH", "20260827", 17.3774]]},
        },
    )
    client = TushareClient("secret-token", min_interval_s=0)

    rows = client.adjustment_factors_by_date(date(2026, 8, 27))

    assert rows[0]["ts_code"] == "600000.SH"
    _, body = fake.calls[0]
    assert body["params"] == {"trade_date": "20260827"}


def test_client_api_error_never_exposes_token(monkeypatch):
    _patch_http(monkeypatch, {"code": 2002, "msg": "没有接口权限", "data": None})
    client = TushareClient("top-secret-token", min_interval_s=0)
    with pytest.raises(TushareError) as exc:
        client.stock_minutes("000017.SZ")
    assert "2002" in str(exc.value)
    assert "top-secret-token" not in str(exc.value)


def test_client_waits_and_retries_tushare_rate_limit(monkeypatch):
    fields = list(tc.MINUTE_FIELDS)
    fake = _HTTPSequence(
        [
            {"code": 40203, "msg": "访问接口频率超限(200次/分钟)", "data": None},
            {"code": 0, "msg": None, "data": {"fields": fields, "items": _sample_items()}},
        ]
    )
    monkeypatch.setattr(tc.httpx, "Client", lambda **kwargs: fake)
    pauses: list[float] = []
    monkeypatch.setattr(tc.time, "sleep", lambda seconds: pauses.append(seconds))
    client = TushareClient(
        "secret-token",
        min_interval_s=0,
        rate_limit_pause_s=61,
        max_rate_limit_retries=1,
    )

    rows = client.stock_minutes("000017.SZ")

    assert len(rows) == 2
    assert len(fake.calls) == 2
    assert pauses == [61]


class _FakeClient:
    def __init__(self, rows: list[dict] | None = None, error: Exception | None = None) -> None:
        fields = list(tc.MINUTE_FIELDS)
        self.rows = (
            rows
            if rows is not None
            else [dict(zip(fields, item, strict=True)) for item in _sample_items()]
        )
        self.error = error
        self.calls: list[dict] = []

    def stock_minutes(self, symbol: str, **kwargs):
        self.calls.append({"symbol": symbol, **kwargs})
        if self.error:
            raise self.error
        return self.rows

    def adjustment_factors(self, symbol: str, **kwargs):
        self.calls.append({"symbol": symbol, **kwargs})
        if self.error:
            raise self.error
        return self.rows

    def adjustment_factors_by_date(self, trade_date: date):
        self.calls.append({"trade_date": trade_date})
        if self.error:
            raise self.error
        return self.rows

    def financial_records(self, table: str, symbol: str):
        self.calls.append({"table": table, "symbol": symbol})
        if self.error:
            raise self.error
        return self.rows

    def close(self):
        pass


def test_provider_normalizes_schema_orders_rows_and_converts_cn_window():
    provider = TushareProvider()
    fake = _FakeClient()
    provider._client = fake
    progress: list[tuple[int, int]] = []

    frame = provider.get_minute(
        ["000017.SZ"],
        datetime(2026, 8, 25, 9, 25, tzinfo=CN_TZ),
        datetime(2026, 8, 25, 15, 5, tzinfo=CN_TZ),
        on_chunk_done=lambda current, total: progress.append((current, total)),
    )

    assert frame.columns == [
        "symbol",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert frame.height == 2
    assert frame.schema["datetime"] == pl.Datetime("us")
    assert frame.schema["volume"] == pl.Float64
    assert frame["datetime"].to_list() == sorted(frame["datetime"].to_list())
    assert frame["volume"].to_list() == [800.0, 1200.0]
    assert frame["amount"].to_list() == [646000.0, 981000.0]
    assert fake.calls[0]["freq"] == "1min"
    assert fake.calls[0]["start_time"] == datetime(2026, 8, 25, 9, 25)
    assert progress == [(1, 1)]


def test_provider_filters_bse_post_market_minutes():
    fields = list(tc.MINUTE_FIELDS)
    rows = [
        dict(zip(fields, item, strict=True))
        for item in (
            ["920001.BJ", "2026-08-25 15:00:00", 10, 10, 10, 10, 100, 1000],
            ["920001.BJ", "2026-08-25 15:01:00", 10, 10, 10, 10, 100, 1000],
            ["920001.BJ", "2026-08-25 15:30:00", 10, 10, 10, 10, 100, 1000],
            ["920002.BJ", "2026-08-25 09:30:00", 0, 0, 0, 0, 0, 0],
        )
    ]

    frame = TushareProvider._minute_df(rows)

    assert frame.height == 1
    assert frame["datetime"].item() == datetime(2026, 8, 25, 15, 0)


def test_provider_rejects_unpurchased_asset_types():
    with pytest.raises(TushareError, match="仅覆盖 A股历史分钟"):
        TushareProvider().get_minute(["510300.SH"], None, None, asset_type="etf")


def test_provider_empty_symbols_does_not_create_client():
    provider = TushareProvider()
    assert provider.get_minute([], None, None).is_empty()
    assert provider._client is None


def test_provider_converts_cumulative_adjustment_factors_to_sparse_event_ratios():
    provider = TushareProvider()
    fake = _FakeClient(
        rows=[
            {"ts_code": "600000.SH", "trade_date": "20200103", "adj_factor": 12.0},
            {"ts_code": "600000.SH", "trade_date": "20200102", "adj_factor": 10.0},
            {"ts_code": "600000.SH", "trade_date": "20200106", "adj_factor": 12.0},
            {"ts_code": "600000.SH", "trade_date": "20200107", "adj_factor": 15.0},
        ]
    )
    provider._client = fake
    progress: list[tuple[int, int]] = []

    frame = provider.get_adj_factors(
        ["600000.SH"],
        datetime(2020, 1, 1),
        datetime(2020, 1, 31),
        on_chunk_done=lambda current, total: progress.append((current, total)),
    )

    assert frame.columns == [
        "symbol", "asset_type", "source", "trade_date", "ex_factor",
    ]
    assert frame["trade_date"].to_list() == [date(2020, 1, 3), date(2020, 1, 7)]
    assert frame["ex_factor"].to_list() == pytest.approx([1.2, 1.25])
    assert progress == [(1, 1)]
    assert fake.calls[0] == {
        "symbol": "600000.SH",
        "start_time": datetime(2020, 1, 1),
        "end_time": datetime(2020, 1, 31),
    }


def test_provider_adj_factor_rejects_non_stock_assets():
    with pytest.raises(TushareError, match="仅支持 A股"):
        TushareProvider().get_adj_factors(["510300.SH"], None, None, asset_type="etf")


def test_provider_uses_date_queries_for_full_universe_and_keeps_prior_context():
    provider = TushareProvider()
    fake = _FakeClient(rows=[])

    def rows_for_date(trade_date: date):
        fake.calls.append({"trade_date": trade_date})
        factor = 10.0 if trade_date < date(2026, 8, 25) else 12.0
        return [
            {"ts_code": symbol, "trade_date": trade_date.strftime("%Y%m%d"), "adj_factor": factor}
            for symbol in ("600000.SH", "000001.SZ")
        ]

    fake.adjustment_factors_by_date = rows_for_date
    provider._client = fake
    symbols = ["600000.SH", "000001.SZ"] + [f"600{i:03d}.SH" for i in range(19)]

    frame = provider.get_adj_factors(
        symbols,
        datetime(2026, 8, 25),
        datetime(2026, 8, 27),
    )

    assert frame.height == 2
    assert set(frame["symbol"].to_list()) == {"600000.SH", "000001.SZ"}
    assert set(frame["trade_date"].to_list()) == {date(2026, 8, 25)}
    assert frame["ex_factor"].to_list() == pytest.approx([1.2, 1.2])
    assert len(fake.calls) == 24


def test_provider_streams_bounded_symbol_batches():
    provider = TushareProvider()
    fake = _FakeClient()
    provider._client = fake
    emitted: list[pl.DataFrame] = []
    progress: list[tuple[int, int]] = []

    provider.stream_minute(
        [f"00000{i}.SZ" for i in range(5)],
        datetime(2026, 8, 25),
        datetime(2026, 8, 26),
        batch_symbols=2,
        on_batch=emitted.append,
        on_chunk_done=lambda current, total: progress.append((current, total)),
    )

    assert len(emitted) == 3
    assert all(frame.height <= 2 for frame in emitted)
    assert len(fake.calls) == 5
    assert progress[-1] == (5, 5)


def test_get_api_key_uses_secrets_store_before_environment(monkeypatch):
    monkeypatch.setenv(tp.API_KEY_ENV, "token-from-env")
    monkeypatch.setattr(tp.secrets_store, "load", lambda: {tp.SECRETS_FIELD: "token-from-ui"})
    assert tp.get_api_key() == "token-from-ui"


def test_availability_requires_token(monkeypatch):
    monkeypatch.setattr(tp, "get_api_key", lambda: "")
    ok, reason = tp.availability()
    assert ok is False
    assert tp.API_KEY_ENV in reason


def test_probe_validates_stk_mins_permission(monkeypatch):
    monkeypatch.setattr(tp, "TushareClient", lambda *args, **kwargs: _FakeClient())
    assert tp.probe_api_key("candidate") == (True, "ok")


def test_probe_rejects_missing_permission(monkeypatch):
    monkeypatch.setattr(
        tp,
        "TushareClient",
        lambda *args, **kwargs: _FakeClient(
            error=TushareError("Tushare API 错误 code=2002: 没有接口权限")
        ),
    )
    ok, reason = tp.probe_api_key("candidate")
    assert ok is False
    assert "2002" in reason


def test_manifest_declares_adj_factor_minute_financial_and_ui_token_field():
    from app.data_providers.custom import loader

    manifest = loader.plugin_manifest("tushare")
    assert manifest is not None
    assert manifest["datasets"] == ["adj_factor", "minute", "financial"]
    assert manifest["api_key_env"] == tp.API_KEY_ENV


def test_client_posts_standard_financial_contract(monkeypatch):
    fields = list(tc.FINANCIAL_FIELDS["metrics"])
    fake = _patch_http(
        monkeypatch,
        {
            "code": 0,
            "data": {
                "fields": fields,
                "items": [["600000.SH", "20260430", "20260331", "1", *([1.0] * (len(fields) - 4))]],
            },
        },
    )
    client = TushareClient("secret-token", min_interval_s=0)

    rows = client.financial_records("metrics", "600000.SH")

    assert rows[0]["ts_code"] == "600000.SH"
    _, body = fake.calls[0]
    assert body["api_name"] == "fina_indicator"
    assert body["params"] == {"ts_code": "600000.SH"}
    assert body["fields"] == ",".join(tc.FINANCIAL_FIELDS["metrics"])


def test_client_posts_main_business_contract(monkeypatch):
    fake = _patch_http(
        monkeypatch,
        {"code": 0, "data": {"fields": [], "items": []}},
    )
    client = TushareClient("secret-token", min_interval_s=0)

    client.main_business_records("600000.SH", kind="P")

    body = fake.calls[0][1]
    assert body["api_name"] == "fina_mainbz"
    assert body["params"] == {"ts_code": "600000.SH", "type": "P"}
    assert body["fields"] == ",".join(tc.MAIN_BUSINESS_FIELDS)


def test_client_posts_auction_and_irm_qa_contracts(monkeypatch):
    fake = _patch_http(
        monkeypatch,
        {"code": 0, "data": {"fields": [], "items": []}},
    )
    client = TushareClient("secret-token", min_interval_s=0)

    client.auction_records("open", date(2026, 8, 26))
    client.irm_qa_records(
        "sz",
        pub_start=date(2026, 8, 24),
        pub_end=date(2026, 8, 26),
    )

    auction = fake.calls[0][1]
    assert auction["api_name"] == "stk_auction_o"
    assert auction["params"] == {"trade_date": "20260826"}
    assert auction["fields"] == ",".join(tc.AUCTION_FIELDS)
    qa = fake.calls[1][1]
    assert qa["api_name"] == "irm_qa_sz"
    assert qa["params"] == {
        "pub_start": "2026-08-24 00:00:00",
        "pub_end": "2026-08-26 23:59:59",
    }
    assert qa["fields"] == ",".join(tc.IRM_QA_FIELDS["sz"])


def test_provider_normalizes_auction_and_irm_qa():
    class _SupplementalClient(_FakeClient):
        def auction_records(self, session: str, trade_date: date):
            return [{
                "ts_code": "600000.SH",
                "trade_date": "20260826",
                "close": 10.2,
                "open": 10.0,
                "high": 10.3,
                "low": 9.9,
                "vol": 123400,
                "amount": 1_250_000,
                "vwap": 10.13,
            }]

        def irm_qa_records(self, exchange: str, *, pub_start: date, pub_end: date):
            return [{
                "ts_code": "002254",
                "name": "泰和新材",
                "trade_date": "20260820",
                "q": "问题",
                "a": "回复",
                "pub_time": "2026-08-26 18:02:00",
                "industry": "化工",
            }]

    provider = TushareProvider()
    provider._client = _SupplementalClient()

    auction = provider.get_auction(date(2026, 8, 26), "open")
    qa = provider.get_irm_qa(
        "sz",
        pub_start=date(2026, 8, 24),
        pub_end=date(2026, 8, 26),
    )

    assert auction.columns == tp._AUCTION_CANONICAL
    assert auction.row(0, named=True)["volume_shares"] == 123400
    assert auction.row(0, named=True)["session"] == "open"
    assert qa.columns == tp._IRM_QA_CANONICAL
    assert qa.row(0, named=True)["symbol"] == "002254.SZ"
    assert qa.row(0, named=True)["date"] == "2026-08-26"
    assert qa.row(0, named=True)["trade_date"] == "2026-08-20"


class _FinancialClient(_FakeClient):
    def __init__(self, table_rows: dict[str, list[dict]]) -> None:
        super().__init__(rows=[])
        self.table_rows = table_rows

    def financial_records(self, table: str, symbol: str):
        self.calls.append({"table": table, "symbol": symbol})
        return self.table_rows.get(table, [])


def test_provider_normalizes_metrics_and_prefers_current_update():
    provider = TushareProvider()
    provider._client = _FinancialClient(
        {
            "metrics": [
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "0",
                    "eps": -9,
                    "ocf_to_or": 0.01,
                },
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "1",
                    "eps": -0.08,
                    "ocf_to_or": 0.0859,
                },
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20251031",
                    "end_date": "20250930",
                    "update_flag": "1",
                    "eps": 0.30,
                    "ocf_to_or": 0.0761,
                },
            ],
        }
    )

    frame = provider.get_financials("metrics", ["603800.SH"], latest_only=False)

    assert frame.height == 2
    latest = frame.filter(pl.col("period_end") == "2026-03-31").row(0, named=True)
    assert latest["eps_basic"] == pytest.approx(-0.08)
    assert latest["operating_cash_to_revenue"] == pytest.approx(8.59)
    assert frame.schema["roe"] == pl.Float64


def test_provider_reports_live_financial_progress():
    provider = TushareProvider()
    provider._client = _FinancialClient(
        {
            "income": [
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "1",
                    "revenue": 10,
                }
            ],
        }
    )
    progress: list[tuple[int, int, int, int]] = []

    frame = provider.get_financials(
        "income",
        ["600000.SH", "000001.SZ"],
        on_progress=lambda *values: progress.append(values),
    )

    assert frame.height == 2
    assert progress == [(1, 2, 1, 0), (2, 2, 2, 0)]


def test_provider_maps_statements_to_panel_schema():
    provider = TushareProvider()
    provider._client = _FinancialClient(
        {
            "income": [
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20260429",
                    "f_ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "1",
                    "revenue": 10,
                    "oper_cost": 6,
                    "n_income": 2,
                    "n_income_attr_p": 1.8,
                }
            ],
            "balance_sheet": [
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "1",
                    "total_assets": 100,
                    "total_liab": 60,
                    "fix_assets": None,
                    "fix_assets_total": 12,
                }
            ],
            "cash_flow": [
                {
                    "ts_code": "603800.SH",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "update_flag": "1",
                    "n_cashflow_act": 8,
                    "n_cashflow_inv_act": -3,
                    "n_cash_flows_fnc_act": -2,
                    "c_pay_acq_const_fiolta": 4,
                    "n_incr_cash_cash_equ": 3,
                }
            ],
        }
    )

    income = provider.get_financials("income", ["603800.SH"])
    balance = provider.get_financials("balance_sheet", ["603800.SH"])
    cash = provider.get_financials("cash_flow", ["603800.SH"])

    assert income.row(0, named=True)["announce_date"] == "2026-04-30"
    assert income.row(0, named=True)["net_income_attributable"] == 1.8
    assert balance.row(0, named=True)["fixed_assets"] == 12
    assert cash.row(0, named=True)["net_operating_cash_flow"] == 8


def test_provider_pins_numeric_schema_beyond_default_inference_window():
    start = date(1990, 1, 1)
    rows = []
    for index in range(121):
        period = (start + timedelta(days=index)).strftime("%Y%m%d")
        rows.append(
            {
                "ts_code": "600000.SH",
                "ann_date": period,
                "end_date": period,
                "update_flag": "1",
                "revenue": 1 if index < 100 else 60_894.44,
            }
        )
    provider = TushareProvider()
    provider._client = _FinancialClient({"income": rows})

    frame = provider.get_financials("income", ["600000.SH"], latest_only=False)

    assert frame.height == 121
    assert frame.schema["revenue"] == pl.Float64
    assert frame["revenue"].max() == pytest.approx(60_894.44)


def test_provider_converts_and_compresses_daily_share_capital():
    provider = TushareProvider()
    provider._client = _FinancialClient(
        {
            "shares": [
                {
                    "ts_code": "603800.SH",
                    "trade_date": "20260825",
                    "total_share": 20783.2,
                    "float_share": 20783.2,
                },
                {
                    "ts_code": "603800.SH",
                    "trade_date": "20260824",
                    "total_share": 20783.2,
                    "float_share": 20783.2,
                },
                {
                    "ts_code": "603800.SH",
                    "trade_date": "20250102",
                    "total_share": 20000,
                    "float_share": 18000,
                },
                {
                    "ts_code": "603800.SH",
                    "trade_date": "20250101",
                    "total_share": 20000,
                    "float_share": 18000,
                },
            ],
        }
    )

    frame = provider.get_financials("shares", ["603800.SH"], latest_only=False)

    assert frame.height == 3
    latest = frame.filter(pl.col("period_end") == "2026-08-25").row(0, named=True)
    assert latest["total_shares"] == pytest.approx(207_832_000)
    assert latest["float_shares"] == pytest.approx(207_832_000)
