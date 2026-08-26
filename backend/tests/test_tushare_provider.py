"""Tushare stk_mins plugin contract tests (no real token or network required)."""
from __future__ import annotations

from datetime import datetime

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


def _patch_http(monkeypatch, payload: dict) -> _HTTP:
    fake = _HTTP(payload)
    monkeypatch.setattr(tc.httpx, "Client", lambda **kwargs: fake)
    return fake


def test_client_posts_stk_mins_contract_and_transposes_rows(monkeypatch):
    fields = list(tc.MINUTE_FIELDS)
    fake = _patch_http(monkeypatch, {"code": 0, "msg": None, "data": {"fields": fields, "items": _sample_items()}})
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


def test_client_api_error_never_exposes_token(monkeypatch):
    _patch_http(monkeypatch, {"code": 2002, "msg": "没有接口权限", "data": None})
    client = TushareClient("top-secret-token", min_interval_s=0)
    with pytest.raises(TushareError) as exc:
        client.stock_minutes("000017.SZ")
    assert "2002" in str(exc.value)
    assert "top-secret-token" not in str(exc.value)


class _FakeClient:
    def __init__(self, rows: list[dict] | None = None, error: Exception | None = None) -> None:
        fields = list(tc.MINUTE_FIELDS)
        self.rows = rows if rows is not None else [dict(zip(fields, item, strict=True)) for item in _sample_items()]
        self.error = error
        self.calls: list[dict] = []

    def stock_minutes(self, symbol: str, **kwargs):
        self.calls.append({"symbol": symbol, **kwargs})
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

    assert frame.columns == ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
    assert frame.height == 2
    assert frame.schema["datetime"] == pl.Datetime("us")
    assert frame.schema["volume"] == pl.Float64
    assert frame["datetime"].to_list() == sorted(frame["datetime"].to_list())
    assert frame["volume"].to_list() == [800.0, 1200.0]
    assert frame["amount"].to_list() == [646000.0, 981000.0]
    assert fake.calls[0]["freq"] == "1min"
    assert fake.calls[0]["start_time"] == datetime(2026, 8, 25, 9, 25)
    assert progress == [(1, 1)]


def test_provider_rejects_unpurchased_asset_types():
    with pytest.raises(TushareError, match="仅覆盖 A股历史分钟"):
        TushareProvider().get_minute(["510300.SH"], None, None, asset_type="etf")


def test_provider_empty_symbols_does_not_create_client():
    provider = TushareProvider()
    assert provider.get_minute([], None, None).is_empty()
    assert provider._client is None


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
        lambda *args, **kwargs: _FakeClient(error=TushareError("Tushare API 错误 code=2002: 没有接口权限")),
    )
    ok, reason = tp.probe_api_key("candidate")
    assert ok is False
    assert "2002" in reason


def test_manifest_declares_only_minute_and_ui_token_field():
    from app.data_providers.custom import loader

    manifest = loader.plugin_manifest("tushare")
    assert manifest is not None
    assert manifest["datasets"] == ["minute"]
    assert manifest["api_key_env"] == tp.API_KEY_ENV
