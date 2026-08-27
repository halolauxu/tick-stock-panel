"""实时行情生命周期: 应用停机不得篡改用户开关。"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.market_time import CN_TZ
from app.services import preferences
from app.services.quote_service import QuoteService


@pytest.fixture(autouse=True)
def _isolated_preferences(tmp_path, monkeypatch):
    path = tmp_path / "preferences.json"
    monkeypatch.setattr(preferences, "_path", lambda: path)
    preferences._invalidate_cache()
    yield
    preferences._invalidate_cache()


def test_application_shutdown_preserves_enabled_preference() -> None:
    preferences.save({"realtime_quotes_enabled": True})
    service = QuoteService()
    service._running = True
    service._enabled = True

    service.shutdown()

    assert service._running is False
    assert service._enabled is False
    assert preferences.get_realtime_quotes_enabled() is True


def test_user_disable_persists_disabled_preference() -> None:
    preferences.save({"realtime_quotes_enabled": True})
    service = QuoteService()
    service._running = True
    service._enabled = True

    service.disable()

    assert service._running is False
    assert service._enabled is False
    assert preferences.get_realtime_quotes_enabled() is False


def test_boot_check_restores_enabled_service_after_restart(monkeypatch) -> None:
    preferences.save({"realtime_quotes_enabled": True})
    monkeypatch.setattr(QuoteService, "is_realtime_allowed", classmethod(lambda cls: True))
    monkeypatch.setattr(QuoteService, "_poll_loop", lambda self: None)
    service = QuoteService()

    service.boot_check()

    assert service._running is True
    assert service._enabled is True
    assert preferences.get_realtime_quotes_enabled() is True
    service.shutdown()


def test_paper_orders_and_positions_are_merged_into_realtime_scope() -> None:
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ", "600000.SH"}
        )
    )

    assert service._include_paper_symbols(["000001.SZ", "000002.SZ"]) == [
        "000001.SZ",
        "000002.SZ",
        "600000.SH",
    ]


def test_quote_records_are_forwarded_to_paper_execution_and_marks() -> None:
    calls = []
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            on_quote_records=lambda records, source: calls.append((records, source))
        )
    )
    records = [{"symbol": "000001.SZ", "last_price": 10.0}]

    service._notify_paper_trading(records)

    assert calls == [(records, "realtime")]


def test_targeted_paper_refresh_does_not_fetch_full_market(monkeypatch) -> None:
    requested: list[list[str]] = []
    forwarded: list[tuple[list[dict], str]] = []

    class Quotes:
        @staticmethod
        def get(*, symbols):
            requested.append(symbols)
            return [{
                "symbol": symbol,
                "last_price": 10.2,
                "prev_close": 10.0,
                "open": 10.1,
                "high": 10.3,
                "low": 10.0,
                "volume": 1_000,
                "timestamp": 1_788_403_805_000,
            } for symbol in symbols]

    monkeypatch.setattr(
        "app.tickflow.client.get_paid_realtime_client",
        lambda: SimpleNamespace(quotes=Quotes()),
    )
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"},
            on_quote_records=lambda records, source: forwarded.append((records, source)),
        )
    )

    result = service.refresh_paper_symbols()

    assert requested == [["000001.SH", "000001.SZ"]]
    assert result["fetched"] == 2
    assert {row["symbol"] for row in forwarded[0][0]} == {"000001.SH", "000001.SZ"}
    assert forwarded[0][1] == "realtime"


def test_targeted_paper_refresh_reuses_recent_result(monkeypatch) -> None:
    requested: list[list[str]] = []

    class Quotes:
        @staticmethod
        def get(*, symbols):
            requested.append(symbols)
            return [{
                "symbol": symbol,
                "last_price": 10.2,
                "prev_close": 10.0,
                "open": 10.1,
                "high": 10.3,
                "low": 10.0,
                "volume": 1_000,
                "timestamp": 1_788_403_805_000,
            } for symbol in symbols]

    monkeypatch.setattr(
        "app.tickflow.client.get_paid_realtime_client",
        lambda: SimpleNamespace(quotes=Quotes()),
    )
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"},
            on_quote_records=lambda records, source: None,
        )
    )

    first = service.refresh_paper_symbols()
    second = service.refresh_paper_symbols()

    assert len(requested) == 1
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert second["records"] == first["records"]


def test_targeted_paper_refresh_surfaces_partial_provider_failure(monkeypatch) -> None:
    requested: list[list[str]] = []

    class Quotes:
        @staticmethod
        def get(*, symbols):
            requested.append(symbols)
            if len(requested) == 1:
                raise RuntimeError("429 rate limit")
            return [{
                "symbol": symbol,
                "last_price": 10.2,
                "prev_close": 10.0,
                "timestamp": 1_788_403_805_000,
            } for symbol in symbols]

    monkeypatch.setattr(
        "app.tickflow.client.get_paid_realtime_client",
        lambda: SimpleNamespace(quotes=Quotes()),
    )
    monkeypatch.setattr(
        "app.tickflow.rate_limits.sleep_between_batches",
        lambda _index, _rpm: None,
    )
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
            },
            on_quote_records=lambda records, source: None,
        )
    )

    result = service.refresh_paper_symbols()

    assert len(requested) == 2
    assert result["fetched"] == 1
    assert result["errors"] == ["批次 1/2: 429 rate limit"]
    assert len(result["missing"]) == 5


def test_open_execution_priority_suppresses_global_quote_poll(monkeypatch) -> None:
    service = QuoteService()
    service._app_state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"}
        )
    )
    monkeypatch.setattr(
        "app.services.quote_service.cn_now",
        lambda: datetime(2026, 8, 28, 9, 30, 10, tzinfo=CN_TZ),
    )

    assert service._paper_execution_priority_active() is True
