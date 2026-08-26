"""实时行情生命周期: 应用停机不得篡改用户开关。"""
from __future__ import annotations

import pytest

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
