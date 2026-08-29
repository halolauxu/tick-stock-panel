from __future__ import annotations

# Requirements: AM-S1-005 and AM-S2-005.
import pytest

from app.alpha_mining.config_store import AlphaConfigStore


def test_alpha_feature_and_automation_are_independently_default_off(tmp_path) -> None:
    store = AlphaConfigStore(tmp_path)
    assert store.get()["enabled"] is False
    assert store.get()["auto_run_enabled"] is False
    updated = store.update({"enabled": True})
    assert updated["enabled"] is True
    assert updated["auto_run_enabled"] is False


def test_alpha_config_rejects_unknown_or_unsafe_values(tmp_path) -> None:
    store = AlphaConfigStore(tmp_path)
    with pytest.raises(ValueError, match="不支持"):
        store.update({"legacy_mining_enabled": False})
    with pytest.raises(ValueError, match="shadow_min"):
        store.update({"shadow_min_fills": 1})
