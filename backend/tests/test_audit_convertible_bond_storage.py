from __future__ import annotations

import importlib.util
from datetime import date, datetime
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_convertible_bond_storage.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_cb_storage", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def test_audit_reconciles_daily_and_minute_units() -> None:
    day = date(2026, 8, 28)
    daily = pl.DataFrame(
        {
            "symbol": ["A.SH"],
            "date": [day],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume_hands": [30.0],
            "amount_cny": [300_000.0],
        }
    )
    minute = pl.DataFrame(
        {
            "symbol": ["A.SH", "A.SH"],
            "datetime": [
                datetime(2026, 8, 28, 9, 30),
                datetime(2026, 8, 28, 9, 31),
            ],
            "open": [100.0, 100.1],
            "high": [100.2, 100.2],
            "low": [99.9, 100.0],
            "close": [100.1, 100.1],
            "volume_hands": [10.0, 20.0],
            "amount_cny": [100_000.0, 200_000.0],
        }
    )

    result = audit.audit_frames(daily, minute)

    assert result["decision"]["passed"] is True
    assert result["reconciliation"]["max_volume_relative_error"] == 0.0
    assert result["reconciliation"]["max_amount_relative_error"] == 0.0


def test_audit_fails_when_traded_daily_row_has_no_minute() -> None:
    day = date(2026, 8, 28)
    daily = pl.DataFrame(
        {
            "symbol": ["A.SH", "B.SH"],
            "date": [day, day],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "volume_hands": [30.0, 20.0],
            "amount_cny": [300_000.0, 200_000.0],
        }
    )
    minute = pl.DataFrame(
        {
            "symbol": ["A.SH"],
            "datetime": [datetime(2026, 8, 28, 9, 30)],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume_hands": [30.0],
            "amount_cny": [300_000.0],
        }
    )

    result = audit.audit_frames(daily, minute)

    assert result["decision"]["passed"] is False
    assert result["reconciliation"]["missing_minute_symbol_sessions"][0][
        "symbol"
    ] == "B.SH"
