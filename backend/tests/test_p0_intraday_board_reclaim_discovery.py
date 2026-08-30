from __future__ import annotations

import importlib.util
from datetime import date, datetime
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_intraday_board_reclaim_discovery.py"
    )
    spec = importlib.util.spec_from_file_location("p0_board_reclaim", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _row(minute: int, *, high: float, low: float, close: float) -> dict:
    return {
        "datetime": datetime(2026, 1, 5, 10, minute),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "amount": 1_000_000.0,
    }


def test_detect_reclaim_requires_touch_break_reclaim_and_next_minute() -> None:
    rows = [
        _row(0, high=11.0, low=10.9, close=10.95),
        _row(1, high=10.95, low=10.70, close=10.75),
        _row(2, high=10.94, low=10.75, close=10.90),
        _row(3, high=10.98, low=10.85, close=10.95),
    ]

    result = study.detect_reclaim(rows, 11.0)

    assert result is not None
    assert result["signal_datetime"] == datetime(2026, 1, 5, 10, 2)
    assert result["entry_datetime"] == datetime(2026, 1, 5, 10, 3)


def test_detect_reclaim_rejects_path_without_two_percent_break() -> None:
    rows = [
        _row(0, high=11.0, low=10.95, close=10.98),
        _row(1, high=10.99, low=10.85, close=10.90),
        _row(2, high=10.99, low=10.90, close=10.95),
        _row(3, high=10.99, low=10.90, close=10.95),
    ]

    assert study.detect_reclaim(rows, 11.0) is None


def test_summary_does_not_promote_small_sample() -> None:
    event = {
        "date": date(2026, 1, 5),
        "entry_datetime": datetime(2026, 1, 5, 10, 3),
        "tradable": True,
        "entry_valid": True,
        "entry_capacity": True,
        "exit_reason": "filled",
        "net_return": 0.02,
    }

    result = study.summarize([event], {event["entry_datetime"]: 0.0})

    assert result["verdict"] == "TERMINATE"
    assert result["checks"]["at_least_200_tradable_events"] is False
