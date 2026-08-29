from __future__ import annotations

from datetime import date, timedelta

from app.alpha_mining.runtime import _aggregate_fold_metrics


def _fold(start: date, days: int, daily_return: float = 0.001) -> dict:
    curve = []
    value = 1.0
    for offset in range(days):
        day = start + timedelta(days=offset)
        curve.append({"date": day.isoformat(), "value": value})
        value *= 1.0 + daily_return
    return {
        "test_start": start.isoformat(),
        "test_end": (start + timedelta(days=days - 1)).isoformat(),
        "metrics": {"equity_curve": curve},
    }


def test_recent_period_metrics_are_not_claimed_without_full_coverage() -> None:
    metrics = _aggregate_fold_metrics([_fold(date(2026, 5, 29), 92)])

    assert metrics["oos_start"] == "2026-05-29"
    assert metrics["oos_end"] == "2026-08-28"
    assert metrics["recent_1y_available"] is False
    assert metrics["recent_3m_available"] is False
    assert metrics["recent_1y_return"] is None
    assert metrics["recent_3m_return"] is None


def test_recent_period_metrics_require_and_report_complete_windows() -> None:
    metrics = _aggregate_fold_metrics([_fold(date(2025, 8, 28), 366)])

    assert metrics["recent_1y_available"] is True
    assert metrics["recent_3m_available"] is True
    assert metrics["recent_1y_return"] is not None
    assert metrics["recent_3m_return"] is not None
    assert metrics["oos_calendar_days"] == 366


def test_missing_backtest_curve_keeps_period_metrics_unavailable() -> None:
    metrics = _aggregate_fold_metrics([
        {
            "test_start": "2025-08-28",
            "test_end": "2026-08-28",
            "error": "inner selection produced no finite candidate",
        },
    ])

    assert metrics["oos_start"] is None
    assert metrics["oos_end"] is None
    assert metrics["recent_1y_return"] is None
    assert metrics["recent_3m_return"] is None
