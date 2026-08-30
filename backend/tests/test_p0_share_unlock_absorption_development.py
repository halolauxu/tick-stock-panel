from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_share_unlock_absorption_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_share_unlock", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _detail(symbol: str, event_date: date, ratio: float = 3.0) -> dict:
    return {
        "symbol": symbol,
        "ann_date": event_date - timedelta(days=30),
        "float_date": event_date,
        "float_shares": 1_000_000.0,
        "float_ratio": ratio,
        "holder_name": "股东",
        "share_type": "限售股份",
    }


def test_aggregate_events_sums_detail_ratio_and_classifies_absorption():
    event_date = date(2020, 6, 1)
    details = pl.DataFrame(
        [_detail("000001.SZ", event_date), _detail("000001.SZ", event_date)]
    )
    event_panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "float_date": [event_date],
            "event_return": [0.02],
            "amount_multiple": [1.6],
        }
    )

    events = study.aggregate_events(details, event_panel)

    assert events.height == 1
    row = events.row(0, named=True)
    assert row["float_ratio_pct"] == 6.0
    assert row["float_shares"] == 2_000_000.0
    assert row["category"] == "absorbed_unlock"
    assert row["ann_date"] == event_date


def test_aggregate_events_uses_one_cooldown_across_categories():
    first = date(2020, 1, 2)
    second = date(2020, 3, 2)
    details = pl.DataFrame(
        [_detail("000001.SZ", first, 6.0), _detail("000001.SZ", second, 6.0)]
    )
    event_panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "float_date": [first, second],
            "event_return": [0.02, -0.01],
            "amount_multiple": [1.6, 0.8],
        }
    )

    events = study.aggregate_events(details, event_panel)

    assert events.height == 1
    assert events.row(0, named=True)["ann_date"] == first


def test_build_event_day_panel_excludes_current_amount_from_baseline():
    start = date(2020, 1, 1)
    rows = []
    for offset in range(21):
        rows.append(
            {
                "symbol": "000001.SZ",
                "date": start + timedelta(days=offset),
                "close": 10.0 if offset < 20 else 10.2,
                "amount": 100.0 if offset < 20 else 200.0,
            }
        )

    panel = study.build_event_day_panel(pl.DataFrame(rows))
    last = panel.row(-1, named=True)

    assert round(last["event_return"], 6) == 0.02
    assert last["amount_multiple"] == 2.0
