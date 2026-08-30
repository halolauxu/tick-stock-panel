from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_cb_intraday_diagnostic.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_cb_intraday", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = _load_module()


def test_universe_uses_previous_day_and_excludes_exchangeable_bonds() -> None:
    days = [date(2026, 8, 3), date(2026, 8, 4)]
    basic = pl.DataFrame(
        {
            "symbol": ["110001.SH", "132001.SH"],
            "cb_type": ["CB", "EB"],
            "list_date": [date(2025, 1, 1), date(2025, 1, 1)],
        }
    )
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH", "132001.SH", "110001.SH", "132001.SH"],
            "date": [days[0], days[0], days[1], days[1]],
            "close": [100.0, 100.0, 120.0, 120.0],
            "amount_cny": [50_000_000.0] * 4,
            "cb_over_rate": [10.0] * 4,
        }
    )

    result = diagnostic.build_causal_universe(basic, daily)

    assert result.height == 1
    assert result["symbol"][0] == "110001.SH"
    assert result["date"][0] == days[1]
    assert result["previous_close"][0] == 100.0


def test_prepare_minutes_never_uses_post_close_exchange_records() -> None:
    day = date(2026, 8, 4)
    universe = pl.DataFrame(
        {
            "symbol": ["110001.SH"],
            "date": [day],
            "previous_date": [date(2026, 8, 3)],
            "previous_close": [100.0],
            "previous_amount_cny": [50_000_000.0],
            "previous_cb_over_rate": [10.0],
        }
    )
    rows = []
    for offset in range(17):
        stamp = datetime(2026, 8, 4, 9, 30) + timedelta(minutes=offset)
        rows.append(
            {
                "symbol": "110001.SH",
                "datetime": stamp,
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0 + offset / 100,
                "volume_hands": 100.0,
                "amount_cny": 1_000_000.0,
            }
        )
    rows.append(
        {
            "symbol": "110001.SH",
            "datetime": datetime(2026, 8, 4, 15, 1),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume_hands": 1.0,
            "amount_cny": 10_000.0,
        }
    )

    result = diagnostic.prepare_minutes(pl.DataFrame(rows), universe)

    assert result.height == 17
    assert result.get_column("clock").max().isoformat() == "09:46:00"
    signal = result.filter(pl.col("clock") == datetime(2026, 8, 4, 9, 45).time())
    assert signal["past_5m"][0] == (100.15 / 100.10) - 1.0
    assert signal["entry_open"][0] == 100.0


def test_evaluation_requires_both_halves_and_capacity() -> None:
    rows = []
    for day_offset in range(16):
        current = date(2026, 8, 1) + timedelta(days=day_offset)
        for event in range(15):
            rows.append(
                {
                    "diagnostic": "past_5m",
                    "arm": "bottom20",
                    "date": current,
                    "datetime": datetime.combine(current, datetime.min.time())
                    + timedelta(minutes=event),
                    "symbol": f"110{event:03d}.SH",
                    "gross_return": 0.003,
                    "capacity_cny": 400_000.0,
                    "universe_size": 100,
                }
            )
    observations = pl.DataFrame(rows)
    for diagnostic_name in ("past_15m", "open_gap"):
        for arm in ("bottom20", "top20"):
            observations = pl.concat(
                [
                    observations,
                    observations.filter(pl.col("arm") == "bottom20").with_columns(
                        pl.lit(diagnostic_name).alias("diagnostic"),
                        pl.lit(arm).alias("arm"),
                        pl.lit(-0.001).alias("gross_return"),
                    ),
                ]
            )
    observations = pl.concat(
        [
            observations,
            observations.filter(
                (pl.col("diagnostic") == "past_5m")
                & (pl.col("arm") == "bottom20")
            ).with_columns(
                pl.lit("top20").alias("arm"),
                pl.lit(-0.001).alias("gross_return"),
            ),
        ]
    )

    result = diagnostic.evaluate(observations)

    assert result["promoted_arms"] == ["past_5m:bottom20"]
