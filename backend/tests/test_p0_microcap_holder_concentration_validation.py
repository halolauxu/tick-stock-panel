from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_holder_concentration_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_holder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_filter_uses_last_preannouncement_cap_snapshot_and_bottom_decile():
    events = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "ann_date": [date(2022, 1, 10), date(2022, 1, 10)],
            "holder_count_change": [-0.2, -0.2],
        }
    )
    snapshots = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "date": [date(2022, 1, 7), date(2022, 1, 11), date(2022, 1, 7)],
            "market_cap": [1.0, 0.5, 2.0],
            "cap_rank": [1, 1, 20],
            "cap_universe_size": [100, 100, 100],
            "cap_decile": [0, 0, 1],
        }
    )

    filtered = study.attach_microcap_filter(events, snapshots)

    assert filtered.height == 1
    row = filtered.row(0, named=True)
    assert row["symbol"] == "000001.SZ"
    assert row["date"] == date(2022, 1, 7)
    assert row["cap_snapshot_age_days"] == 3


def test_validation_gate_requires_zero_unresolved_exits():
    metrics = {
        "tradable_events": 150,
        "announcement_days": 100,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_return": 0.041,
        "mean_excess_return": 0.026,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 2,
        "max_year_positive_excess_share": 0.55,
    }

    assert study.validation_passed(metrics) is True
    assert study.validation_passed({**metrics, "unresolved_exits": 1}) is False
