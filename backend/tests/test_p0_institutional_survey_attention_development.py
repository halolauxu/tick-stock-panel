from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_institutional_survey_attention_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0_institutional_survey_attention_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_attention_spike_uses_prior_history_and_sixty_day_cooldown() -> None:
    frame = pl.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "symbol": ["000001.SZ"] * 4,
            "notice_date": [
                date(2013, 6, 1),
                date(2014, 1, 10),
                date(2014, 2, 1),
                date(2014, 4, 15),
            ],
            "institution_count": [4, 10, 20, 30],
        }
    )

    selected = study.select_attention_spikes(frame)

    assert selected.get_column("event_id").to_list() == ["b", "d"]
    assert selected.get_column("prior_365d_institution_median").to_list() == [4.0, 10.0]
    assert selected.get_column("attention_multiple").to_list() == [2.5, 3.0]


def test_gate_requires_sample_return_stability_capacity_and_integrity() -> None:
    result = {
        "tradable_events": 500,
        "announcement_days": 300,
        "tradable_rate": 0.90,
        "benchmark_coverage": 0.99,
        "entry_capacity_feasible_rate": 0.95,
        "unresolved_exits": 0,
        "mean_net_return": 0.03,
        "mean_excess_return": 0.02,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 5,
        "max_year_positive_excess_share": 0.40,
    }

    assert study.evaluate_gate(result) is True
    result["mean_net_return"] = 0.0299
    assert study.evaluate_gate(result) is False
