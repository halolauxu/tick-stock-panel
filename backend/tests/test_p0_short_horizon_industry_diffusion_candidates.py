from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "build_p0_short_horizon_industry_diffusion_candidates.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_short_horizon_industry_diffusion_candidates", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_limits_two_peers_per_source_and_deduplicates_peer_day() -> None:
    study = _load_module()
    frame = pl.DataFrame(
        {
            "source_event_id": ["s1", "s1", "s1", "s2"],
            "source_symbol": ["S1", "S1", "S1", "S2"],
            "ann_date": [date(2020, 1, 1)] * 4,
            "entry_date": [date(2020, 1, 2)] * 4,
            "symbol": ["A", "B", "C", "A"],
            "l1_code": ["I1"] * 4,
            "source_p_change_min": [50.0, 50.0, 50.0, 80.0],
            "prior_roe": [10.0, 9.0, 8.0, 10.0],
            "five_day_industry_residual": [-0.01, -0.02, -0.03, -0.01],
        }
    )

    ranked = study.rank_peer_candidates(frame)

    assert ranked.filter(pl.col("source_event_id") == "s1").height == 1
    assert ranked.filter(pl.col("source_event_id") == "s2")["symbol"].to_list() == ["A"]
    assert set(ranked["symbol"].to_list()) == {"A", "B"}


def test_candidate_gate_requires_broad_point_in_time_sample() -> None:
    study = _load_module()
    passed = study.evaluate_data_gate(
        {
            "candidate_rows": 200,
            "candidate_symbols": 120,
            "source_announcement_days": 60,
            "industries": 10,
            "duplicate_peer_days": 0,
            "future_financial_rows": 0,
            "missing_entry_dates": 0,
        }
    )
    assert passed["verdict"] == "PASS_TO_INDUSTRY_ACCOUNT"

    failed = study.evaluate_data_gate(
        {
            "candidate_rows": 149,
            "candidate_symbols": 120,
            "source_announcement_days": 60,
            "industries": 10,
            "duplicate_peer_days": 0,
            "future_financial_rows": 0,
            "missing_entry_dates": 0,
        }
    )
    assert failed["verdict"] == "BLOCKED_OR_REJECTED_SAMPLE"
    assert "at_least_150_candidate_rows" in failed["failures"]


def test_source_dates_map_to_prior_quote_and_next_entry() -> None:
    study = _load_module()
    sources = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "ann_date": [date(2020, 1, 4)],
            "period_end": [date(2019, 12, 31)],
            "l1_code": ["I1"],
            "l1_name": ["行业"],
            "p_change_min": [10.0],
            "p_change_max": [20.0],
        }
    )

    mapped = study._source_times(
        sources,
        [date(2020, 1, 3), date(2020, 1, 6)],
    )

    assert mapped["signal_quote_date"].to_list() == [date(2020, 1, 3)]
    assert mapped["entry_date"].to_list() == [date(2020, 1, 6)]
