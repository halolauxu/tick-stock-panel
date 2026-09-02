from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_microcap_resilience_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_microcap_resilience_discovery", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _observations(momentum_sign: float) -> pl.DataFrame:
    signal = date(2020, 1, 3)
    rows = []
    for index in range(30):
        rows.append(
            {
                "date": signal,
                "entry_date": signal + timedelta(days=3),
                "symbol": f"{index:06d}.SZ",
                "market_cap": float(index + 1),
                "cap_decile": 0,
                "return_5d": float(index - 15) / 100,
                "return_20d": float(index) / 100,
                "return_60d": momentum_sign * float(index + 1) / 100,
                "return_120d": float(index) / 200,
                "mean_amount_20d": float(index + 1) * 1_000_000,
                "volatility_20d": float(30 - index) / 100,
                "net_return": float(index) / 1000,
            }
        )
    return pl.DataFrame(rows)


def test_adaptive_uses_cap_in_positive_state_and_resilience_in_negative() -> None:
    positive = study.build_candidate_sets(_observations(1.0))
    negative = study.build_candidate_sets(_observations(-1.0))

    assert positive["adaptive_resilience"].get_column("symbol").to_list() == (
        positive["cap_smallest"].get_column("symbol").to_list()
    )
    assert negative["adaptive_resilience"].get_column("symbol").to_list() == (
        negative["resilience_composite"].get_column("symbol").to_list()
    )


def test_barbell_keeps_twenty_unique_positions() -> None:
    candidates = study.build_candidate_sets(_observations(1.0))
    barbell = candidates["cap_resilience_barbell"]

    assert barbell.height == study.TARGET_POSITIONS
    assert barbell.get_column("symbol").n_unique() == study.TARGET_POSITIONS
    assert barbell.get_column("selection_rank").to_list() == list(range(1, 21))


def test_screen_requires_2026_above_thirty_and_prior_years_positive() -> None:
    rows = []
    for year in range(2014, 2027):
        weekly_return = 0.31 if year == 2026 else 0.01
        rows.append(
            {
                "date": date(year, 1, 2),
                "entry_date": date(year, 1, 3),
                "symbol": "000001.SZ",
                "net_return": weekly_return,
            }
        )
    candidates = pl.DataFrame(rows)
    result = study.summarize_candidate(candidates)

    assert result["screen_checks"] == {
        "return_2026_above_30pct": False,
        "every_year_2014_2025_positive": True,
    }
    assert result["metrics"]["yearly"][-1]["return"] == pytest.approx(0.0155)
