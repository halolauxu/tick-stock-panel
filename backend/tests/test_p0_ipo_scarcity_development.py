from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_ipo_scarcity_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_ipo_scarcity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_scarcity_features_use_point_in_time_shares_and_listing_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "date": [date(2020, 6, 1)],
            "list_date": [date(2020, 3, 1)],
            "float_shares": [30.0],
            "total_shares": [100.0],
        }
    )
    monkeypatch.setattr(study.shared, "attach_stock_features", lambda value: value)

    result = study.attach_scarcity_features(panel)

    assert result["listing_age_days"][0] == 92
    assert result["float_share_ratio"][0] == pytest.approx(0.30)


def test_contract_constants_define_a_bounded_non_limit_chasing_signal() -> None:
    assert study.MIN_LISTING_AGE_DAYS == 60
    assert study.MAX_LISTING_AGE_DAYS == 365
    assert study.MAX_FLOAT_SHARE_RATIO == 0.50
    assert study.MIN_MOMENTUM_20D == 0.10
    assert study.MAX_MOMENTUM_20D == 0.50
    assert study.MAX_SIGNAL_DAY_RETURN == 0.05


def test_point_in_time_join_keeps_pre_60_day_history_for_feature_warmup(
    tmp_path: Path,
) -> None:
    research = tmp_path / "research"
    shares_dir = tmp_path / "financials" / "shares"
    research.mkdir(parents=True)
    shares_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "list_date": [date(2020, 1, 1)],
            "delist_date": [None],
        }
    ).write_parquet(research / "historical_stock_universe_all_a.parquet")
    pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "name": ["测试股份"],
            "start_date": [date(2020, 1, 1)],
            "end_date": [None],
        }
    ).write_parquet(research / "historical_stock_names_all_a.parquet")
    pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "announce_date": ["2020-01-01"],
            "period_end": ["2019-12-31"],
            "total_shares": [100.0],
            "float_shares": [30.0],
        }
    ).write_parquet(shares_dir / "A.SZ.parquet")
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "date": [date(2020, 1, 10), date(2020, 3, 1)],
        }
    )

    result = study.attach_ipo_point_in_time_data(panel, tmp_path)

    assert result.get_column("date").to_list() == [
        date(2020, 1, 10),
        date(2020, 3, 1),
    ]
