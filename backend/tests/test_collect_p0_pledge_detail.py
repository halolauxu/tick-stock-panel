from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "collect_p0_pledge_detail", ROOT / "research" / "collect_p0_pledge_detail.py"
)
assert SPEC and SPEC.loader
collector = module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ts_code": "600001.SH",
        "ann_date": "20180105",
        "holder_name": "股东",
        "pledge_amount": 100.0,
        "start_date": "20180103",
        "end_date": None,
        "is_release": "0",
        "release_date": None,
        "pledgor": "机构",
        "holding_amount": 1000.0,
        "pledged_amount": 200.0,
        "p_total_ratio": None,
        "h_total_ratio": 20.0,
        "is_buyback": "0",
    }
    row.update(overrides)
    return row


def test_normalize_uses_reported_or_same_row_derived_ratio() -> None:
    frame, invalid = collector.normalize(
        [_row(), _row(ts_code="600002.SH", p_total_ratio=6.5)], 2018
    )

    assert invalid == 0
    actual = frame.sort("symbol").select(
        "symbol", "pledge_ratio", "pledge_ratio_source"
    )
    assert actual.to_dicts() == [
        {
            "symbol": "600001.SH",
            "pledge_ratio": 2.0,
            "pledge_ratio_source": "derived_same_row",
        },
        {
            "symbol": "600002.SH",
            "pledge_ratio": 6.5,
            "pledge_ratio_source": "reported",
        },
    ]


def test_normalize_retains_missing_ratio_for_coverage_audit() -> None:
    frame, invalid = collector.normalize(
        [_row(holding_amount=None, h_total_ratio=None, pledge_amount=None)], 2018
    )

    assert invalid == 0
    assert frame["pledge_ratio"].null_count() == 1
    assert frame["pledge_ratio_source"].to_list() == ["missing"]


def test_material_filter_excludes_release_rows() -> None:
    frame, _ = collector.normalize(
        [
            _row(p_total_ratio=5.0),
            _row(ts_code="600002.SH", p_total_ratio=9.0, is_release="1"),
        ],
        2018,
    )
    material = frame.filter(
        ~pl.col("is_release").is_in(["1", "Y", "是"])
        & (pl.col("pledge_ratio") >= 5.0)
    )

    assert material["symbol"].to_list() == ["600001.SH"]
