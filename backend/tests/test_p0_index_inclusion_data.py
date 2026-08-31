from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_index_inclusion_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_index_inclusion_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_month_ranges_cover_frozen_months_once() -> None:
    result = study.month_ranges(date(2020, 1, 15), date(2020, 3, 2))

    assert result == [
        (date(2020, 1, 1), date(2020, 1, 31)),
        (date(2020, 2, 1), date(2020, 2, 29)),
        (date(2020, 3, 1), date(2020, 3, 31)),
    ]


def test_normalize_weights_filters_bad_rows_and_deduplicates() -> None:
    rows = [
        {
            "index_code": "000300.SH",
            "con_code": "600001.SH",
            "trade_date": "20200630",
            "weight": 0.5,
        },
        {
            "index_code": "000300.SH",
            "con_code": "600001.SH",
            "trade_date": "20200630",
            "weight": 0.6,
        },
        {
            "index_code": "000300.SH",
            "con_code": "600002.SH",
            "trade_date": "20200630",
            "weight": 0.0,
        },
    ]

    result = study.normalize_weights(rows)

    assert result.height == 1
    assert result["weight_pct"][0] == 0.6


def test_derive_regular_additions_only_compares_may_june_and_november_december() -> None:
    rows = []
    for index_code in study.INDEX_CODES:
        rows.extend(
            [
                {
                    "index_code": index_code,
                    "con_code": "600001.SH",
                    "trade_date": "20200529",
                    "weight": 1.0,
                },
                {
                    "index_code": index_code,
                    "con_code": "600001.SH",
                    "trade_date": "20200630",
                    "weight": 0.8,
                },
                {
                    "index_code": index_code,
                    "con_code": "600002.SH",
                    "trade_date": "20200630",
                    "weight": 0.2,
                },
            ]
        )
    weights = study.normalize_weights(rows)

    result = study.derive_regular_additions(weights)

    assert result.height == 2
    assert set(result["symbol"].to_list()) == {"600002.SH"}
    assert set(result["cycle_month"].to_list()) == {date(2020, 6, 1)}
