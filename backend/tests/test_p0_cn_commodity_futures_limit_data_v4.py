from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_cn_commodity_futures_limit_data_v4.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_limits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_normalize_limits_keeps_point_in_time_execution_fields() -> None:
    rows = [
        {
            "trade_date": "20201231",
            "ts_code": "M2105.DCE",
            "name": "豆粕2105",
            "up_limit": "3300",
            "down_limit": "2900",
            "m_ratio": "8",
            "cont": "M",
            "exchange": "DCE",
        }
    ]

    result = study.normalize_limits(rows)

    assert result["date"][0] == date(2020, 12, 31)
    assert result["contract"][0] == "M2105.DCE"
    assert result["up_limit"][0] == 3300.0
    assert result["down_limit"][0] == 2900.0
    assert result["margin_rate_pct"][0] == 8.0
