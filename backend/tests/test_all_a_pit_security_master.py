from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_all_a_pit_security_master.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("all_a_master", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


master = _load_module()


def test_normalize_universe_keeps_all_sh_sz_boards_and_excludes_beijing() -> None:
    rows = [
        {
            "ts_code": symbol,
            "name": symbol,
            "market": market,
            "exchange": exchange,
            "list_status": "L",
            "list_date": "20200101",
            "delist_date": None,
        }
        for symbol, market, exchange in (
            ("000001.SZ", "主板", "SZSE"),
            ("300001.SZ", "创业板", "SZSE"),
            ("600000.SH", "主板", "SSE"),
            ("688001.SH", "科创板", "SSE"),
            ("920001.BJ", "北交所", "BSE"),
        )
    ]

    universe = master.normalize_universe(rows)

    assert universe.get_column("symbol").to_list() == [
        "000001.SZ",
        "300001.SZ",
        "600000.SH",
        "688001.SH",
    ]


def test_normalize_names_adds_only_missing_symbol_fallback() -> None:
    universe = master.normalize_universe(
        [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "market": "主板",
                "exchange": "SZSE",
                "list_status": "L",
                "list_date": "19910403",
                "delist_date": None,
            },
            {
                "ts_code": "300001.SZ",
                "name": "特锐德",
                "market": "创业板",
                "exchange": "SZSE",
                "list_status": "L",
                "list_date": "20091030",
                "delist_date": None,
            },
        ]
    )
    rows = [
        {
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "start_date": "20120802",
            "end_date": None,
            "ann_date": "20120801",
            "change_reason": "其他",
        }
    ]

    names = master.normalize_names(rows, universe)

    fallback = names.filter(names["change_reason"] == "stock_basic_fallback")
    assert fallback.height == 1
    assert fallback["symbol"].item() == "300001.SZ"
    assert fallback["start_date"].item() == date(2009, 10, 30)
    audit = master.validate_master(universe, names)
    assert audit["name_symbols"] == 2
    assert audit["prefix_counts"] == [
        {"prefix": "00", "len": 1},
        {"prefix": "30", "len": 1},
    ]
