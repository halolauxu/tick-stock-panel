from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_top_inst_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_top_inst", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _row(side: str = "0") -> dict:
    return {
        "trade_date": "20260130",
        "ts_code": "000001.SZ",
        "exalter": "机构专用",
        "side": side,
        "buy": "1000000",
        "buy_rate": "10.0",
        "sell": "200000",
        "sell_rate": "2.0",
        "net_buy": "800000",
        "reason": "涨幅偏离值达7%的证券",
    }


def test_trading_dates_use_only_requested_quarter(tmp_path: Path) -> None:
    root = tmp_path / "kline_daily_enriched"
    for value in ("2026-01-05", "2026-03-31", "2026-04-01", "2025-01-02"):
        (root / f"date={value}").mkdir(parents=True)

    result = collector.trading_dates(tmp_path, 2026, 1)

    assert result == [date(2026, 1, 5), date(2026, 3, 31)]


def test_normalize_converts_numbers_and_dates() -> None:
    frame = collector.normalize([_row()])

    assert frame["trade_date"][0] == date(2026, 1, 30)
    assert frame["symbol"][0] == "000001.SZ"
    assert frame["net_buy"][0] == 800000.0


def test_normalize_preserves_distinct_rank_sides() -> None:
    frame = collector.normalize([_row("0"), _row("1")])

    assert frame.height == 2
