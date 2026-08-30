from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_flow_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_flow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_attach_flow_features_is_trailing_and_uses_daily_amount() -> None:
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(3)]
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ"] * 3,
            "date": days,
            "amount": [100.0, 200.0, 300.0],
        }
    )
    flow = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "trade_date": [days[0], days[2]],
            "buy_lg_cny": [20.0, 90.0],
            "buy_elg_cny": [0.0, 0.0],
            "sell_lg_cny": [10.0, 30.0],
            "sell_elg_cny": [0.0, 0.0],
        }
    )

    result = study.attach_flow_features(panel, flow)

    assert result["flow_observations_20d"].to_list() == [1, 1, 2]
    assert result["large_net_flow_20d_cny"].to_list() == [10.0, 10.0, 70.0]
    assert result["amount_20d_cny"].to_list() == [100.0, 300.0, 600.0]
    assert result["large_net_flow_ratio_20d"][-1] == pytest.approx(70 / 600)


def test_flow_candidates_select_only_positive_top_ten_in_bottom_decile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_date = date(2020, 1, 3)
    entry_date = date(2020, 1, 6)
    rows = []
    for rank in range(1, 21):
        ratio = 0.30 - rank * 0.01 if rank <= 10 else -0.10
        rows.append(
            {
                "symbol": f"{rank:06d}.SZ",
                "date": signal_date,
                "market_cap": float(rank),
                "market_cap_rank": rank,
                "amount": 100_000_000.0,
                "daily_return": 0.01,
                "flow_observations_20d": 20,
                "large_net_flow_20d_cny": ratio * 1_000_000_000.0,
                "amount_20d_cny": 1_000_000_000.0,
                "large_net_flow_ratio_20d": ratio,
            }
        )
    panel = pl.DataFrame(rows)
    monkeypatch.setattr(
        study,
        "_weekly_signal_rows",
        lambda _: panel.with_columns(pl.lit(entry_date).alias("entry_date")),
    )

    selected = study.build_candidates(panel, use_flow=True)

    assert selected.height == 10
    assert selected["cap_rank"].to_list() == list(range(1, 11))
    assert selected["large_net_flow_ratio_20d"].to_list() == sorted(
        selected["large_net_flow_ratio_20d"].to_list(), reverse=True
    )


def test_gate_requires_absolute_return_control_improvement_and_integrity() -> None:
    flow = {
        "metrics": {
            "account_annualized": 0.61,
            "account_max_drawdown": -0.30,
            "positive_account_years": 6,
            "mean_cash_ratio": 0.10,
        },
        "execution": {
            "buy": {"execution_rate": 0.90},
            "sell": {"execution_rate": 0.91},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }
    control = {"metrics": {"account_annualized": 0.45}}

    passed = study.evaluate_gate(flow, control)
    assert passed["passed"] is True
    assert passed["annualized_improvement_vs_microcap_control"] == pytest.approx(
        0.16
    )

    flow["integrity"]["ending_unresolved_positions"] = 1
    failed = study.evaluate_gate(flow, control)
    assert failed["passed"] is False
    assert failed["checks"]["ending_positions_resolved"] is False
