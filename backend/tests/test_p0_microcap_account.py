from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_account.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_account", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


account = _load_module()


def _candidates(rows: list[tuple[date, date, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": signal_date,
                "entry_date": entry_date,
                "symbol": symbol,
                "cap_rank": rank,
                "signal_amount": 10_000_000.0,
            }
            for signal_date, entry_date, symbol, rank in rows
        ]
    )


def _quote(
    entry_date: date,
    symbol: str,
    *,
    raw_open: float = 10.0,
    adjusted_open: float | None = None,
    close: float | None = None,
    limit_up: float = 11.0,
    limit_down: float = 9.0,
    exact: bool = True,
    excluded_name: bool = False,
) -> dict:
    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "quote_date": entry_date,
        "open": adjusted_open or raw_open,
        "raw_open": raw_open,
        "close": close or adjusted_open or raw_open,
        "raw_close": close or raw_open,
        "entry_volume": 1_000_000.0,
        "entry_amount": 10_000_000.0,
        "limit_up_price": limit_up,
        "limit_down_price": limit_down,
        "exact_quote": exact,
        "is_excluded_name": excluded_name,
    }


def test_affordable_shares_respects_lot_and_minimum_commission() -> None:
    shares = account.affordable_shares(10.0, 15_000.0, 15_000.0)

    assert shares == 1400
    gross = shares * 10.0
    assert gross * 1.0005 + account.commission(gross) <= 15_000.0
    assert account.commission(gross) == 5.0


def test_affordable_shares_supports_convertible_bond_lot() -> None:
    shares = account.affordable_shares(
        100.0, 10_500.0, 10_500.0, lot_size=10
    )

    assert shares == 100
    assert shares % 10 == 0


def test_simulator_backfills_blocked_buy_then_sells_before_next_buy() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d0, d1, "B.SZ", 2),
            (d2, d3, "C.SZ", 1),
            (d2, d3, "B.SZ", 2),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ", raw_open=11.0, limit_up=11.0),
            _quote(d1, "B.SZ"),
            _quote(d1, "C.SZ"),
            _quote(d3, "A.SZ"),
            _quote(d3, "B.SZ", raw_open=10.5, close=10.5),
            _quote(d3, "C.SZ", raw_open=5.0, close=5.0),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
    )

    assert [row["symbol"] for row in result["trades"]] == [
        "B.SZ",
        "B.SZ",
        "C.SZ",
    ]
    assert result["orders"][0]["reason"] == "limit_up"
    assert result["trades"][1]["side"] == "SELL"
    assert result["trades"][2]["side"] == "BUY"
    assert result["ending_positions"][0]["symbol"] == "C.SZ"
    assert result["max_cash_reconciliation_error"] == pytest.approx(0.0)


def test_blocked_exit_keeps_position_and_prevents_replacement() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d2, d3, "B.SZ", 1),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d1, "B.SZ"),
            _quote(d3, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "B.SZ"),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
    )

    assert result["ending_positions"][0]["symbol"] == "A.SZ"
    rejected_sell = [
        row
        for row in result["orders"]
        if row["side"] == "SELL" and row["status"] == "REJECTED"
    ]
    assert rejected_sell[0]["reason"] == "limit_down"
    assert not any(
        row["side"] == "BUY" and row["symbol"] == "B.SZ"
        for row in result["orders"]
    )


def test_simulator_can_disable_stamp_tax_for_etf_account() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [(d0, d1, "A.SZ", 1), (d2, d3, "B.SZ", 1)]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d1, "B.SZ"),
            _quote(d3, "A.SZ"),
            _quote(d3, "B.SZ"),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
        stamp_tax_rate=0.0,
    )

    sell = next(row for row in result["trades"] if row["side"] == "SELL")
    assert sell["stamp_tax"] == 0.0


def test_forced_rebalance_can_close_and_reopen_same_symbol() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [(d0, d1, "A.SZ", 1), (d2, d3, "A.SZ", 1)]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(
                d3,
                "A.SZ",
                raw_open=11.0,
                close=11.0,
                limit_up=12.1,
            ),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
        force_rebalance_dates={d3},
        allow_same_day_reentry=True,
    )

    assert [(row["date"], row["side"]) for row in result["trades"]] == [
        (d1, "BUY"),
        (d3, "SELL"),
        (d3, "BUY"),
    ]
    assert result["ending_positions"][0]["start_date"] == d3
    assert result["max_cash_reconciliation_error"] == pytest.approx(0.0)


def test_cost_multiplier_doubles_modeled_friction() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d3 = date(2024, 1, 15)
    candidates = _candidates([(d0, d1, "A.SZ", 1)])
    execution = pl.DataFrame([_quote(d1, "A.SZ"), _quote(d3, "A.SZ")])

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
        action_dates=[d1, d3],
        cost_multiplier=2.0,
    )

    buy = next(row for row in result["trades"] if row["side"] == "BUY")
    sell = next(row for row in result["trades"] if row["side"] == "SELL")
    assert buy["commission"] == pytest.approx(account.commission(buy["gross"]) * 2)
    assert buy["slippage"] == pytest.approx(
        buy["gross"] * account.baseline.SLIPPAGE_PCT * 2
    )
    assert sell["stamp_tax"] == pytest.approx(
        sell["gross"] * account.baseline.STAMP_TAX_CURRENT * 2
    )


def test_max_holding_clock_exits_even_when_symbol_remains_desired() -> None:
    days = [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
    ]
    candidates = _candidates(
        [(day, day, "A.SZ", 1) for day in days]
    )
    execution = pl.DataFrame([_quote(day, "A.SZ") for day in days])

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=days,
        max_holding_sessions=3,
    )

    assert [(row["date"], row["side"]) for row in result["trades"]] == [
        (days[0], "BUY"),
        (days[2], "SELL"),
        (days[3], "BUY"),
    ]
    forced = next(row for row in result["orders"] if row["side"] == "SELL")
    assert forced["exit_trigger"] == "max_holding_sessions"
    assert result["ending_positions"][0]["start_date"] == days[3]


def test_cooldown_blocks_reentry_for_requested_trading_sessions() -> None:
    days = [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 12),
    ]
    candidates = _candidates(
        [(day, day, "A.SZ", 1) for day in days]
    )
    execution = pl.DataFrame([_quote(day, "A.SZ") for day in days])

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=days,
        max_holding_sessions=3,
        cooldown_sessions=2,
    )

    assert [(row["date"], row["side"]) for row in result["trades"]] == [
        (days[0], "BUY"),
        (days[2], "SELL"),
        (days[5], "BUY"),
    ]
    cooldown_skips = [
        row
        for row in result["orders"]
        if row.get("reason") == "cooldown"
    ]
    assert [row["date"] for row in cooldown_skips] == days[3:5]


def test_entry_gate_blocks_only_new_positions() -> None:
    days = [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
    ]
    candidates = _candidates(
        [(day, day, "A.SZ", 1) for day in days]
    )
    execution = pl.DataFrame([_quote(day, "A.SZ") for day in days])

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=days,
        max_holding_sessions=3,
        entry_gate_by_date={
            days[0]: True,
            days[1]: False,
            days[2]: False,
            days[3]: False,
            days[4]: True,
        },
    )

    assert [(row["date"], row["side"]) for row in result["trades"]] == [
        (days[0], "BUY"),
        (days[2], "SELL"),
        (days[4], "BUY"),
    ]
    blocked = [
        row
        for row in result["orders"]
        if row.get("reason") == "entry_gate"
    ]
    assert [row["date"] for row in blocked] == [days[3]]


def test_exposure_budget_caps_buys_without_forcing_desired_sales() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d0, d1, "B.SZ", 2),
            (d2, d3, "A.SZ", 1),
            (d2, d3, "C.SZ", 2),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d1, "B.SZ"),
            _quote(d1, "C.SZ"),
            _quote(d3, "A.SZ"),
            _quote(d3, "B.SZ"),
            _quote(d3, "C.SZ"),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=2,
        target_exposure_by_date={d1: 0.20, d3: 0.0},
    )

    first = result["snapshots"][0]
    assert first["target_exposure"] == pytest.approx(0.20)
    assert first["actual_exposure"] <= 0.20
    assert {row["symbol"] for row in result["ending_positions"]} == {
        "A.SZ"
    }
    assert not any(
        row["date"] == d3
        and row["side"] == "SELL"
        and row["symbol"] == "A.SZ"
        for row in result["orders"]
    )
    assert not any(
        row["date"] == d3
        and row["side"] == "BUY"
        and row["symbol"] == "C.SZ"
        for row in result["orders"]
    )
    assert result["snapshots"][1]["risk_budget_blocked_slots"] == 1


def test_residual_exposure_budget_is_skipped_before_order_submission() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d2, d3, "A.SZ", 1),
            (d2, d3, "B.SZ", 2),
            (d2, d3, "C.SZ", 3),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ", raw_open=5.0, close=15.0),
            _quote(d1, "B.SZ", raw_open=5.0),
            _quote(d1, "C.SZ", raw_open=5.0),
            _quote(d3, "A.SZ", raw_open=15.0, close=15.0),
            _quote(d3, "B.SZ", raw_open=5.0),
            _quote(d3, "C.SZ", raw_open=5.0),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=3,
        target_exposure_by_date={d1: 0.10, d3: 0.10},
    )

    skipped = [
        row
        for row in result["orders"]
        if row["status"] == "PRETRADE_SKIPPED"
        and row["reason"] == "portfolio_risk_budget"
    ]
    assert len(skipped) == 1
    summary = account.execution_summary(result["orders"])
    assert summary["buy"]["pretrade_skip_reasons"] == {
        "portfolio_risk_budget": 1
    }


def test_daily_equity_uses_stale_mark_without_future_backfill() -> None:
    d1, d2, d3 = date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)
    simulation = {
        "intervals": [
            {
                "position_id": 1,
                "symbol": "A.SZ",
                "units": 100.0,
                "start_date": d1,
                "end_date": None,
            }
        ],
        "snapshots": [{"date": d1, "cash": 1_000.0}],
        "ending_positions": [{"symbol": "A.SZ"}],
    }
    quotes = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "date": [d1, d3],
            "close": [10.0, 12.0],
        }
    )

    daily, integrity = account.build_daily_equity(
        simulation,
        quotes,
        [d1, d2, d3],
        initial_cash=2_000.0,
    )

    assert daily["equity"].to_list() == [2_000.0, 2_000.0, 2_200.0]
    assert daily["daily_return"].to_list() == pytest.approx([0.0, 0.0, 0.1])
    assert integrity["stale_position_days"] == 1
    assert integrity["longest_stale_trading_days"] == 1
    assert integrity["ending_unresolved_positions"] == 0


def test_main_board_st_uses_five_percent_limit_and_cannot_be_new_buy() -> None:
    days = [date(2024, 1, 8), date(2024, 1, 9)]
    source = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": days,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "raw_close": [10.0, 10.0],
            "volume": [1_000.0, 1_000.0],
            "amount": [1_000_000.0, 1_000_000.0],
            "name": ["ST样本", "ST样本"],
        }
    )

    quotes = account.prepare_quote_panel(source)

    assert quotes["limit_up_price"][1] == pytest.approx(10.5)
    assert quotes["limit_down_price"][1] == pytest.approx(9.5)
    quote = _quote(days[1], "000001.SZ", excluded_name=True)
    assert account._buy_rejection(quote, 10_000.0) == (
        "became_risk_warning"
    )
    position = {"units": 1_000.0}
    assert account._sell_rejection(position, quote) is None


def test_execution_rate_excludes_signal_day_capacity_skips() -> None:
    summary = account.execution_summary(
        [
            {
                "side": "BUY",
                "status": "PRETRADE_SKIPPED",
                "reason": "signal_capacity",
            },
            {"side": "BUY", "status": "FILLED", "reason": None},
            {"side": "SELL", "status": "FILLED", "reason": None},
        ]
    )

    assert summary["buy"]["orders"] == 1
    assert summary["buy"]["filled"] == 1
    assert summary["buy"]["execution_rate"] == 1.0
    assert summary["buy"]["pretrade_skipped"] == 1


def test_gate_uses_independently_started_validation_and_stress_accounts() -> None:
    def result(period: str) -> dict:
        return {
            "metrics": {
                "period": period,
                "account_annualized": 0.20,
                "annualized_excess": 0.12,
                "positive_account_years": 2,
            },
            "execution": {
                "buy": {"execution_rate": 0.90},
                "sell": {"execution_rate": 0.90},
            },
            "integrity": {
                "ending_unresolved_positions": 0,
                "max_cash_reconciliation_error": 0.0,
            },
        }

    independent = {
        "validation": result("validation"),
        "known_stress": result("known_stress"),
    }

    assert account.evaluate_gate(independent)["verdict"] == "CONTINUE_TO_ESCAPE"
    independent["known_stress"]["metrics"]["account_annualized"] = 0.10
    decision = account.evaluate_gate(independent)
    assert decision["verdict"] == "DOWNGRADE"
    assert "known_stress_annualized" in decision["failures"]


def test_independent_account_respects_requested_initial_cash() -> None:
    signal_day = date(2020, 12, 31)
    entry_day = date(2021, 1, 4)
    candidates = _candidates(
        [(signal_day, entry_day, "A.SZ", 1)]
    )
    execution = pl.DataFrame([_quote(entry_day, "A.SZ")])
    quotes = pl.DataFrame(
        {"symbol": ["A.SZ"], "date": [entry_day], "close": [10.0]}
    )
    weekly_market = pl.DataFrame(
        {
            "date": [entry_day],
            "period": ["validation"],
            "market_net": [0.0],
        }
    )

    result = account.run_independent_account(
        "validation",
        candidates,
        execution,
        quotes,
        [entry_day],
        weekly_market,
        initial_cash=20_000.0,
    )

    assert result["daily_equity"][0]["equity"] == pytest.approx(20_000.0)
    assert result["account"]["ending_equity"] == pytest.approx(20_000.0)


def test_confirmed_delisting_writes_position_off_without_fake_sale() -> None:
    buy_day = date(2024, 5, 27)
    settlement_day = date(2024, 7, 1)
    candidates = _candidates([(date(2024, 5, 24), buy_day, "A.SZ", 1)])
    execution = pl.DataFrame([_quote(buy_day, "A.SZ", raw_open=1.0)])

    simulation = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=[buy_day, settlement_day],
        delist_dates={"A.SZ": date(2024, 6, 27)},
    )

    assert simulation["ending_positions"] == []
    assert len(simulation["settlements"]) == 1
    settlement = simulation["settlements"][0]
    assert settlement["date"] == settlement_day
    assert settlement["effective_delist_date"] == date(2024, 6, 27)
    assert settlement["status"] == "DELISTED_WRITE_OFF"
    assert settlement["recovery_value"] == 0.0
    assert settlement["recognized_loss"] < 0
    assert [row["side"] for row in simulation["orders"]] == ["BUY"]

    daily, integrity = account.build_daily_equity(
        simulation,
        pl.DataFrame(
            {"symbol": ["A.SZ"], "date": [buy_day], "close": [1.0]}
        ),
        [buy_day, settlement_day],
        initial_cash=20_000.0,
    )
    assert daily[-1, "position_count"] == 0
    assert daily[-1, "equity"] == pytest.approx(simulation["ending_cash"])
    assert integrity["ending_unresolved_positions"] == 0


def test_cb_settlement_waits_until_after_delist_date() -> None:
    buy_day = date(2020, 3, 16)
    delist_day = date(2020, 3, 17)
    settlement_day = date(2020, 3, 23)
    candidates = _candidates([(date(2020, 3, 13), buy_day, "A.SZ", 1)])
    execution = pl.DataFrame([_quote(buy_day, "A.SZ", raw_open=10.0)])

    simulation = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=[buy_day, delist_day, settlement_day],
        delist_dates={"A.SZ": delist_day},
        delist_settlement_status="CB_DELISTED_ZERO_RECOVERY",
        settle_only_after_delist_date=True,
    )

    rejected = [
        row for row in simulation["orders"] if row["side"] == "SELL"
    ]
    assert len(rejected) == 1
    assert rejected[0]["date"] == delist_day
    assert rejected[0]["reason"] == "missing_market_data"
    assert simulation["settlements"][0]["date"] == settlement_day
    assert simulation["settlements"][0]["effective_delist_date"] == delist_day
    assert simulation["settlements"][0]["status"] == (
        "CB_DELISTED_ZERO_RECOVERY"
    )


def test_cb_settlement_can_credit_conservative_face_value() -> None:
    buy_day = date(2020, 3, 16)
    delist_day = date(2020, 3, 17)
    settlement_day = date(2020, 3, 23)
    candidates = _candidates([(date(2020, 3, 13), buy_day, "A.SZ", 1)])
    execution = pl.DataFrame(
        [
            _quote(
                buy_day,
                "A.SZ",
                raw_open=110.0,
                limit_up=121.0,
                limit_down=99.0,
            )
        ]
    )

    simulation = account.simulate_account(
        candidates,
        execution,
        initial_cash=20_000.0,
        target_positions=1,
        action_dates=[buy_day, delist_day, settlement_day],
        lot_size=10,
        stamp_tax_rate=0.0,
        delist_dates={"A.SZ": delist_day},
        delist_settlement_status="CB_DELISTED_FACE_VALUE_RECOVERY",
        settle_only_after_delist_date=True,
        delist_recovery_per_raw_share=100.0,
    )

    settlement = simulation["settlements"][0]
    assert settlement["raw_shares"] == 180
    assert settlement["recovery_value"] == 18_000.0
    assert settlement["recognized_loss"] == pytest.approx(-1_800.0)
    assert simulation["ending_cash"] == pytest.approx(18_185.1)
    assert simulation["max_cash_reconciliation_error"] == pytest.approx(0.0)
