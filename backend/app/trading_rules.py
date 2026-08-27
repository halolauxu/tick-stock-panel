"""Shared deterministic trading rules for backtests and event-driven paper accounts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TradingCostModel:
    """The existing project cost convention: percentage commission and slippage."""

    commission_pct: float = 0.0002
    stamp_tax_pct: float = 0.001
    slippage_bps: float = 5.0

    def buy_cost_pct(self) -> float:
        return float(self.commission_pct) + float(self.slippage_bps) / 10_000.0

    def sell_cost_pct(self) -> float:
        return (
            float(self.commission_pct)
            + float(self.stamp_tax_pct)
            + float(self.slippage_bps) / 10_000.0
        )

    def buy_cash_required(self, price: float, quantity: int) -> float:
        return float(price) * int(quantity) * (1 + self.buy_cost_pct())

    def sell_cash_received(self, price: float, quantity: int) -> float:
        return float(price) * int(quantity) * (1 - self.sell_cost_pct())


def round_lot_quantity(
    allocation: float,
    price: float,
    cost_model: TradingCostModel,
    *,
    lot_size: int = 100,
) -> int:
    """Return the largest whole-lot buy quantity that fits the cash allocation."""
    if not math.isfinite(allocation) or not math.isfinite(price):
        return 0
    if allocation <= 0 or price <= 0 or lot_size <= 0:
        return 0
    unit_cash = price * (1 + cost_model.buy_cost_pct())
    return int(math.floor(allocation / unit_cash / lot_size) * lot_size)


def is_same_day_t_plus_one_locked(acquired_on: date | str, trading_on: date | str) -> bool:
    """A-share shares acquired on a date are not sellable on that same date."""
    return str(acquired_on)[:10] == str(trading_on)[:10]


def is_same_price_bar(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> bool:
    """Return whether a daily bar is effectively a one-price bar."""
    prices = (open_price, high_price, low_price, close_price)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in prices):
        return False
    tolerance = max(abs(float(close_price)) * 1e-4, 0.01)
    return max(prices) - min(prices) <= tolerance


def is_one_price_locked(
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    limit_price: float,
) -> bool:
    """Match the backtest one-price-board tolerance using raw prices."""
    prices = (open_price, high_price, low_price, close_price, limit_price)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in prices):
        return False
    tolerance = max(abs(float(close_price)) * 1e-4, 0.01)
    return is_same_price_bar(*prices[:4]) and abs(float(close_price) - float(limit_price)) <= tolerance
