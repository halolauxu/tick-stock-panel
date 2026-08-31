"""Causal NumPy implementations of the frozen public Alpha101 subset."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

ALPHA_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    25,
    33,
    34,
    41,
    52,
    53,
    54,
    57,
    101,
)


class Alpha101Context:
    def __init__(
        self,
        *,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        amount: np.ndarray,
    ) -> None:
        arrays = {
            "open": _matrix(open),
            "high": _matrix(high),
            "low": _matrix(low),
            "close": _matrix(close),
            "volume": _matrix(volume),
            "amount": _matrix(amount),
        }
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("Alpha101 inputs must have one common shape")
        for name, value in arrays.items():
            setattr(self, name, value)
        self.returns = safe_divide(self.close, delay(self.close, 1)) - np.float32(1.0)
        self.vwap = safe_divide(self.amount, self.volume * np.float32(100.0))
        self.adv20 = rolling_mean(self.volume, 20)

    @classmethod
    def from_arrays(cls, **values: Any) -> Alpha101Context:
        return cls(**values)


def _matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("Alpha101 inputs must be two-dimensional")
    result = np.array(result, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def _clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    result[~np.isfinite(result)] = np.nan
    return result


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    result = np.full(shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def delay(values: np.ndarray, periods: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if periods < 0:
        raise ValueError("delay periods must be non-negative")
    result = np.full(source.shape, np.nan, dtype=np.float32)
    if periods == 0:
        result[:] = source
    elif periods < source.shape[0]:
        result[periods:] = source[:-periods]
    return result


def delta(values: np.ndarray, periods: int) -> np.ndarray:
    return _clean(np.asarray(values, dtype=np.float32) - delay(values, periods))


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result = np.full(source.shape, np.nan, dtype=np.float32)
    if source.shape[0] < window:
        return result
    finite = np.isfinite(source)
    filled = np.where(finite, source, np.float32(0.0)).astype(np.float64)
    cumulative = np.cumsum(filled, axis=0, dtype=np.float64)
    counts = np.cumsum(finite.astype(np.int32), axis=0, dtype=np.int32)
    sums = cumulative[window - 1 :].copy()
    valid_counts = counts[window - 1 :].copy()
    if window < source.shape[0]:
        sums[1:] -= cumulative[:-window]
        valid_counts[1:] -= counts[:-window]
    output = sums.astype(np.float32)
    output[valid_counts != window] = np.nan
    result[window - 1 :] = output
    return result


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_sum(values, window) / np.float32(window)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    mean = rolling_mean(source, window)
    mean_square = rolling_mean(source * source, window)
    variance = mean_square - mean * mean
    variance[np.isfinite(variance) & (variance < 0)] = 0.0
    return _clean(np.sqrt(variance))


@njit(cache=False, nogil=True, parallel=True)
def _rolling_extreme_kernel(
    source: np.ndarray, window: int, maximum: bool
) -> np.ndarray:
    result = np.full(source.shape, np.nan, dtype=np.float32)
    for asset_id in prange(source.shape[1]):
        for row in range(window - 1, source.shape[0]):
            value = source[row - window + 1, asset_id]
            if not np.isfinite(value):
                continue
            valid = True
            for offset in range(1, window):
                candidate = source[row - window + 1 + offset, asset_id]
                if not np.isfinite(candidate):
                    valid = False
                    break
                if (maximum and candidate > value) or (
                    not maximum and candidate < value
                ):
                    value = candidate
            if valid:
                result[row, asset_id] = value
    return result


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    return _rolling_extreme_kernel(_matrix(values), int(window), False)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    return _rolling_extreme_kernel(_matrix(values), int(window), True)


def rolling_cov(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    left_mean = rolling_mean(left, window)
    right_mean = rolling_mean(right, window)
    product_mean = rolling_mean(left * right, window)
    return _clean(product_mean - left_mean * right_mean)


def rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    covariance = rolling_cov(left, right, window)
    left_std = rolling_std(left, window)
    right_std = rolling_std(right, window)
    return safe_divide(covariance, left_std * right_std)


def cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.full(source.shape, np.nan, dtype=np.float32)
    for row in range(source.shape[0]):
        valid_ids = np.flatnonzero(np.isfinite(source[row]))
        count = valid_ids.size
        if count == 0:
            continue
        if count == 1:
            result[row, valid_ids[0]] = np.float32(0.5)
            continue
        row_values = source[row, valid_ids]
        order = np.argsort(row_values, kind="stable")
        ordered = row_values[order]
        boundaries = np.concatenate(
            (
                np.array([0]),
                np.flatnonzero(ordered[1:] != ordered[:-1]) + 1,
                np.array([count]),
            )
        )
        repetitions = np.diff(boundaries)
        average_positions = (
            boundaries[:-1] + 1 + boundaries[1:]
        ) / 2.0
        ranks = np.empty(count, dtype=np.float32)
        ranks[order] = np.repeat(average_positions, repetitions)
        result[row, valid_ids] = (ranks - 1.0) / np.float32(count - 1)
    return result


@njit(cache=False, nogil=True, parallel=True)
def _ts_rank_kernel(source: np.ndarray, window: int) -> np.ndarray:
    result = np.full(source.shape, np.nan, dtype=np.float32)
    denominator = float(window - 1)
    for asset_id in prange(source.shape[1]):
        for row in range(window - 1, source.shape[0]):
            current = source[row, asset_id]
            if not np.isfinite(current):
                continue
            less = 0
            equal = 0
            valid = True
            for offset in range(window):
                candidate = source[row - window + 1 + offset, asset_id]
                if not np.isfinite(candidate):
                    valid = False
                    break
                if candidate < current:
                    less += 1
                elif candidate == current:
                    equal += 1
            if valid:
                if window == 1:
                    result[row, asset_id] = 0.5
                else:
                    result[row, asset_id] = (
                        less + 0.5 * (equal - 1)
                    ) / denominator
    return result


def ts_rank(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    return _ts_rank_kernel(_matrix(values), int(window))


@njit(cache=False, nogil=True, parallel=True)
def _ts_argmax_kernel(source: np.ndarray, window: int) -> np.ndarray:
    result = np.full(source.shape, np.nan, dtype=np.float32)
    for asset_id in prange(source.shape[1]):
        for row in range(window - 1, source.shape[0]):
            maximum = -np.inf
            position = -1
            for offset in range(window):
                candidate = source[row - window + 1 + offset, asset_id]
                if not np.isfinite(candidate):
                    position = -1
                    break
                if candidate >= maximum:
                    maximum = candidate
                    position = offset + 1
            if position > 0:
                result[row, asset_id] = position
    return result


def ts_argmax(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    return _ts_argmax_kernel(_matrix(values), int(window))


def sign(values: np.ndarray) -> np.ndarray:
    result = np.sign(values).astype(np.float32)
    result[~np.isfinite(values)] = np.nan
    return result


def decay_linear_2(values: np.ndarray) -> np.ndarray:
    return _clean(
        (delay(values, 1) + np.float32(2.0) * values) / np.float32(3.0)
    )


def _finish(values: np.ndarray) -> np.ndarray:
    return _clean(values).astype(np.float32, copy=False)


def compute_alpha101(context: Alpha101Context, alpha_id: int) -> np.ndarray:
    if alpha_id not in ALPHA_IDS:
        raise ValueError(f"unsupported frozen Alpha101 formula: {alpha_id}")
    o = context.open
    h = context.high
    low = context.low
    c = context.close
    v = context.volume
    ret = context.returns
    vwap = context.vwap
    adv20 = context.adv20

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if alpha_id == 1:
            source = np.where(ret < 0, rolling_std(ret, 20), c)
            value = sign(source) * np.abs(source) ** np.float32(2.0)
            return _finish(cross_sectional_rank(ts_argmax(value, 5)) - 0.5)
        if alpha_id == 2:
            log_volume = np.full(v.shape, np.nan, dtype=np.float32)
            np.log(v, out=log_volume, where=np.isfinite(v) & (v > 0))
            return _finish(
                -rolling_corr(
                    cross_sectional_rank(delta(log_volume, 2)),
                    cross_sectional_rank(safe_divide(c - o, o)),
                    6,
                )
            )
        if alpha_id == 3:
            return _finish(
                -rolling_corr(
                    cross_sectional_rank(o),
                    cross_sectional_rank(v),
                    10,
                )
            )
        if alpha_id == 4:
            return _finish(-ts_rank(cross_sectional_rank(low), 9))
        if alpha_id == 5:
            return _finish(
                cross_sectional_rank(o - rolling_mean(vwap, 10))
                * -np.abs(cross_sectional_rank(c - vwap))
            )
        if alpha_id == 6:
            return _finish(-rolling_corr(o, v, 10))
        if alpha_id == 7:
            close_delta = delta(c, 7)
            value = -ts_rank(np.abs(close_delta), 60) * sign(close_delta)
            result = np.where(adv20 < v, value, np.float32(-1.0))
            result[~np.isfinite(adv20) | ~np.isfinite(v)] = np.nan
            return _finish(result)
        if alpha_id == 8:
            product = rolling_sum(o, 5) * rolling_sum(ret, 5)
            return _finish(-cross_sectional_rank(product - delay(product, 10)))
        if alpha_id == 9:
            close_delta = delta(c, 1)
            minimum = rolling_min(close_delta, 5)
            maximum = rolling_max(close_delta, 5)
            result = np.where(
                minimum > 0,
                close_delta,
                np.where(maximum < 0, close_delta, -close_delta),
            )
            result[~np.isfinite(minimum) | ~np.isfinite(maximum)] = np.nan
            return _finish(result)
        if alpha_id == 10:
            close_delta = delta(c, 1)
            minimum = rolling_min(close_delta, 4)
            maximum = rolling_max(close_delta, 4)
            result = np.where(
                minimum > 0,
                close_delta,
                np.where(maximum < 0, close_delta, -close_delta),
            )
            result[~np.isfinite(minimum) | ~np.isfinite(maximum)] = np.nan
            return _finish(cross_sectional_rank(result))
        if alpha_id == 11:
            spread = vwap - c
            return _finish(
                (
                    cross_sectional_rank(rolling_max(spread, 3))
                    + cross_sectional_rank(rolling_min(spread, 3))
                )
                * cross_sectional_rank(delta(v, 3))
            )
        if alpha_id == 12:
            return _finish(sign(delta(v, 1)) * -delta(c, 1))
        if alpha_id == 13:
            return _finish(
                -cross_sectional_rank(
                    rolling_cov(
                        cross_sectional_rank(c),
                        cross_sectional_rank(v),
                        5,
                    )
                )
            )
        if alpha_id == 14:
            return _finish(
                -cross_sectional_rank(delta(ret, 3)) * rolling_corr(o, v, 10)
            )
        if alpha_id == 15:
            correlation = rolling_corr(
                cross_sectional_rank(h), cross_sectional_rank(v), 3
            )
            return _finish(-rolling_sum(cross_sectional_rank(correlation), 3))
        if alpha_id == 16:
            return _finish(
                -cross_sectional_rank(
                    rolling_cov(
                        cross_sectional_rank(h),
                        cross_sectional_rank(v),
                        5,
                    )
                )
            )
        if alpha_id == 17:
            return _finish(
                -cross_sectional_rank(ts_rank(c, 10))
                * cross_sectional_rank(delta(delta(c, 1), 1))
                * cross_sectional_rank(ts_rank(safe_divide(v, adv20), 5))
            )
        if alpha_id == 18:
            return _finish(
                -cross_sectional_rank(
                    rolling_std(np.abs(c - o), 5)
                    + (c - o)
                    + rolling_corr(c, o, 10)
                )
            )
        if alpha_id == 19:
            return _finish(
                -sign((c - delay(c, 7)) + delta(c, 7))
                * (
                    np.float32(1.0)
                    + cross_sectional_rank(
                        np.float32(1.0) + rolling_sum(ret, 250)
                    )
                )
            )
        if alpha_id == 20:
            return _finish(
                -cross_sectional_rank(o - delay(h, 1))
                * cross_sectional_rank(o - delay(c, 1))
                * cross_sectional_rank(o - delay(low, 1))
            )
        if alpha_id == 22:
            return _finish(
                -delta(rolling_corr(h, v, 5), 5)
                * cross_sectional_rank(rolling_std(c, 20))
            )
        if alpha_id == 23:
            high_mean = rolling_mean(h, 20)
            result = np.where(high_mean < h, -delta(h, 2), np.float32(0.0))
            result[~np.isfinite(high_mean)] = np.nan
            return _finish(result)
        if alpha_id == 25:
            return _finish(
                cross_sectional_rank(-ret * adv20 * vwap * (h - c))
            )
        if alpha_id == 33:
            return _finish(cross_sectional_rank(-(np.float32(1.0) - safe_divide(o, c))))
        if alpha_id == 34:
            return _finish(
                cross_sectional_rank(
                    np.float32(1.0)
                    - cross_sectional_rank(
                        safe_divide(rolling_std(ret, 2), rolling_std(ret, 5))
                    )
                    + np.float32(1.0)
                    - cross_sectional_rank(delta(c, 1))
                )
            )
        if alpha_id == 41:
            return _finish(np.sqrt(h * low) - vwap)
        if alpha_id == 52:
            low_5 = rolling_min(low, 5)
            return _finish(
                (-low_5 + delay(low_5, 5))
                * cross_sectional_rank(
                    (rolling_sum(ret, 240) - rolling_sum(ret, 20))
                    / np.float32(220.0)
                )
                * ts_rank(v, 5)
            )
        if alpha_id == 53:
            position = safe_divide(
                (c - low) - (h - c),
                c - low,
            )
            return _finish(-delta(position, 9))
        if alpha_id == 54:
            return _finish(
                safe_divide(
                    -(low - c) * np.power(o, 5),
                    (low - h) * np.power(c, 5),
                )
            )
        if alpha_id == 57:
            return _finish(
                -safe_divide(
                    c - vwap,
                    decay_linear_2(cross_sectional_rank(ts_argmax(c, 30))),
                )
            )
        return _finish(safe_divide(c - o, (h - low) + np.float32(0.001)))
