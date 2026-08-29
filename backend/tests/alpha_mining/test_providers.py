from __future__ import annotations

# Requirements: AM-S2-011 through AM-S2-014 and AM-S3-004 through AM-S3-008.
from datetime import date, datetime

import polars as pl
import pytest

from app.alpha_mining.contracts import (
    EventProviderManifest,
    FeatureProviderManifest,
    PointInTimeDataRequest,
)
from app.alpha_mining.providers import (
    ProviderContractError,
    StandardFeatureProvider,
    TimestampedEventProvider,
)


def test_research_feature_provider_rejects_duplicate_keys() -> None:
    provider = StandardFeatureProvider(
        FeatureProviderManifest("test", "daily", "1", True, "date"),
        lambda _request: pl.DataFrame({
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "factor": [1.0, 2.0],
        }),
    )
    request = PointInTimeDataRequest("stock", "2026-01-05", "2026-01-05", "after_close", ("factor",))
    with pytest.raises(ProviderContractError, match="重复"):
        provider.load(request)


def test_weekend_event_maps_to_next_trading_decision_date() -> None:
    provider = TimestampedEventProvider(
        EventProviderManifest("events", "event_history", "1", "published_at", "effective_date"),
        lambda _request: pl.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ"],
            "published_at": [datetime(2026, 1, 4, 10), datetime(2026, 1, 5, 16)],
        }),
        lambda _request: [date(2026, 1, 5), date(2026, 1, 6)],
    )
    request = PointInTimeDataRequest("stock", "2026-01-04", "2026-01-06", "after_close", ())
    result = provider.load_events(request)
    assert result["effective_date"].to_list() == [date(2026, 1, 5), date(2026, 1, 6)]
