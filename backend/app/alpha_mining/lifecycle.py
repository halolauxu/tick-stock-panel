"""Shared qualification rules for the non-bypassable Alpha lifecycle."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.services.mining_preflight import enriched_partition_dates


def is_strict_full_history_request(
    data_dir: Path | str,
    request: Mapping[str, Any],
) -> bool:
    if request.get("budget_profile") != "strict":
        return False
    asset_type = str(request.get("asset_type") or "stock")
    dates = enriched_partition_dates(Path(data_dir), asset_type)
    if not dates:
        return False
    start = str(request.get("start") or "")
    end = str(request.get("end") or "")
    return start <= dates[0].isoformat() and end >= dates[-1].isoformat()
