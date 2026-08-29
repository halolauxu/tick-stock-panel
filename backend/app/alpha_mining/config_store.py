"""Independent Alpha feature and automation configuration."""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_DEFAULT = {
    "enabled": False,
    "auto_run_enabled": False,
    "auto_run_profile": "strict",
    "shadow_min_trading_days": 60,
    "shadow_min_fills": 200,
    "shadow_min_factor_round_trips": 30,
    "shadow_min_rank_ic": 0.02,
}


class AlphaConfigStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.path = Path(data_dir).resolve() / "alpha_mining" / "config.json"

    def get(self) -> dict[str, Any]:
        with _LOCK:
            if not self.path.is_file():
                return dict(_DEFAULT)
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Alpha配置损坏") from exc
            if not isinstance(value, dict):
                raise ValueError("Alpha配置必须是对象")
            return _validate({**_DEFAULT, **value})

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(patch) - set(_DEFAULT))
        if unknown:
            raise ValueError(f"不支持的Alpha配置: {unknown}")
        with _LOCK:
            value = _validate({**self.get(), **patch})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return value


def _validate(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value.get("enabled"), bool):
        raise ValueError("enabled必须是布尔值")
    if not isinstance(value.get("auto_run_enabled"), bool):
        raise ValueError("auto_run_enabled必须是布尔值")
    if value.get("auto_run_profile") not in {"balanced", "strict"}:
        raise ValueError("自动研究只允许balanced或strict")
    for field, minimum, maximum in (
        ("shadow_min_trading_days", 20, 500),
        ("shadow_min_fills", 20, 5000),
        ("shadow_min_factor_round_trips", 10, 2500),
    ):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or not minimum <= raw <= maximum:
            raise ValueError(f"{field}必须在{minimum}到{maximum}之间")
    rank_ic = value.get("shadow_min_rank_ic")
    if isinstance(rank_ic, bool) or not isinstance(rank_ic, (int, float)) or not -1 <= rank_ic <= 1:
        raise ValueError("shadow_min_rank_ic必须在-1到1之间")
    value["shadow_min_rank_ic"] = float(rank_ic)
    return value
