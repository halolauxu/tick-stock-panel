"""Failure boundary that keeps the optional Alpha subsystem out of core startup."""
# Requirements: AM-S1-005, AM-S1-008, AM-S1-009.
from __future__ import annotations

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)


def load_alpha_api() -> ModuleType | None:
    try:
        return importlib.import_module("app.api.alpha_mining")
    except Exception as exc:  # the optional subsystem must not take down core routes
        logger.error("Alpha mining API unavailable; core application remains active: %s", exc)
        return None
