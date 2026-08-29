"""Engine registry with startup freeze and per-engine failure isolation."""
# Requirements: AM-S2-006 through AM-S2-016.
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaDiscoveryEngine,
    CandidateSpec,
    TrainOnlyContext,
    TrialBudget,
)


class AlphaEngineRegistryError(RuntimeError):
    pass


_FORBIDDEN_ENGINE_IMPORTS = (
    "app.plugins",
    "app.data_providers",
    "requests",
    "httpx",
    "tushare",
    "akshare",
    "tickflow",
)


class AlphaEngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, AlphaDiscoveryEngine] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register(self, engine: AlphaDiscoveryEngine) -> None:
        if self._frozen:
            raise AlphaEngineRegistryError("Alpha engine registry is frozen")
        if not isinstance(engine, AlphaDiscoveryEngine):
            raise AlphaEngineRegistryError("engine does not implement AlphaDiscoveryEngine")
        manifest = engine.manifest
        if manifest.api_version != ENGINE_API_VERSION:
            raise AlphaEngineRegistryError(
                f"engine {manifest.engine_id} uses API version {manifest.api_version}; "
                f"expected {ENGINE_API_VERSION}"
            )
        if manifest.engine_id in self._engines:
            raise AlphaEngineRegistryError(f"duplicate engine id: {manifest.engine_id}")
        self._engines[manifest.engine_id] = engine

    def freeze(self) -> None:
        self._frozen = True

    def get(self, engine_id: str) -> AlphaDiscoveryEngine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise AlphaEngineRegistryError(f"unknown Alpha engine: {engine_id}") from exc

    def list(self) -> tuple[AlphaDiscoveryEngine, ...]:
        return tuple(self._engines[key] for key in sorted(self._engines))

    def discover(
        self,
        engine_ids: Iterable[str],
        context: TrainOnlyContext,
        budget: TrialBudget,
    ) -> tuple[list[CandidateSpec], list[dict[str, str]]]:
        candidates: list[CandidateSpec] = []
        failures: list[dict[str, str]] = []
        for engine_id in engine_ids:
            try:
                engine = self.get(str(engine_id))
                discovered = engine.discover(context, budget)
                for candidate in discovered[: budget.max_candidates]:
                    if candidate.engine_id != engine.manifest.engine_id:
                        raise AlphaEngineRegistryError("candidate engine id does not match owner")
                    if candidate.engine_version != engine.manifest.version:
                        raise AlphaEngineRegistryError("candidate engine version does not match owner")
                    candidates.append(candidate)
            except Exception as exc:  # one research path must not abort the run
                failures.append({
                    "engine_id": str(engine_id),
                    "stage": "discover",
                    "error": str(exc)[:500],
                })
        return candidates, failures


def load_builtin_registry(
    package_name: str = "app.alpha_mining.engines",
) -> tuple[AlphaEngineRegistry, list[dict[str, str]]]:
    """Auto-discover modules exporting ``ENGINE``; no orchestrator edit is needed."""
    registry = AlphaEngineRegistry()
    failures: list[dict[str, str]] = []
    package = importlib.import_module(package_name)
    module_names = sorted(
        item.name for item in pkgutil.iter_modules(package.__path__) if not item.name.startswith("_")
    )
    for module_name in module_names:
        qualified = f"{package_name}.{module_name}"
        try:
            spec = importlib.util.find_spec(qualified)
            if spec is None or spec.origin is None:
                raise AlphaEngineRegistryError("engine source is unavailable for boundary audit")
            _audit_engine_source(Path(spec.origin).read_text(encoding="utf-8"))
            module = importlib.import_module(qualified)
            engine: Any = module.ENGINE
            _audit_engine_module(module)
            registry.register(engine)
        except Exception as exc:
            failures.append({
                "engine_id": module_name,
                "stage": "load",
                "error": str(exc)[:500],
            })
    registry.freeze()
    return registry, failures


def _audit_engine_module(module: Any) -> None:
    """Enforce the provider boundary for auto-loaded discovery modules."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise AlphaEngineRegistryError("engine source is unavailable for boundary audit") from exc
    _audit_engine_source(source)


def _audit_engine_source(source: str) -> None:
    """Reject supplier/network imports before executing an engine module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AlphaEngineRegistryError("engine source cannot be parsed for boundary audit") from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    violations = sorted(
        token
        for token in _FORBIDDEN_ENGINE_IMPORTS
        if any(name == token or name.startswith(f"{token}.") for name in imported)
    )
    if violations:
        raise AlphaEngineRegistryError(
            f"engine bypasses ResearchProvider boundary: {violations}"
        )
