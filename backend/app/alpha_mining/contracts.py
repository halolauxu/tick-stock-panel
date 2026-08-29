"""Stable contracts between the Alpha orchestrator and discovery engines."""
# Requirements: AM-S2-001 through AM-S2-014.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl

from app.alpha_mining.taxonomy import (
    DISCOVERY_CLASSES,
    INFORMATION_DOMAINS,
    MECHANISM_CLASSES,
    PREDICTION_OBJECTS,
)

ENGINE_API_VERSION = "1.0"
EngineReadiness = Literal["ready", "blocked"]


@dataclass(frozen=True)
class DatasetRequirement:
    dataset_id: str
    minimum_coverage: float = 0.95
    pit_required: bool = True
    timestamp_field: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between zero and one")
        if self.pit_required and self.timestamp_field is not None and not self.timestamp_field:
            raise ValueError("timestamp_field must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "minimum_coverage": self.minimum_coverage,
            "pit_required": self.pit_required,
            "timestamp_field": self.timestamp_field,
        }


@dataclass(frozen=True)
class AlphaEngineManifest:
    engine_id: str
    version: str
    api_version: str
    name: str
    family: str
    information_domains: tuple[str, ...]
    mechanism_classes: tuple[str, ...]
    economic_mechanism: str
    discovery_classes: tuple[str, ...]
    discovery_method: str
    prediction_objects: tuple[str, ...]
    forecast_targets: tuple[str, ...]
    frequencies: tuple[str, ...] = ("1d",)
    decision_clocks: tuple[str, ...] = ("after_close",)
    required_datasets: tuple[DatasetRequirement, ...] = (
        DatasetRequirement("daily_enriched", 0.95, True, "date"),
    )
    forecast_horizons: tuple[int, ...] = (5,)
    output_candidate_types: tuple[str, ...] = ("factor_rank",)
    parameter_contract_version: str = "1.0"
    artifact_contract_version: str = "1.0"
    auto_run_allowed: bool = False
    required_features: tuple[str, ...] = ()
    optional_features: tuple[str, ...] = ()
    asset_types: tuple[str, ...] = ("stock",)
    readiness: EngineReadiness = "ready"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.engine_id or len(self.engine_id) > 80:
            raise ValueError("engine_id must contain 1 to 80 characters")
        if not self.version:
            raise ValueError("engine version is required")
        if not self.name:
            raise ValueError("engine name is required")
        if not self.information_domains:
            raise ValueError("at least one information domain is required")
        unknown_domains = sorted(set(self.information_domains) - INFORMATION_DOMAINS)
        if unknown_domains:
            raise ValueError(f"unknown information domains: {unknown_domains}")
        unknown_mechanisms = sorted(set(self.mechanism_classes) - MECHANISM_CLASSES)
        if not self.mechanism_classes or unknown_mechanisms:
            raise ValueError(f"invalid mechanism classes: {unknown_mechanisms}")
        unknown_methods = sorted(set(self.discovery_classes) - DISCOVERY_CLASSES)
        if not self.discovery_classes or unknown_methods:
            raise ValueError(f"invalid discovery classes: {unknown_methods}")
        unknown_objects = sorted(set(self.prediction_objects) - PREDICTION_OBJECTS)
        if not self.prediction_objects or unknown_objects:
            raise ValueError(f"invalid prediction objects: {unknown_objects}")
        if not self.forecast_targets:
            raise ValueError("at least one forecast target is required")
        if not self.frequencies or not self.decision_clocks:
            raise ValueError("frequency and decision clock contracts are required")
        if not self.required_datasets:
            raise ValueError("at least one dataset requirement is required")
        if not self.forecast_horizons or any(value <= 0 for value in self.forecast_horizons):
            raise ValueError("positive forecast horizons are required")
        if not self.output_candidate_types:
            raise ValueError("at least one output candidate type is required")
        if not self.parameter_contract_version or not self.artifact_contract_version:
            raise ValueError("parameter and artifact contract versions are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "version": self.version,
            "api_version": self.api_version,
            "name": self.name,
            "family": self.family,
            "information_domains": list(self.information_domains),
            "mechanism_classes": list(self.mechanism_classes),
            "economic_mechanism": self.economic_mechanism,
            "discovery_classes": list(self.discovery_classes),
            "discovery_method": self.discovery_method,
            "prediction_objects": list(self.prediction_objects),
            "forecast_targets": list(self.forecast_targets),
            "frequencies": list(self.frequencies),
            "decision_clocks": list(self.decision_clocks),
            "required_datasets": [item.to_dict() for item in self.required_datasets],
            "forecast_horizons": list(self.forecast_horizons),
            "output_candidate_types": list(self.output_candidate_types),
            "parameter_contract_version": self.parameter_contract_version,
            "artifact_contract_version": self.artifact_contract_version,
            "auto_run_allowed": self.auto_run_allowed,
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "asset_types": list(self.asset_types),
            "readiness": self.readiness,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class DataCatalogContext:
    asset_type: str
    start: str
    end: str
    available_features: tuple[str, ...]
    datasets: Mapping[str, DataQualification] = field(default_factory=dict)
    decision_clock: str = "after_close"
    observations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataQualification:
    ready: bool
    reasons: tuple[str, ...]
    observations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PointInTimeDataRequest:
    asset_type: str
    start: str
    end: str
    decision_clock: str
    columns: tuple[str, ...]
    as_of: str | None = None


@dataclass(frozen=True)
class FeatureProviderManifest:
    provider_id: str
    dataset_id: str
    version: str
    pit_capable: bool
    timestamp_field: str
    supported_frequencies: tuple[str, ...] = ("1d",)


@dataclass(frozen=True)
class EventProviderManifest:
    provider_id: str
    dataset_id: str
    version: str
    published_at_field: str
    effective_date_field: str


@dataclass(frozen=True)
class RendererManifest:
    renderer_id: str
    version: str
    candidate_types: tuple[str, ...]
    schema_version: str


@dataclass(frozen=True, slots=True)
class TrainOnlyContext:
    """The only research frame an engine receives; outer-test rows are absent."""

    frame: pl.DataFrame
    date_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_column: str
    asset_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.date_labels:
            raise ValueError("train-only context requires date labels")
        if self.target_column not in self.frame.columns:
            raise ValueError(f"training target is missing: {self.target_column}")
        missing = sorted(set(self.feature_names) - set(self.frame.columns))
        if missing:
            raise ValueError(f"training features are missing: {missing}")


FeatureContext = TrainOnlyContext


@dataclass(frozen=True)
class TrialBudget:
    max_candidates: int = 8
    max_trials: int = 64
    min_cross_section: int = 20
    min_dates: int = 60

    def __post_init__(self) -> None:
        if self.max_candidates <= 0 or self.max_trials <= 0:
            raise ValueError("trial budget values must be positive")
        if self.min_cross_section < 2 or self.min_dates < 2:
            raise ValueError("minimum sample sizes must be at least two")


@dataclass(frozen=True)
class CandidateSpec:
    recipe_id: str
    engine_id: str
    engine_version: str
    name: str
    thesis: str
    signal_kind: str
    features: tuple[str, ...]
    directions: tuple[int, ...]
    weights: tuple[float, ...]
    parameters: Mapping[str, Any]
    train_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        width = len(self.features)
        if not self.recipe_id or not self.engine_id or not self.engine_version:
            raise ValueError("candidate identity is incomplete")
        if width == 0 or len(self.directions) != width or len(self.weights) != width:
            raise ValueError("candidate feature, direction, and weight widths differ")
        if any(direction not in (-1, 1) for direction in self.directions):
            raise ValueError("candidate directions must be -1 or 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "name": self.name,
            "thesis": self.thesis,
            "signal_kind": self.signal_kind,
            "features": list(self.features),
            "directions": list(self.directions),
            "weights": list(self.weights),
            "parameters": dict(self.parameters),
            "train_evidence": dict(self.train_evidence),
        }


@dataclass(frozen=True)
class FrozenSignalSpec:
    recipe_id: str
    engine_id: str
    engine_version: str
    signal_kind: str
    definition: Mapping[str, Any]

    @classmethod
    def from_candidate(cls, candidate: CandidateSpec) -> FrozenSignalSpec:
        total = sum(abs(float(value)) for value in candidate.weights) or 1.0
        scoring = {
            feature: abs(float(weight)) / total
            for feature, weight in zip(candidate.features, candidate.weights, strict=True)
        }
        directions = {
            feature: "high" if direction > 0 else "low"
            for feature, direction in zip(candidate.features, candidate.directions, strict=True)
        }
        definition = MappingProxyType({
            "kind": "factor_rank",
            "scoring": scoring,
            "directions": directions,
            "parameters": dict(candidate.parameters),
        })
        return cls(
            recipe_id=candidate.recipe_id,
            engine_id=candidate.engine_id,
            engine_version=candidate.engine_version,
            signal_kind=candidate.signal_kind,
            definition=definition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "signal_kind": self.signal_kind,
            "definition": dict(self.definition),
        }


@runtime_checkable
class AlphaDiscoveryEngine(Protocol):
    manifest: AlphaEngineManifest

    def preflight(self, context: DataCatalogContext) -> DataQualification: ...

    def discover(
        self,
        context: TrainOnlyContext,
        budget: TrialBudget,
    ) -> list[CandidateSpec]: ...

    def materialize(
        self,
        candidate: CandidateSpec,
        context: FeatureContext,
    ) -> FrozenSignalSpec: ...


@runtime_checkable
class ResearchFeatureProvider(Protocol):
    manifest: FeatureProviderManifest

    def qualify(self, request: PointInTimeDataRequest) -> DataQualification: ...

    def load(self, request: PointInTimeDataRequest) -> pl.DataFrame: ...


@runtime_checkable
class ResearchEventProvider(Protocol):
    manifest: EventProviderManifest

    def qualify(self, request: PointInTimeDataRequest) -> DataQualification: ...

    def load_events(self, request: PointInTimeDataRequest) -> pl.DataFrame: ...


@runtime_checkable
class CandidateRenderer(Protocol):
    manifest: RendererManifest

    def render(self, candidate: FrozenSignalSpec) -> Mapping[str, Any]: ...


def qualify_manifest_datasets(
    manifest: AlphaEngineManifest,
    catalog: Mapping[str, DataQualification],
) -> DataQualification:
    reasons: list[str] = []
    observations: dict[str, Any] = {}
    for requirement in manifest.required_datasets:
        qualification = catalog.get(requirement.dataset_id)
        if qualification is None:
            reasons.append(f"缺少数据集: {requirement.dataset_id}")
            continue
        coverage = _coverage(qualification.observations.get("coverage"))
        observations[requirement.dataset_id] = {
            "ready": qualification.ready,
            "coverage": coverage,
        }
        if not qualification.ready:
            reasons.extend(qualification.reasons or (f"数据集不可用: {requirement.dataset_id}",))
        elif coverage is not None and coverage < requirement.minimum_coverage:
            reasons.append(
                f"数据集覆盖不足: {requirement.dataset_id}={coverage:.4f}"
                f"<{requirement.minimum_coverage:.4f}"
            )
        if requirement.pit_required and not bool(
            qualification.observations.get("pit_verified", False)
        ):
            reasons.append(f"数据集未通过PIT审计: {requirement.dataset_id}")
    return DataQualification(
        ready=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        observations=observations,
    )


def _coverage(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None
