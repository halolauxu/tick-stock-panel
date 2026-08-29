"""Timestamped corporate-event and event-sequence discovery."""
from __future__ import annotations

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DataQualification,
    DatasetRequirement,
    qualify_manifest_datasets,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates


class EventSequenceEngine:
    manifest = AlphaEngineManifest(
        engine_id="event_sequence",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="事件及事件序列",
        family="event_sequence",
        information_domains=("corporate_event", "event_sequence"),
        mechanism_classes=("behavioral_underreaction", "expectation_revision"),
        economic_mechanism="公开事件被分步定价, 多事件的顺序和间隔改变后续收益分布",
        discovery_classes=("event_study", "sequence_pattern"),
        discovery_method="按实际发布时间映射决策日后构造次数、方向、间隔和顺序特征",
        prediction_objects=("forward_net_return", "gap_risk", "untradable_risk"),
        forecast_targets=("1d", "3d", "5d", "10d", "20d", "60d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
            DatasetRequirement("event_history", 0.90, True, "published_at"),
        ),
        forecast_horizons=(1, 3, 5, 10, 20, 60),
        output_candidate_types=("factor_rank",),
        description="事件只能在发布时间后的合法决策时点进入研究, 周末事件顺延。",
    )

    def preflight(self, context: DataCatalogContext):
        data = qualify_manifest_datasets(self.manifest, context.datasets)
        event_features = tuple(name for name in context.available_features if name.startswith("event_"))
        reasons = list(data.reasons)
        if not event_features:
            reasons.append("缺少通过ResearchEventProvider生成的event_*特征")
        return DataQualification(not reasons, tuple(reasons), {**data.observations, "event_features": event_features})

    def discover(self, context, budget):
        features = tuple(name for name in context.feature_names if name.startswith("event_"))
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="event_sequence",
            name_prefix="事件序列",
            thesis="该公开事件序列特征在训练窗中对未来净收益保持稳定方向。",
            features=features,
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = EventSequenceEngine()
