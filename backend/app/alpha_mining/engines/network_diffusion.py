"""Point-in-time industry/concept/supply-chain diffusion discovery."""
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


class NetworkDiffusionEngine:
    manifest = AlphaEngineManifest(
        engine_id="network_diffusion",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="行业与网络扩散",
        family="network_diffusion",
        information_domains=("industry", "concept_network", "price_volume"),
        mechanism_classes=("information_diffusion", "structural_flow"),
        economic_mechanism="主题和产业信息先在领涨节点定价, 再沿点时关系扩散到同群股票",
        discovery_classes=("network_diffusion",),
        discovery_method="使用历史成员关系计算群体领先、广度、扩散速度和个股滞后",
        prediction_objects=("forward_net_return", "market_residual_return"),
        forecast_targets=("3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
            DatasetRequirement("industry_pit", 0.90, True, "in_date"),
        ),
        forecast_horizons=(3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="只接受带生效区间的历史关系; 当前概念快照被统一数据层拒绝。",
    )

    def preflight(self, context: DataCatalogContext):
        data = qualify_manifest_datasets(self.manifest, context.datasets)
        network = tuple(name for name in context.available_features if name.startswith("network_") or name.startswith("industry_"))
        reasons = list(data.reasons)
        if not network:
            reasons.append("缺少点时行业/网络扩散特征")
        return DataQualification(not reasons, tuple(reasons), {**data.observations, "network_features": network})

    def discover(self, context, budget):
        features = tuple(name for name in context.feature_names if name.startswith("network_") or name.startswith("industry_"))
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="network_diffusion",
            name_prefix="扩散特征",
            thesis="历史关系网络上的领先与滞后特征在训练窗中具有可重复的未来净收益排序。",
            features=features,
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = NetworkDiffusionEngine()
