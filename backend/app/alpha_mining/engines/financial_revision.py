"""Point-in-time financial factor discovery."""
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

FINANCIAL_FEATURES = (
    "pb_latest",
    "roe_latest",
    "gross_margin_latest",
    "net_margin_latest",
    "revenue_yoy_latest",
    "net_income_yoy_latest",
    "debt_ratio_latest",
    "financial_revision_revenue",
    "financial_revision_profit",
)


class FinancialRevisionEngine:
    manifest = AlphaEngineManifest(
        engine_id="financial_revision",
        version="1.1.0",
        api_version=ENGINE_API_VERSION,
        name="公告时点财务因子",
        family="financial_revision",
        information_domains=("fundamentals", "corporate_event"),
        mechanism_classes=("relative_mispricing", "expectation_revision"),
        economic_mechanism="公告后已知的估值、盈利质量、增长或修订字段可能形成横截面错价",
        discovery_classes=("cross_sectional_rank",),
        discovery_method="逐项检验公告时点可见财务字段与未来净收益的日度横截面排序关系",
        prediction_objects=("forward_net_return", "market_residual_return", "gap_risk"),
        forecast_targets=("3d", "5d", "10d", "20d", "60d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
            DatasetRequirement("financial_pit", 0.90, True, "announce_date"),
        ),
        forecast_horizons=(3, 5, 10, 20, 60),
        output_candidate_types=("factor_rank",),
        description="财务字段在公告前必须为空; 报告期不能替代公告时间。",
    )

    def preflight(self, context: DataCatalogContext):
        data = qualify_manifest_datasets(self.manifest, context.datasets)
        available = tuple(name for name in FINANCIAL_FEATURES if name in context.available_features)
        reasons = list(data.reasons)
        if not available:
            reasons.append("缺少公告日门控的财务特征")
        return DataQualification(not reasons, tuple(reasons), {**data.observations, "financial_features": available})

    def discover(self, context, budget):
        features = tuple(name for name in FINANCIAL_FEATURES if name in context.feature_names)
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="financial_revision",
            name_prefix="公告财务因子",
            thesis="公告后可见的财务字段在训练窗中对未来净收益具有稳定横截面排序; 字段本身不等同于预期差。",
            features=features,
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = FinancialRevisionEngine()
