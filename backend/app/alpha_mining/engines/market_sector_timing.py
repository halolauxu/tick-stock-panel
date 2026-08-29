"""Market-residual cross-sectional discovery."""
from __future__ import annotations

import polars as pl

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DatasetRequirement,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates


class MarketSectorTimingEngine:
    manifest = AlphaEngineManifest(
        engine_id="market_sector_timing",
        version="1.1.0",
        api_version=ENGINE_API_VERSION,
        name="市场残差截面因子",
        family="market_sector_timing",
        information_domains=("price_volume", "liquidity", "fundamentals"),
        mechanism_classes=("relative_mispricing",),
        economic_mechanism="剥离同期市场平均收益后仍可排序的个股特征可能反映横截面错价",
        discovery_classes=("residual_attribution", "cross_sectional_rank"),
        discovery_method="逐项检验事前因子与未来市场残差收益的日度横截面排序关系",
        prediction_objects=("market_residual_return",),
        forecast_targets=("1d", "3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(1, 3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="当前实现只研究市场残差截面排序, 不宣称已经实现市场或行业状态分层。",
    )

    def preflight(self, context: DataCatalogContext):
        from app.alpha_mining.contracts import qualify_manifest_datasets

        return qualify_manifest_datasets(self.manifest, context.datasets)

    def discover(self, context, budget):
        frame = context.frame
        if "target_residual_return" in frame.columns:
            frame = frame.with_columns(
                pl.col("target_residual_return").alias(context.target_column)
            )
            context = type(context)(
                frame=frame,
                date_labels=context.date_labels,
                feature_names=context.feature_names,
                target_column=context.target_column,
                asset_type=context.asset_type,
                metadata=context.metadata,
            )
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="market_sector_timing",
            name_prefix="市场残差因子",
            thesis="训练窗中该事前因子对未来市场残差收益具有稳定横截面排序。",
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = MarketSectorTimingEngine()
