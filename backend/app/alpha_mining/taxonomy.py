"""Closed vocabularies for the four Alpha discovery dimensions.

The vocabularies are extensible by a contract-version change. Free-form human
descriptions stay in the engine manifest, while these IDs make coverage and
admission mechanically auditable.
"""
from __future__ import annotations

INFORMATION_DOMAINS = frozenset({
    "price_volume",
    "liquidity",
    "fundamentals",
    "market_regime",
    "industry",
    "concept_network",
    "corporate_event",
    "event_sequence",
    "strategy_residual",
    "portfolio",
    "auction_microstructure",
    "holder_supply",
    "event_text",
    "cross_asset",
})

MECHANISM_CLASSES = frozenset({
    "risk_compensation",
    "behavioral_underreaction",
    "behavioral_overreaction",
    "liquidity_pressure",
    "information_diffusion",
    "expectation_revision",
    "crowding_unwind",
    "structural_flow",
    "relative_mispricing",
    "portfolio_complementarity",
})

DISCOVERY_CLASSES = frozenset({
    "cross_sectional_rank",
    "conditional_time_series",
    "matched_outcome_attribution",
    "event_study",
    "sequence_pattern",
    "network_diffusion",
    "revision_surprise",
    "residual_attribution",
    "nonlinear_interaction",
    "relative_value",
})

PREDICTION_OBJECTS = frozenset({
    "forward_net_return",
    "market_residual_return",
    "mfe",
    "mae",
    "gap_risk",
    "untradable_risk",
    "rank_outperformance",
})


def coverage_matrix(manifests) -> dict[str, list[dict[str, object]]]:
    """Project manifests into all four dimensions without hiding empty cells."""
    rows = [manifest.to_dict() for manifest in manifests]
    return {
        "information_domain": _dimension(rows, "information_domains", INFORMATION_DOMAINS),
        "mechanism": _dimension(rows, "mechanism_classes", MECHANISM_CLASSES),
        "discovery": _dimension(rows, "discovery_classes", DISCOVERY_CLASSES),
        "prediction_object": _dimension(rows, "prediction_objects", PREDICTION_OBJECTS),
    }


def _dimension(
    rows: list[dict[str, object]],
    field: str,
    vocabulary: frozenset[str],
) -> list[dict[str, object]]:
    output = []
    for value in sorted(vocabulary):
        engine_ids = [
            str(row["engine_id"])
            for row in rows
            if value in set(row.get(field) or [])
        ]
        output.append({"id": value, "engine_ids": engine_ids, "covered": bool(engine_ids)})
    return output
