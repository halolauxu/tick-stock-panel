"""Non-bypassable research-state machine for Alpha experiments."""
from __future__ import annotations

from enum import StrEnum


class AlphaExperimentState(StrEnum):
    DRAFT = "draft"
    REGISTERED = "registered"
    DATA_READY = "data_ready"
    DISCOVERY = "discovery"
    FROZEN = "frozen"
    OUTER_EVALUATED = "outer_evaluated"
    REJECTED = "rejected"
    RESEARCH_CANDIDATE = "research_candidate"
    SHADOW = "shadow"
    CHALLENGER = "challenger"
    CHAMPION = "champion"


class InvalidAlphaStateTransitionError(ValueError):
    pass


_ALLOWED: dict[AlphaExperimentState, frozenset[AlphaExperimentState]] = {
    AlphaExperimentState.DRAFT: frozenset({AlphaExperimentState.REGISTERED}),
    AlphaExperimentState.REGISTERED: frozenset({AlphaExperimentState.DATA_READY}),
    AlphaExperimentState.DATA_READY: frozenset({AlphaExperimentState.DISCOVERY}),
    AlphaExperimentState.DISCOVERY: frozenset({AlphaExperimentState.FROZEN}),
    AlphaExperimentState.FROZEN: frozenset({AlphaExperimentState.OUTER_EVALUATED}),
    AlphaExperimentState.OUTER_EVALUATED: frozenset({
        AlphaExperimentState.REJECTED,
        AlphaExperimentState.RESEARCH_CANDIDATE,
    }),
    AlphaExperimentState.RESEARCH_CANDIDATE: frozenset({AlphaExperimentState.SHADOW}),
    AlphaExperimentState.SHADOW: frozenset({
        AlphaExperimentState.REJECTED,
        AlphaExperimentState.CHALLENGER,
    }),
    AlphaExperimentState.CHALLENGER: frozenset({
        AlphaExperimentState.REJECTED,
        AlphaExperimentState.CHAMPION,
    }),
    AlphaExperimentState.REJECTED: frozenset(),
    AlphaExperimentState.CHAMPION: frozenset(),
}


def transition_alpha_state(
    current: AlphaExperimentState | str,
    target: AlphaExperimentState | str,
) -> AlphaExperimentState:
    source = AlphaExperimentState(current)
    destination = AlphaExperimentState(target)
    if destination not in _ALLOWED[source]:
        raise InvalidAlphaStateTransitionError(
            f"invalid Alpha state transition: {source} -> {destination}"
        )
    return destination
