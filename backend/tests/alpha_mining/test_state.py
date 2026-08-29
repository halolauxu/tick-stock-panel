from __future__ import annotations

# Requirements: AM-S6-011.
import pytest

from app.alpha_mining.state import (
    AlphaExperimentState,
    InvalidAlphaStateTransitionError,
    transition_alpha_state,
)


def test_alpha_state_cannot_skip_outer_evaluation() -> None:
    with pytest.raises(InvalidAlphaStateTransitionError):
        transition_alpha_state(
            AlphaExperimentState.FROZEN,
            AlphaExperimentState.RESEARCH_CANDIDATE,
        )


def test_alpha_state_reaches_research_candidate_only_through_outer_evaluation() -> None:
    state = AlphaExperimentState.DRAFT
    for target in (
        AlphaExperimentState.REGISTERED,
        AlphaExperimentState.DATA_READY,
        AlphaExperimentState.DISCOVERY,
        AlphaExperimentState.FROZEN,
        AlphaExperimentState.OUTER_EVALUATED,
        AlphaExperimentState.RESEARCH_CANDIDATE,
    ):
        state = transition_alpha_state(state, target)
    assert state is AlphaExperimentState.RESEARCH_CANDIDATE
