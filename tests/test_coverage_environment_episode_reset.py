from __future__ import annotations

from unittest.mock import Mock

import pytest

from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState


def create_state() -> CoverageState:
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3),
        uncovered_branches=2,
    )


def create_environment(
    *,
    reset_callback=None,
) -> CoverageEnvironment:
    initial_state = create_state()

    def transition(
        state: CoverageState,
        action: Action,
    ) -> CoverageState:
        return CoverageState(
            coverage_percentage=50.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(1,),
            uncovered_branches=1,
        )

    return CoverageEnvironment(
        initial_state=initial_state,
        actions=(
            Action(scenario_index=0),
            Action(scenario_index=1),
        ),
        transition_function=transition,
        episode_reset_callback=reset_callback,
    )


def test_reset_calls_episode_reset_callback() -> None:
    callback = Mock()

    environment = create_environment(
        reset_callback=callback,
    )

    environment.step(
        environment.available_actions[0]
    )

    environment.reset()

    callback.assert_called_once_with()


def test_reset_restores_state_and_actions() -> None:
    environment = create_environment()

    initial_state = environment.current_state
    initial_actions = environment.available_actions

    environment.step(
        initial_actions[0]
    )

    result = environment.reset()

    assert result is initial_state
    assert environment.current_state is initial_state
    assert environment.available_actions == initial_actions
    assert environment.is_done is False


def test_reset_without_callback_remains_supported() -> None:
    environment = create_environment()

    environment.step(
        environment.available_actions[0]
    )

    result = environment.reset()

    assert result.coverage_percentage == 0.0
    assert len(environment.available_actions) == 2


def test_environment_rejects_invalid_reset_callback() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "episode_reset_callback callable "
            "veya None olmalıdır"
        ),
    ):
        create_environment(
            reset_callback="invalid",
        )