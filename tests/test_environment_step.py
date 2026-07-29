from dataclasses import FrozenInstanceError

import pytest

from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep


def create_state(
    coverage_percentage: float = 50.0,
    executed_tests: int = 1,
) -> CoverageState:
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=(),
        uncovered_branches=0,
    )


def test_environment_step_can_be_created() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=5.0,
        done=False,
    )

    assert step.state == state
    assert step.reward == 5.0
    assert step.done is False


def test_environment_step_accepts_integer_reward() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=10,
        done=False,
    )

    assert step.reward == 10


def test_environment_step_accepts_negative_reward() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=-2.5,
        done=False,
    )

    assert step.reward == -2.5


def test_environment_step_accepts_zero_reward() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=0.0,
        done=False,
    )

    assert step.reward == 0.0


def test_environment_step_can_mark_episode_as_done() -> None:
    state = create_state(
        coverage_percentage=100.0,
        executed_tests=5,
    )

    step = EnvironmentStep(
        state=state,
        reward=15.0,
        done=True,
    )

    assert step.done is True


def test_environment_step_rejects_invalid_state() -> None:
    with pytest.raises(
        TypeError,
        match="state must be a CoverageState instance.",
    ):
        EnvironmentStep(
            state="invalid",  # type: ignore[arg-type]
            reward=1.0,
            done=False,
        )


@pytest.mark.parametrize(
    "invalid_reward",
    [
        "5",
        None,
        True,
        False,
    ],
)
def test_environment_step_rejects_non_numeric_reward(
    invalid_reward: object,
) -> None:
    state = create_state()

    with pytest.raises(
        TypeError,
        match="reward must be a number.",
    ):
        EnvironmentStep(
            state=state,
            reward=invalid_reward,  # type: ignore[arg-type]
            done=False,
        )


@pytest.mark.parametrize(
    "invalid_done",
    [
        0,
        1,
        "False",
        None,
    ],
)
def test_environment_step_rejects_non_boolean_done(
    invalid_done: object,
) -> None:
    state = create_state()

    with pytest.raises(
        TypeError,
        match="done must be a boolean.",
    ):
        EnvironmentStep(
            state=state,
            reward=1.0,
            done=invalid_done,  # type: ignore[arg-type]
        )


def test_environment_step_is_immutable() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=5.0,
        done=False,
    )

    with pytest.raises(FrozenInstanceError):
        step.reward = 10.0  # type: ignore[misc]


def test_environment_steps_with_same_values_are_equal() -> None:
    state = create_state()

    first_step = EnvironmentStep(
        state=state,
        reward=5.0,
        done=False,
    )

    second_step = EnvironmentStep(
        state=state,
        reward=5.0,
        done=False,
    )

    assert first_step == second_step


def test_environment_step_has_readable_representation() -> None:
    state = create_state()

    step = EnvironmentStep(
        state=state,
        reward=5.0,
        done=False,
    )

    representation = repr(step)

    assert "EnvironmentStep" in representation
    assert "reward=5.0" in representation
    assert "done=False" in representation