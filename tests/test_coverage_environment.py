import pytest

from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.reward_calculator import RewardCalculator


def create_state(
    coverage_percentage: float = 50.0,
    executed_tests: int = 0,
) -> CoverageState:
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=(),
        uncovered_branches=0,
    )


def increase_coverage(
    current_state: CoverageState,
    action: Action,
) -> CoverageState:
    coverage_gain = float((action.scenario_index + 1) * 10)

    return create_state(
        coverage_percentage=min(
            100.0,
            current_state.coverage_percentage + coverage_gain,
        ),
        executed_tests=current_state.executed_tests + 1,
    )


def test_environment_can_be_created() -> None:
    initial_state = create_state()

    actions = (
        Action(scenario_index=0),
        Action(scenario_index=1),
    )

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=actions,
        transition_function=increase_coverage,
    )

    assert environment.current_state == initial_state
    assert environment.available_actions == actions
    assert environment.is_done is False


def test_step_updates_state_and_returns_environment_step() -> None:
    action = Action(scenario_index=0)

    environment = CoverageEnvironment(
        initial_state=create_state(50.0),
        actions=(action,),
        transition_function=increase_coverage,
    )

    step = environment.step(action)

    assert isinstance(step, EnvironmentStep)
    assert step.state.coverage_percentage == 60.0
    assert step.state.executed_tests == 1
    assert step.reward == pytest.approx(9.9)


def test_used_action_is_removed() -> None:
    first_action = Action(scenario_index=0)
    second_action = Action(scenario_index=1)

    environment = CoverageEnvironment(
        initial_state=create_state(),
        actions=(first_action, second_action),
        transition_function=increase_coverage,
    )

    environment.step(first_action)

    assert environment.available_actions == (second_action,)


def test_episode_ends_when_full_coverage_is_reached() -> None:
    action = Action(scenario_index=1)

    environment = CoverageEnvironment(
        initial_state=create_state(90.0),
        actions=(action,),
        transition_function=increase_coverage,
    )

    step = environment.step(action)

    assert step.state.coverage_percentage == 100.0
    assert step.done is True
    assert environment.is_done is True


def test_episode_ends_when_no_actions_remain() -> None:
    action = Action(scenario_index=0)

    environment = CoverageEnvironment(
        initial_state=create_state(20.0),
        actions=(action,),
        transition_function=increase_coverage,
    )

    step = environment.step(action)

    assert step.done is True
    assert environment.is_done is True


def test_reset_restores_initial_state_and_actions() -> None:
    first_action = Action(scenario_index=0)
    second_action = Action(scenario_index=1)
    initial_state = create_state(40.0)

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=(first_action, second_action),
        transition_function=increase_coverage,
    )

    environment.step(first_action)
    reset_state = environment.reset()

    assert reset_state == initial_state
    assert environment.current_state == initial_state
    assert environment.available_actions == (
        first_action,
        second_action,
    )


def test_step_rejects_unavailable_action() -> None:
    environment = CoverageEnvironment(
        initial_state=create_state(),
        actions=(Action(scenario_index=0),),
        transition_function=increase_coverage,
    )

    with pytest.raises(
        ValueError,
        match="action is not available.",
    ):
        environment.step(Action(scenario_index=5))


def test_environment_rejects_duplicate_actions() -> None:
    action = Action(scenario_index=1)

    with pytest.raises(
        ValueError,
        match="actions cannot contain duplicates.",
    ):
        CoverageEnvironment(
            initial_state=create_state(),
            actions=(action, action),
            transition_function=increase_coverage,
        )


def test_transition_must_return_coverage_state() -> None:
    def invalid_transition(
        current_state: CoverageState,
        action: Action,
    ) -> str:
        return "invalid"

    action = Action(scenario_index=0)

    environment = CoverageEnvironment(
        initial_state=create_state(),
        actions=(action,),
        transition_function=invalid_transition,  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match=(
            "transition_function must return "
            "a CoverageState instance."
        ),
    ):
        environment.step(action)


def test_custom_reward_calculator_is_used() -> None:
    action = Action(scenario_index=0)

    environment = CoverageEnvironment(
        initial_state=create_state(50.0),
        actions=(action,),
        transition_function=increase_coverage,
        reward_calculator=RewardCalculator(
            test_execution_cost=1.0,
        ),
    )

    step = environment.step(action)

    assert step.reward == pytest.approx(9.0)