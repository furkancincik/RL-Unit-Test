import random

import pytest

from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.state_encoder import StateEncoder


def create_initial_state() -> CoverageState:
    return CoverageState(
        coverage_percentage=50.0,
        executed_tests=0,
        missing_lines=(
            10,
            20,
            30,
            40,
        ),
        uncovered_branches=2,
    )


def create_actions() -> tuple[Action, ...]:
    return (
        Action(scenario_index=0),
        Action(scenario_index=1),
    )


def create_agent(
    epsilon: float = 0.0,
) -> QLearningAgent:
    return QLearningAgent(
        q_table=QTable(),
        policy=EpsilonGreedyPolicy(
            epsilon=epsilon,
            random_generator=random.Random(42),
        ),
        learning_rate=0.5,
        discount_factor=0.9,
    )


def create_trainer() -> QLearningTrainer:
    return QLearningTrainer(
        agent=create_agent(),
        state_encoder=StateEncoder(),
    )


def transition_to_full_coverage(
    state: CoverageState,
    action: Action,
) -> CoverageState:
    return CoverageState(
        coverage_percentage=100.0,
        executed_tests=state.executed_tests + 1,
        missing_lines=(),
        uncovered_branches=0,
    )


def transition_by_action(
    state: CoverageState,
    action: Action,
) -> CoverageState:
    if action.scenario_index == 0:
        return CoverageState(
            coverage_percentage=70.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(30, 40),
            uncovered_branches=1,
        )

    return CoverageState(
        coverage_percentage=100.0,
        executed_tests=state.executed_tests + 1,
        missing_lines=(),
        uncovered_branches=0,
    )


def test_trainer_can_be_created() -> None:
    agent = create_agent()
    state_encoder = StateEncoder()

    trainer = QLearningTrainer(
        agent=agent,
        state_encoder=state_encoder,
    )

    assert trainer.agent is agent
    assert trainer.state_encoder is state_encoder


def test_trainer_rejects_invalid_agent() -> None:
    with pytest.raises(
        TypeError,
        match="agent must be a QLearningAgent instance.",
    ):
        QLearningTrainer(
            agent="invalid",  # type: ignore[arg-type]
            state_encoder=StateEncoder(),
        )


def test_trainer_rejects_invalid_state_encoder() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state_encoder must be a "
            "StateEncoder instance."
        ),
    ):
        QLearningTrainer(
            agent=create_agent(),
            state_encoder="invalid",  # type: ignore[arg-type]
        )


def test_train_step_applies_selected_action() -> None:
    actions = create_actions()

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=actions,
        transition_function=transition_to_full_coverage,
    )

    trainer = create_trainer()

    step = trainer.train_step(
        environment=environment,
    )

    assert step.state.is_fully_covered
    assert step.done is True
    assert step.action in actions
    assert step.done_reason == "FULL_COVERAGE"
    assert environment.current_state == step.state
    assert len(environment.available_actions) == 1


def test_train_step_updates_q_table() -> None:
    actions = create_actions()

    initial_state = create_initial_state()

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=actions,
        transition_function=transition_to_full_coverage,
    )

    trainer = create_trainer()

    initial_state_key = trainer.state_encoder.encode(
        state=initial_state,
    )

    selected_action = trainer.agent.select_action(
        state_key=initial_state_key,
        actions=actions,
    )

    old_q_value = trainer.agent.q_table.get_value(
        state_key=initial_state_key,
        action=selected_action,
    )

    trainer.train_step(
        environment=environment,
    )

    new_q_value = trainer.agent.q_table.get_value(
        state_key=initial_state_key,
        action=selected_action,
    )

    assert old_q_value == 0.0
    assert new_q_value != old_q_value


def test_train_step_rejects_invalid_environment() -> None:
    trainer = create_trainer()

    with pytest.raises(
        TypeError,
        match=(
            "environment must be a "
            "CoverageEnvironment instance."
        ),
    ):
        trainer.train_step(
            environment="invalid",  # type: ignore[arg-type]
        )


def test_train_step_rejects_completed_episode() -> None:
    environment = CoverageEnvironment(
        initial_state=CoverageState(
            coverage_percentage=100.0,
            executed_tests=1,
            missing_lines=(),
            uncovered_branches=0,
        ),
        actions=create_actions(),
        transition_function=transition_to_full_coverage,
    )

    trainer = create_trainer()

    with pytest.raises(
        RuntimeError,
        match="cannot train on a completed episode.",
    ):
        trainer.train_step(
            environment=environment,
        )


def test_train_episode_runs_until_full_coverage() -> None:
    actions = create_actions()

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=actions,
        transition_function=transition_by_action,
    )

    trainer = create_trainer()

    steps = trainer.train_episode(
        environment=environment,
    )

    assert len(steps) == 2
    assert steps[0].done is False
    assert steps[1].done is True
    assert environment.current_state.is_fully_covered
    assert environment.is_done is True


def test_train_episode_stops_when_actions_are_exhausted() -> None:
    initial_state = create_initial_state()

    actions = (
        Action(scenario_index=0),
    )

    def partial_transition(
        state: CoverageState,
        action: Action,
    ) -> CoverageState:
        return CoverageState(
            coverage_percentage=70.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(30, 40),
            uncovered_branches=1,
        )

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=actions,
        transition_function=partial_transition,
    )

    trainer = create_trainer()

    steps = trainer.train_episode(
        environment=environment,
    )

    assert len(steps) == 1
    assert steps[0].done is True
    assert environment.current_state.is_fully_covered is False
    assert environment.available_actions == ()
    assert environment.is_done is True


def test_train_episode_resets_environment_by_default() -> None:
    actions = create_actions()

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=actions,
        transition_function=transition_to_full_coverage,
    )

    environment.step(
        action=actions[0],
    )

    trainer = create_trainer()

    steps = trainer.train_episode(
        environment=environment,
    )

    assert len(steps) == 1
    assert environment.current_state.is_fully_covered
    assert len(environment.available_actions) == 1


def test_train_episode_without_reset_keeps_current_state() -> None:
    actions = create_actions()

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=actions,
        transition_function=transition_to_full_coverage,
    )

    environment.step(
        action=actions[0],
    )

    trainer = create_trainer()

    steps = trainer.train_episode(
        environment=environment,
        reset=False,
    )

    assert steps == ()


@pytest.mark.parametrize(
    "reset",
    [
        0,
        1,
        "True",
        None,
    ],
)
def test_train_episode_rejects_invalid_reset(
    reset: object,
) -> None:
    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=create_actions(),
        transition_function=transition_to_full_coverage,
    )

    trainer = create_trainer()

    with pytest.raises(
        TypeError,
        match="reset must be a bool value.",
    ):
        trainer.train_episode(
            environment=environment,
            reset=reset,  # type: ignore[arg-type]
        )
