import math
import random

import pytest

from rl.action import Action
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_table import QTable
from rl.state_key import StateKey


def create_q_table() -> QTable:
    return QTable()


def create_policy() -> EpsilonGreedyPolicy:
    return EpsilonGreedyPolicy(
        epsilon=0.1,
        random_generator=random.Random(42),
    )


def create_agent(
    learning_rate: float = 0.5,
    discount_factor: float = 0.9,
) -> QLearningAgent:
    return QLearningAgent(
        q_table=create_q_table(),
        policy=create_policy(),
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )


def create_state_key(
    coverage_bucket: int = 5,
    missing_lines_bucket: int = 2,
    uncovered_branches_bucket: int = 1,
    executed_tests_bucket: int = 0,
) -> StateKey:
    """
    QLearningAgent testlerinde kullanılacak StateKey nesnesini oluşturur.

    executed_tests_bucket, RL state yapısına sonradan eklenen
    çalıştırılmış test sayısı bilgisini temsil eder.
    """
    return StateKey(
        coverage_bucket=coverage_bucket,
        missing_lines_bucket=missing_lines_bucket,
        uncovered_branches_bucket=uncovered_branches_bucket,
        executed_tests_bucket=executed_tests_bucket,
    )


def create_action(
    scenario_index: int = 0,
) -> Action:
    return Action(
        scenario_index=scenario_index,
    )


def test_agent_can_be_created() -> None:
    q_table = create_q_table()
    policy = create_policy()

    agent = QLearningAgent(
        q_table=q_table,
        policy=policy,
        learning_rate=0.5,
        discount_factor=0.9,
    )

    assert agent.q_table is q_table
    assert agent.policy is policy
    assert agent.learning_rate == 0.5
    assert agent.discount_factor == 0.9


def test_agent_converts_integer_rates_to_float() -> None:
    agent = QLearningAgent(
        q_table=create_q_table(),
        policy=create_policy(),
        learning_rate=1,
        discount_factor=0,
    )

    assert agent.learning_rate == 1.0
    assert agent.discount_factor == 0.0


@pytest.mark.parametrize(
    "learning_rate",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_agent_accepts_valid_learning_rates(
    learning_rate: float,
) -> None:
    agent = create_agent(
        learning_rate=learning_rate,
    )

    assert agent.learning_rate == learning_rate


@pytest.mark.parametrize(
    "discount_factor",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_agent_accepts_valid_discount_factors(
    discount_factor: float,
) -> None:
    agent = create_agent(
        discount_factor=discount_factor,
    )

    assert agent.discount_factor == discount_factor


@pytest.mark.parametrize(
    "name,value",
    [
        ("learning_rate", -0.01),
        ("learning_rate", 1.01),
        ("discount_factor", -0.01),
        ("discount_factor", 1.01),
    ],
)
def test_agent_rejects_rates_outside_valid_range(
    name: str,
    value: float,
) -> None:
    kwargs = {
        "q_table": create_q_table(),
        "policy": create_policy(),
        "learning_rate": 0.5,
        "discount_factor": 0.9,
    }

    kwargs[name] = value

    with pytest.raises(
        ValueError,
        match=(
            f"{name} must be between 0.0 and 1.0."
        ),
    ):
        QLearningAgent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name,value",
    [
        ("learning_rate", math.nan),
        ("learning_rate", math.inf),
        ("discount_factor", math.nan),
        ("discount_factor", -math.inf),
    ],
)
def test_agent_rejects_non_finite_rates(
    name: str,
    value: float,
) -> None:
    kwargs = {
        "q_table": create_q_table(),
        "policy": create_policy(),
        "learning_rate": 0.5,
        "discount_factor": 0.9,
    }

    kwargs[name] = value

    with pytest.raises(
        ValueError,
        match=f"{name} must be a finite value.",
    ):
        QLearningAgent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name,value",
    [
        ("learning_rate", True),
        ("learning_rate", "0.5"),
        ("discount_factor", False),
        ("discount_factor", None),
    ],
)
def test_agent_rejects_invalid_rate_types(
    name: str,
    value: object,
) -> None:
    kwargs = {
        "q_table": create_q_table(),
        "policy": create_policy(),
        "learning_rate": 0.5,
        "discount_factor": 0.9,
    }

    kwargs[name] = value

    with pytest.raises(
        TypeError,
        match=f"{name} must be a numeric value.",
    ):
        QLearningAgent(**kwargs)  # type: ignore[arg-type]


def test_agent_rejects_invalid_q_table() -> None:
    with pytest.raises(
        TypeError,
        match="q_table must be a QTable instance.",
    ):
        QLearningAgent(
            q_table="invalid",  # type: ignore[arg-type]
            policy=create_policy(),
            learning_rate=0.5,
            discount_factor=0.9,
        )


def test_agent_rejects_invalid_policy() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "policy must be an "
            "EpsilonGreedyPolicy instance."
        ),
    ):
        QLearningAgent(
            q_table=create_q_table(),
            policy="invalid",  # type: ignore[arg-type]
            learning_rate=0.5,
            discount_factor=0.9,
        )


def test_agent_select_action_uses_policy_and_q_table() -> None:
    q_table = QTable()

    policy = EpsilonGreedyPolicy(
        epsilon=0.0,
        random_generator=random.Random(42),
    )

    agent = QLearningAgent(
        q_table=q_table,
        policy=policy,
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key=state_key,
        action=first_action,
        value=2.0,
    )

    q_table.set_value(
        state_key=state_key,
        action=second_action,
        value=8.0,
    )

    selected_action = agent.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
    )

    assert selected_action == second_action


def test_agent_update_stores_new_q_value() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=2.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=6.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=10.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    assert result == pytest.approx(8.7)

    stored_value = agent.q_table.get_value(
        state_key=state_key,
        action=action,
    )

    assert stored_value == pytest.approx(8.7)


def test_agent_update_uses_highest_next_q_value() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    current_action = create_action(0)
    first_next_action = create_action(1)
    second_next_action = create_action(2)
    third_next_action = create_action(3)

    agent.q_table.set_value(
        state_key=state_key,
        action=current_action,
        value=4.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=first_next_action,
        value=3.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=second_next_action,
        value=10.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=third_next_action,
        value=7.0,
    )

    result = agent.update(
        state_key=state_key,
        action=current_action,
        reward=2.0,
        next_state_key=next_state_key,
        next_actions=(
            first_next_action,
            second_next_action,
            third_next_action,
        ),
    )

    expected_value = (
        4.0
        + 0.5
        * (
            2.0
            + 0.9 * 10.0
            - 4.0
        )
    )

    assert result == pytest.approx(
        expected_value,
    )


def test_agent_update_uses_zero_for_unknown_current_q_value() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=4.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=2.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    expected_value = (
        0.0
        + 0.5
        * (
            2.0
            + 0.9 * 4.0
            - 0.0
        )
    )

    assert result == pytest.approx(
        expected_value,
    )


def test_agent_update_uses_zero_for_unknown_next_q_values() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=4.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=2.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    expected_value = (
        4.0
        + 0.5
        * (
            2.0
            + 0.9 * 0.0
            - 4.0
        )
    )

    assert result == pytest.approx(
        expected_value,
    )


def test_agent_update_accepts_empty_next_actions() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=6.0,
        next_state_key=next_state_key,
        next_actions=(),
    )

    assert result == pytest.approx(3.0)


def test_agent_terminal_update_ignores_future_q_value() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=10,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=2.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=100.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=10.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
        terminal=True,
    )

    expected_value = (
        2.0
        + 0.5
        * (
            10.0
            - 2.0
        )
    )

    assert result == pytest.approx(
        expected_value,
    )


def test_agent_update_with_zero_learning_rate_keeps_old_value() -> None:
    agent = create_agent(
        learning_rate=0.0,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=7.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=100.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=50.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    assert result == pytest.approx(7.0)


def test_agent_update_with_one_learning_rate_uses_target_value() -> None:
    agent = create_agent(
        learning_rate=1.0,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=2.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=10.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=5.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    expected_target = (
        5.0
        + 0.9 * 10.0
    )

    assert result == pytest.approx(
        expected_target,
    )


def test_agent_update_with_zero_discount_factor_ignores_future_value(
) -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.0,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=6,
    )

    action = create_action(0)
    next_action = create_action(1)

    agent.q_table.set_value(
        state_key=state_key,
        action=action,
        value=2.0,
    )

    agent.q_table.set_value(
        state_key=next_state_key,
        action=next_action,
        value=100.0,
    )

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=10.0,
        next_state_key=next_state_key,
        next_actions=(next_action,),
    )

    expected_value = (
        2.0
        + 0.5
        * (
            10.0
            - 2.0
        )
    )

    assert result == pytest.approx(
        expected_value,
    )


def test_agent_update_accepts_negative_reward() -> None:
    agent = create_agent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_key = create_state_key(
        coverage_bucket=5,
    )

    next_state_key = create_state_key(
        coverage_bucket=4,
    )

    action = create_action(0)

    result = agent.update(
        state_key=state_key,
        action=action,
        reward=-10.0,
        next_state_key=next_state_key,
        next_actions=(),
    )

    assert result == pytest.approx(-5.0)


def test_agent_update_rejects_invalid_state_key() -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match="state_key must be a StateKey instance.",
    ):
        agent.update(
            state_key="invalid",  # type: ignore[arg-type]
            action=create_action(),
            reward=1.0,
            next_state_key=create_state_key(),
            next_actions=(),
        )


def test_agent_update_rejects_invalid_action() -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match="action must be an Action instance.",
    ):
        agent.update(
            state_key=create_state_key(),
            action="invalid",  # type: ignore[arg-type]
            reward=1.0,
            next_state_key=create_state_key(),
            next_actions=(),
        )


@pytest.mark.parametrize(
    "reward",
    [
        True,
        False,
        "10",
        None,
    ],
)
def test_agent_update_rejects_invalid_reward_type(
    reward: object,
) -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match="reward must be a numeric value.",
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=reward,  # type: ignore[arg-type]
            next_state_key=create_state_key(),
            next_actions=(),
        )


@pytest.mark.parametrize(
    "reward",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_agent_update_rejects_non_finite_reward(
    reward: float,
) -> None:
    agent = create_agent()

    with pytest.raises(
        ValueError,
        match="reward must be a finite value.",
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=reward,
            next_state_key=create_state_key(),
            next_actions=(),
        )


def test_agent_update_rejects_invalid_next_state_key() -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match="next_state_key must be a StateKey instance.",
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=1.0,
            next_state_key="invalid",  # type: ignore[arg-type]
            next_actions=(),
        )


def test_agent_update_rejects_non_tuple_next_actions() -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match=(
            "next_actions must be a tuple "
            "of Action instances."
        ),
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=1.0,
            next_state_key=create_state_key(),
            next_actions=[create_action()],  # type: ignore[arg-type]
        )


def test_agent_update_rejects_invalid_next_action_item() -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match=(
            "next_actions must contain only "
            "Action instances."
        ),
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=1.0,
            next_state_key=create_state_key(),
            next_actions=(
                create_action(),
                "invalid",  # type: ignore[arg-type]
            ),
        )


@pytest.mark.parametrize(
    "terminal",
    [
        0,
        1,
        "False",
        None,
    ],
)
def test_agent_update_rejects_invalid_terminal(
    terminal: object,
) -> None:
    agent = create_agent()

    with pytest.raises(
        TypeError,
        match="terminal must be a bool value.",
    ):
        agent.update(
            state_key=create_state_key(),
            action=create_action(),
            reward=1.0,
            next_state_key=create_state_key(),
            next_actions=(),
            terminal=terminal,  # type: ignore[arg-type]
        )


def test_create_state_key_supports_executed_tests_bucket() -> None:
    state_key = create_state_key(
        coverage_bucket=7,
        missing_lines_bucket=1,
        uncovered_branches_bucket=0,
        executed_tests_bucket=4,
    )

    assert state_key.coverage_bucket == 7
    assert state_key.missing_lines_bucket == 1
    assert state_key.uncovered_branches_bucket == 0
    assert state_key.executed_tests_bucket == 4

