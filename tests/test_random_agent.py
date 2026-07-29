import pytest

from rl.action import Action
from rl.random_agent import RandomAgent


def test_random_agent_can_be_created() -> None:
    agent = RandomAgent()

    assert isinstance(agent, RandomAgent)


def test_random_agent_selects_available_action() -> None:
    agent = RandomAgent(seed=42)

    actions = (
        Action(scenario_index=0),
        Action(scenario_index=1),
        Action(scenario_index=2),
    )

    selected_action = agent.select_action(actions)

    assert selected_action in actions


def test_random_agent_returns_action_instance() -> None:
    agent = RandomAgent(seed=5)

    actions = (
        Action(scenario_index=0),
        Action(scenario_index=1),
    )

    selected_action = agent.select_action(actions)

    assert isinstance(selected_action, Action)


def test_random_agent_is_deterministic_with_same_seed() -> None:
    actions = (
        Action(scenario_index=0),
        Action(scenario_index=1),
        Action(scenario_index=2),
        Action(scenario_index=3),
    )

    first_agent = RandomAgent(seed=123)
    second_agent = RandomAgent(seed=123)

    first_selection = first_agent.select_action(actions)
    second_selection = second_agent.select_action(actions)

    assert first_selection == second_selection


def test_random_agent_can_select_from_single_action() -> None:
    agent = RandomAgent(seed=10)

    action = Action(scenario_index=4)

    selected_action = agent.select_action((action,))

    assert selected_action == action


def test_random_agent_rejects_empty_actions() -> None:
    agent = RandomAgent()

    with pytest.raises(
        ValueError,
        match="available_actions cannot be empty.",
    ):
        agent.select_action(())


def test_random_agent_rejects_invalid_action_collection() -> None:
    agent = RandomAgent()

    with pytest.raises(
        TypeError,
        match=(
            "available_actions must contain only "
            "Action instances."
        ),
    ):
        agent.select_action(
            (
                Action(scenario_index=0),
                "invalid",
            )
        )


@pytest.mark.parametrize(
    "invalid_actions",
    [
        None,
        10,
        3.5,
        {"action": 1},
        "actions",
        b"actions",
    ],
)
def test_random_agent_rejects_non_sequence_actions(
    invalid_actions: object,
) -> None:
    agent = RandomAgent()

    with pytest.raises(
        TypeError,
        match=(
            "available_actions must be a sequence "
            "of Action instances."
        ),
    ):
        agent.select_action(
            invalid_actions  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        1.5,
        "42",
        True,
        False,
    ],
)
def test_random_agent_rejects_invalid_seed(
    invalid_seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer or None.",
    ):
        RandomAgent(
            seed=invalid_seed  # type: ignore[arg-type]
        )


def test_random_agent_accepts_integer_seed() -> None:
    agent = RandomAgent(seed=42)

    assert isinstance(agent, RandomAgent)


def test_random_agent_accepts_none_seed() -> None:
    agent = RandomAgent(seed=None)

    assert isinstance(agent, RandomAgent)