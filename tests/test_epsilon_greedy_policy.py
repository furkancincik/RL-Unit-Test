import math
import random

import pytest

from rl.action import Action
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_table import QTable
from rl.state_key import StateKey


class ControlledRandom(random.Random):
    """
    Testlerde rastgele davranışı kontrol etmek için kullanılır.
    """

    def __init__(
        self,
        random_value: float,
        choice_index: int = 0,
    ) -> None:
        super().__init__()

        self._random_value = random_value
        self._choice_index = choice_index

    def random(self) -> float:
        return self._random_value

    def choice(self, sequence):  # type: ignore[no-untyped-def]
        return sequence[self._choice_index]


def create_state_key() -> StateKey:
    return StateKey(
        coverage_bucket=5,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
        executed_tests_bucket=3,
    )


def create_action(
    scenario_index: int = 0,
) -> Action:
    return Action(
        scenario_index=scenario_index,
    )


def test_policy_can_be_created() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.25,
    )

    assert policy.epsilon == 0.25


def test_policy_converts_integer_epsilon_to_float() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=1,
    )

    assert policy.epsilon == 1.0
    assert isinstance(policy.epsilon, float)


@pytest.mark.parametrize(
    "epsilon",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_policy_accepts_valid_epsilon_values(
    epsilon: float,
) -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=epsilon,
    )

    assert policy.epsilon == epsilon


@pytest.mark.parametrize(
    "epsilon",
    [
        -0.01,
        -1.0,
        1.01,
        2.0,
    ],
)
def test_policy_rejects_epsilon_outside_valid_range(
    epsilon: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="epsilon must be between 0.0 and 1.0.",
    ):
        EpsilonGreedyPolicy(
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_policy_rejects_non_finite_epsilon(
    epsilon: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="epsilon must be a finite value.",
    ):
        EpsilonGreedyPolicy(
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        True,
        False,
        "0.5",
        None,
    ],
)
def test_policy_rejects_invalid_epsilon_type(
    epsilon: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="epsilon must be a numeric value.",
    ):
        EpsilonGreedyPolicy(
            epsilon=epsilon,  # type: ignore[arg-type]
        )


def test_policy_accepts_random_generator() -> None:
    random_generator = random.Random(42)

    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
        random_generator=random_generator,
    )

    assert policy.epsilon == 0.5


def test_policy_rejects_invalid_random_generator() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "random_generator must be "
            "a random.Random instance."
        ),
    ):
        EpsilonGreedyPolicy(
            epsilon=0.5,
            random_generator="invalid",  # type: ignore[arg-type]
        )


def test_select_action_explores_when_random_value_is_below_epsilon(
) -> None:
    random_generator = ControlledRandom(
        random_value=0.10,
        choice_index=1,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        100.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        0.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert result == second_action


def test_select_action_exploits_when_random_value_is_equal_to_epsilon(
) -> None:
    random_generator = ControlledRandom(
        random_value=0.20,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        2.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        8.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert result == second_action


def test_select_action_exploits_when_random_value_is_above_epsilon(
) -> None:
    random_generator = ControlledRandom(
        random_value=0.80,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        3.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        9.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert result == second_action


def test_select_action_with_zero_epsilon_always_exploits() -> None:
    random_generator = ControlledRandom(
        random_value=0.0,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.0,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        1.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        10.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert result == second_action


def test_select_action_with_one_epsilon_always_explores() -> None:
    random_generator = ControlledRandom(
        random_value=0.99,
        choice_index=0,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=1.0,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        0.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        100.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert result == first_action


def test_select_action_chooses_between_tied_best_actions() -> None:
    random_generator = ControlledRandom(
        random_value=0.90,
        choice_index=1,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.10,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)
    third_action = create_action(2)

    q_table.set_value(
        state_key,
        first_action,
        7.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        2.0,
    )

    q_table.set_value(
        state_key,
        third_action,
        7.0,
    )

    result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
            third_action,
        ),
        q_table=q_table,
    )

    assert result == third_action


def test_select_action_returns_action_when_q_values_are_unknown(
) -> None:
    random_generator = ControlledRandom(
        random_value=0.90,
        choice_index=1,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.10,
        random_generator=random_generator,
    )

    actions = (
        create_action(0),
        create_action(1),
        create_action(2),
    )

    result = policy.select_action(
        state_key=create_state_key(),
        actions=actions,
        q_table=QTable(),
    )

    assert result == actions[1]


def test_select_action_rejects_empty_actions() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "actions must contain at least "
            "one Action instance."
        ),
    ):
        policy.select_action(
            state_key=create_state_key(),
            actions=(),
            q_table=QTable(),
        )


def test_select_action_rejects_invalid_state_key() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
    )

    with pytest.raises(
        TypeError,
        match="state_key must be a StateKey instance.",
    ):
        policy.select_action(
            state_key="invalid",  # type: ignore[arg-type]
            actions=(create_action(),),
            q_table=QTable(),
        )


def test_select_action_rejects_non_tuple_actions() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
    )

    with pytest.raises(
        TypeError,
        match="actions must be a tuple of Action instances.",
    ):
        policy.select_action(
            state_key=create_state_key(),
            actions=[create_action()],  # type: ignore[arg-type]
            q_table=QTable(),
        )


def test_select_action_rejects_invalid_action_item() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
    )

    with pytest.raises(
        TypeError,
        match="actions must contain only Action instances.",
    ):
        policy.select_action(
            state_key=create_state_key(),
            actions=(
                create_action(),
                "invalid",  # type: ignore[arg-type]
            ),
            q_table=QTable(),
        )


def test_select_action_rejects_invalid_q_table() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.5,
    )

    with pytest.raises(
        TypeError,
        match="q_table must be a QTable instance.",
    ):
        policy.select_action(
            state_key=create_state_key(),
            actions=(create_action(),),
            q_table="invalid",  # type: ignore[arg-type]
        )


def test_set_epsilon_updates_policy_value() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    result = policy.set_epsilon(
        0.20
    )

    assert result == pytest.approx(0.20)
    assert policy.epsilon == pytest.approx(0.20)


@pytest.mark.parametrize(
    "epsilon",
    [
        -0.1,
        1.1,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_set_epsilon_rejects_invalid_value(
    epsilon: float,
) -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    with pytest.raises(
        ValueError,
    ):
        policy.set_epsilon(
            epsilon
        )

    assert policy.epsilon == pytest.approx(0.30)


@pytest.mark.parametrize(
    "epsilon",
    [
        True,
        "0.2",
        None,
    ],
)
def test_set_epsilon_rejects_invalid_type(
    epsilon: object,
) -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    with pytest.raises(
        TypeError,
        match="epsilon must be a numeric value.",
    ):
        policy.set_epsilon(
            epsilon  # type: ignore[arg-type]
        )

    assert policy.epsilon == pytest.approx(0.30)


def test_decay_epsilon_reduces_current_value() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    result = policy.decay_epsilon(
        decay_rate=0.5,
        minimum_epsilon=0.05,
    )

    assert result == pytest.approx(0.15)
    assert policy.epsilon == pytest.approx(0.15)


def test_decay_epsilon_respects_minimum_value() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.10,
    )

    first_result = policy.decay_epsilon(
        decay_rate=0.50,
        minimum_epsilon=0.05,
    )

    second_result = policy.decay_epsilon(
        decay_rate=0.50,
        minimum_epsilon=0.05,
    )

    assert first_result == pytest.approx(0.05)
    assert second_result == pytest.approx(0.05)
    assert policy.epsilon == pytest.approx(0.05)


def test_decay_epsilon_allows_zero_minimum() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
    )

    result = policy.decay_epsilon(
        decay_rate=0.0,
        minimum_epsilon=0.0,
    )

    assert result == pytest.approx(0.0)
    assert policy.epsilon == pytest.approx(0.0)


def test_decay_epsilon_rejects_minimum_above_current_epsilon() -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
    )

    with pytest.raises(
        ValueError,
        match=(
            "minimum_epsilon cannot be greater "
            "than current epsilon."
        ),
    ):
        policy.decay_epsilon(
            decay_rate=0.90,
            minimum_epsilon=0.30,
        )

    assert policy.epsilon == pytest.approx(0.20)


@pytest.mark.parametrize(
    "decay_rate",
    [
        -0.1,
        1.1,
        math.nan,
        math.inf,
    ],
)
def test_decay_epsilon_rejects_invalid_decay_rate(
    decay_rate: float,
) -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    with pytest.raises(
        ValueError,
    ):
        policy.decay_epsilon(
            decay_rate=decay_rate,
            minimum_epsilon=0.05,
        )

    assert policy.epsilon == pytest.approx(0.30)


@pytest.mark.parametrize(
    "minimum_epsilon",
    [
        -0.1,
        1.1,
        math.nan,
        math.inf,
    ],
)
def test_decay_epsilon_rejects_invalid_minimum(
    minimum_epsilon: float,
) -> None:
    policy = EpsilonGreedyPolicy(
        epsilon=0.30,
    )

    with pytest.raises(
        ValueError,
    ):
        policy.decay_epsilon(
            decay_rate=0.90,
            minimum_epsilon=minimum_epsilon,
        )

    assert policy.epsilon == pytest.approx(0.30)


def test_updated_epsilon_changes_exploration_decision() -> None:
    random_generator = ControlledRandom(
        random_value=0.15,
        choice_index=1,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.20,
        random_generator=random_generator,
    )

    q_table = QTable()
    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(
        state_key,
        first_action,
        10.0,
    )

    q_table.set_value(
        state_key,
        second_action,
        0.0,
    )

    exploring_result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert exploring_result == second_action

    policy.set_epsilon(
        0.10
    )

    random_generator._choice_index = 0

    exploiting_result = policy.select_action(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
        q_table=q_table,
    )

    assert exploiting_result == first_action


def test_policy_counts_exploration_and_exploitation_without_changing_selection() -> None:
    random_generator = ControlledRandom(random_value=0.1)
    policy = EpsilonGreedyPolicy(
        epsilon=0.2,
        random_generator=random_generator,
    )
    actions = (create_action(0), create_action(1))

    assert policy.exploration_selection_count == 0
    assert policy.exploitation_selection_count == 0
    assert policy.select_action(create_state_key(), actions, QTable()) == actions[0]
    random_generator._random_value = 0.9
    assert policy.select_action(create_state_key(), actions, QTable()) == actions[0]

    assert policy.exploration_selection_count == 1
    assert policy.exploitation_selection_count == 1

