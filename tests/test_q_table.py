import pytest

from rl.action import Action
from rl.q_table import QTable
from rl.state_key import StateKey


def create_state_key(
    coverage_bucket: int = 5,
    missing_lines_bucket: int = 2,
    uncovered_branches_bucket: int = 1,
) -> StateKey:
    return StateKey(
        coverage_bucket=coverage_bucket,
        missing_lines_bucket=missing_lines_bucket,
        uncovered_branches_bucket=uncovered_branches_bucket,
    )


def create_action(
    scenario_index: int = 0,
) -> Action:
    return Action(
        scenario_index=scenario_index,
    )


def test_q_table_can_be_created() -> None:
    q_table = QTable()

    assert len(q_table) == 0


def test_unknown_state_action_value_is_zero() -> None:
    q_table = QTable()

    result = q_table.get_value(
        state_key=create_state_key(),
        action=create_action(),
    )

    assert result == 0.0


def test_q_table_can_store_value() -> None:
    q_table = QTable()

    state_key = create_state_key()
    action = create_action()

    q_table.set_value(
        state_key=state_key,
        action=action,
        value=4.5,
    )

    assert q_table.get_value(state_key, action) == 4.5


def test_q_table_can_update_existing_value() -> None:
    q_table = QTable()

    state_key = create_state_key()
    action = create_action()

    q_table.set_value(state_key, action, 2.0)
    q_table.set_value(state_key, action, 7.5)

    assert q_table.get_value(state_key, action) == 7.5


def test_q_table_stores_integer_value_as_float() -> None:
    q_table = QTable()

    state_key = create_state_key()
    action = create_action()

    q_table.set_value(state_key, action, 3)

    result = q_table.get_value(state_key, action)

    assert result == 3.0
    assert isinstance(result, float)


def test_same_state_can_have_multiple_actions() -> None:
    q_table = QTable()

    state_key = create_state_key()

    first_action = create_action(0)
    second_action = create_action(1)

    q_table.set_value(state_key, first_action, 2.5)
    q_table.set_value(state_key, second_action, 8.0)

    assert q_table.get_value(state_key, first_action) == 2.5
    assert q_table.get_value(state_key, second_action) == 8.0
    assert len(q_table) == 1


def test_different_states_are_stored_separately() -> None:
    q_table = QTable()

    first_state = create_state_key(
        coverage_bucket=5,
    )
    second_state = create_state_key(
        coverage_bucket=6,
    )

    action = create_action()

    q_table.set_value(first_state, action, 1.5)
    q_table.set_value(second_state, action, 9.0)

    assert q_table.get_value(first_state, action) == 1.5
    assert q_table.get_value(second_state, action) == 9.0
    assert len(q_table) == 2


def test_contains_state_returns_false_for_unknown_state() -> None:
    q_table = QTable()

    assert q_table.contains_state(
        create_state_key()
    ) is False


def test_contains_state_returns_true_for_known_state() -> None:
    q_table = QTable()

    state_key = create_state_key()

    q_table.set_value(
        state_key,
        create_action(),
        1.0,
    )

    assert q_table.contains_state(state_key) is True


def test_clear_removes_all_values() -> None:
    q_table = QTable()

    state_key = create_state_key()
    action = create_action()

    q_table.set_value(state_key, action, 4.0)

    q_table.clear()

    assert len(q_table) == 0
    assert q_table.contains_state(state_key) is False
    assert q_table.get_value(state_key, action) == 0.0


@pytest.mark.parametrize(
    "method_name",
    [
        "get_value",
        "set_value",
        "contains_state",
    ],
)
def test_q_table_rejects_invalid_state_key(
    method_name: str,
) -> None:
    q_table = QTable()
    invalid_state = "invalid"
    action = create_action()

    with pytest.raises(
        TypeError,
        match="state_key must be a StateKey instance.",
    ):
        if method_name == "get_value":
            q_table.get_value(
                invalid_state,  # type: ignore[arg-type]
                action,
            )
        elif method_name == "set_value":
            q_table.set_value(
                invalid_state,  # type: ignore[arg-type]
                action,
                1.0,
            )
        else:
            q_table.contains_state(
                invalid_state  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "method_name",
    [
        "get_value",
        "set_value",
    ],
)
def test_q_table_rejects_invalid_action(
    method_name: str,
) -> None:
    q_table = QTable()
    state_key = create_state_key()
    invalid_action = "invalid"

    with pytest.raises(
        TypeError,
        match="action must be an Action instance.",
    ):
        if method_name == "get_value":
            q_table.get_value(
                state_key,
                invalid_action,  # type: ignore[arg-type]
            )
        else:
            q_table.set_value(
                state_key,
                invalid_action,  # type: ignore[arg-type]
                1.0,
            )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "1.0",
        None,
        True,
        False,
    ],
)
def test_q_table_rejects_non_numeric_value(
    invalid_value: object,
) -> None:
    q_table = QTable()

    with pytest.raises(
        TypeError,
        match="value must be a finite number.",
    ):
        q_table.set_value(
            create_state_key(),
            create_action(),
            invalid_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_q_table_rejects_non_finite_value(
    invalid_value: float,
) -> None:
    q_table = QTable()

    with pytest.raises(
        ValueError,
        match="value must be finite.",
    ):
        q_table.set_value(
            create_state_key(),
            create_action(),
            invalid_value,
        )



def test_max_value_returns_zero_for_empty_actions() -> None:
    q_table = QTable()

    result = q_table.max_value(
        state_key=create_state_key(),
        actions=(),
    )

    assert result == 0.0


def test_max_value_returns_zero_for_unknown_actions() -> None:
    q_table = QTable()

    actions = (
        create_action(0),
        create_action(1),
        create_action(2),
    )

    result = q_table.max_value(
        state_key=create_state_key(),
        actions=actions,
    )

    assert result == 0.0


def test_max_value_returns_highest_q_value() -> None:
    q_table = QTable()

    state_key = create_state_key()
    first_action = create_action(0)
    second_action = create_action(1)
    third_action = create_action(2)

    q_table.set_value(state_key, first_action, 2.5)
    q_table.set_value(state_key, second_action, 8.0)
    q_table.set_value(state_key, third_action, -1.0)

    result = q_table.max_value(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
            third_action,
        ),
    )

    assert result == 8.0


def test_max_value_considers_only_given_actions() -> None:
    q_table = QTable()

    state_key = create_state_key()
    first_action = create_action(0)
    second_action = create_action(1)
    excluded_action = create_action(2)

    q_table.set_value(state_key, first_action, 2.0)
    q_table.set_value(state_key, second_action, 4.0)
    q_table.set_value(state_key, excluded_action, 100.0)

    result = q_table.max_value(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
        ),
    )

    assert result == 4.0


def test_best_actions_returns_empty_tuple_for_empty_actions() -> None:
    q_table = QTable()

    result = q_table.best_actions(
        state_key=create_state_key(),
        actions=(),
    )

    assert result == ()


def test_best_actions_returns_all_actions_when_values_are_unknown() -> None:
    q_table = QTable()

    actions = (
        create_action(0),
        create_action(1),
        create_action(2),
    )

    result = q_table.best_actions(
        state_key=create_state_key(),
        actions=actions,
    )

    assert result == actions


def test_best_actions_returns_single_best_action() -> None:
    q_table = QTable()

    state_key = create_state_key()
    first_action = create_action(0)
    second_action = create_action(1)
    third_action = create_action(2)

    q_table.set_value(state_key, first_action, 2.0)
    q_table.set_value(state_key, second_action, 9.0)
    q_table.set_value(state_key, third_action, 4.0)

    result = q_table.best_actions(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
            third_action,
        ),
    )

    assert result == (second_action,)


def test_best_actions_returns_all_tied_best_actions() -> None:
    q_table = QTable()

    state_key = create_state_key()
    first_action = create_action(0)
    second_action = create_action(1)
    third_action = create_action(2)

    q_table.set_value(state_key, first_action, 7.5)
    q_table.set_value(state_key, second_action, 3.0)
    q_table.set_value(state_key, third_action, 7.5)

    result = q_table.best_actions(
        state_key=state_key,
        actions=(
            first_action,
            second_action,
            third_action,
        ),
    )

    assert result == (
        first_action,
        third_action,
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "max_value",
        "best_actions",
    ],
)
def test_q_table_query_methods_reject_invalid_state_key(
    method_name: str,
) -> None:
    q_table = QTable()

    with pytest.raises(
        TypeError,
        match="state_key must be a StateKey instance.",
    ):
        if method_name == "max_value":
            q_table.max_value(
                "invalid",  # type: ignore[arg-type]
                (create_action(),),
            )
        else:
            q_table.best_actions(
                "invalid",  # type: ignore[arg-type]
                (create_action(),),
            )


@pytest.mark.parametrize(
    "method_name",
    [
        "max_value",
        "best_actions",
    ],
)
def test_q_table_query_methods_reject_non_tuple_actions(
    method_name: str,
) -> None:
    q_table = QTable()
    actions = [create_action()]

    with pytest.raises(
        TypeError,
        match="actions must be a tuple of Action instances.",
    ):
        if method_name == "max_value":
            q_table.max_value(
                create_state_key(),
                actions,  # type: ignore[arg-type]
            )
        else:
            q_table.best_actions(
                create_state_key(),
                actions,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "method_name",
    [
        "max_value",
        "best_actions",
    ],
)
def test_q_table_query_methods_reject_invalid_action_item(
    method_name: str,
) -> None:
    q_table = QTable()

    actions = (
        create_action(),
        "invalid",
    )

    with pytest.raises(
        TypeError,
        match="actions must contain only Action instances.",
    ):
        if method_name == "max_value":
            q_table.max_value(
                create_state_key(),
                actions,  # type: ignore[arg-type]
            )
        else:
            q_table.best_actions(
                create_state_key(),
                actions,  # type: ignore[arg-type]
            )