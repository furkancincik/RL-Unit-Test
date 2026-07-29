from dataclasses import FrozenInstanceError

import pytest

from rl.action import Action


def test_action_can_be_created_with_zero_index() -> None:
    action = Action(scenario_index=0)

    assert action.scenario_index == 0


def test_action_stores_positive_scenario_index() -> None:
    action = Action(scenario_index=7)

    assert action.scenario_index == 7


def test_actions_with_same_index_are_equal() -> None:
    first_action = Action(scenario_index=3)
    second_action = Action(scenario_index=3)

    assert first_action == second_action


def test_actions_with_different_indexes_are_not_equal() -> None:
    first_action = Action(scenario_index=1)
    second_action = Action(scenario_index=2)

    assert first_action != second_action


def test_action_rejects_negative_scenario_index() -> None:
    with pytest.raises(
        ValueError,
        match="scenario_index cannot be negative.",
    ):
        Action(scenario_index=-1)


@pytest.mark.parametrize(
    "invalid_index",
    [
        1.5,
        "1",
        None,
        True,
        False,
    ],
)
def test_action_rejects_non_integer_scenario_index(
    invalid_index: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="scenario_index must be an integer.",
    ):
        Action(scenario_index=invalid_index)  # type: ignore[arg-type]


def test_action_is_immutable() -> None:
    action = Action(scenario_index=2)

    with pytest.raises(FrozenInstanceError):
        action.scenario_index = 5  # type: ignore[misc]


def test_action_has_readable_representation() -> None:
    action = Action(scenario_index=4)

    assert repr(action) == "Action(scenario_index=4)"


def test_action_can_be_used_in_a_set() -> None:
    actions = {
        Action(scenario_index=1),
        Action(scenario_index=1),
        Action(scenario_index=2),
    }

    assert len(actions) == 2