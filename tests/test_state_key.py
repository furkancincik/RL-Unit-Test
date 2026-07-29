from dataclasses import FrozenInstanceError

import pytest

from rl.state_key import StateKey


def test_state_key_can_be_created() -> None:
    state_key = StateKey(
        coverage_bucket=7,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
    )

    assert state_key.coverage_bucket == 7
    assert state_key.missing_lines_bucket == 2
    assert state_key.uncovered_branches_bucket == 1


def test_state_key_accepts_zero_buckets() -> None:
    state_key = StateKey(
        coverage_bucket=0,
        missing_lines_bucket=0,
        uncovered_branches_bucket=0,
    )

    assert state_key == StateKey(
        coverage_bucket=0,
        missing_lines_bucket=0,
        uncovered_branches_bucket=0,
    )


def test_state_keys_with_same_values_are_equal() -> None:
    first_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=3,
        uncovered_branches_bucket=2,
    )

    second_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=3,
        uncovered_branches_bucket=2,
    )

    assert first_key == second_key


def test_state_keys_with_different_values_are_not_equal() -> None:
    first_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=3,
        uncovered_branches_bucket=2,
    )

    second_key = StateKey(
        coverage_bucket=6,
        missing_lines_bucket=3,
        uncovered_branches_bucket=2,
    )

    assert first_key != second_key


def test_state_key_is_hashable() -> None:
    first_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
    )

    second_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
    )

    state_keys = {
        first_key,
        second_key,
    }

    assert len(state_keys) == 1


@pytest.mark.parametrize(
    "field_name",
    [
        "coverage_bucket",
        "missing_lines_bucket",
        "uncovered_branches_bucket",
    ],
)
def test_state_key_rejects_negative_bucket(
    field_name: str,
) -> None:
    values = {
        "coverage_bucket": 1,
        "missing_lines_bucket": 1,
        "uncovered_branches_bucket": 1,
    }

    values[field_name] = -1

    with pytest.raises(
        ValueError,
        match=rf"{field_name} cannot be negative\.",
    ):
        StateKey(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("coverage_bucket", 1.5),
        ("coverage_bucket", "1"),
        ("coverage_bucket", True),
        ("missing_lines_bucket", None),
        ("missing_lines_bucket", False),
        ("uncovered_branches_bucket", 2.5),
        ("uncovered_branches_bucket", "2"),
    ],
)
def test_state_key_rejects_non_integer_bucket(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "coverage_bucket": 1,
        "missing_lines_bucket": 1,
        "uncovered_branches_bucket": 1,
    }

    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=rf"{field_name} must be an integer\.",
    ):
        StateKey(**values)  # type: ignore[arg-type]


def test_state_key_is_immutable() -> None:
    state_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
    )

    with pytest.raises(FrozenInstanceError):
        state_key.coverage_bucket = 8  # type: ignore[misc]


def test_state_key_has_readable_representation() -> None:
    state_key = StateKey(
        coverage_bucket=5,
        missing_lines_bucket=2,
        uncovered_branches_bucket=1,
    )

    representation = repr(state_key)

    assert "StateKey" in representation
    assert "coverage_bucket=5" in representation
    assert "missing_lines_bucket=2" in representation
    assert "uncovered_branches_bucket=1" in representation