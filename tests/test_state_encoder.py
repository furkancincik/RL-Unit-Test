import pytest

from rl.coverage_state import CoverageState
from rl.state_encoder import StateEncoder
from rl.state_key import StateKey


def create_state(
    coverage_percentage: float,
    missing_lines: tuple[int, ...] = (),
    uncovered_branches: int = 0,
    executed_tests: int = 0,
) -> CoverageState:
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=missing_lines,
        uncovered_branches=uncovered_branches,
    )


def test_state_encoder_can_be_created_with_defaults() -> None:
    encoder = StateEncoder()

    assert encoder.coverage_bucket_size == 10.0
    assert encoder.missing_lines_bucket_size == 5
    assert encoder.uncovered_branches_bucket_size == 5
    assert encoder.executed_tests_bucket_size == 1


def test_encoder_converts_coverage_state_to_state_key() -> None:
    encoder = StateEncoder()

    state = create_state(
        coverage_percentage=72.5,
        missing_lines=(8, 12, 15),
        uncovered_branches=2,
        executed_tests=3,
    )

    result = encoder.encode(state)

    assert result == StateKey(
        coverage_bucket=7,
        missing_lines_bucket=0,
        uncovered_branches_bucket=0,
        executed_tests_bucket=3,
    )


@pytest.mark.parametrize(
    ("coverage_percentage", "expected_bucket"),
    [
        (0.0, 0),
        (9.9, 0),
        (10.0, 1),
        (49.9, 4),
        (50.0, 5),
        (99.9, 9),
        (100.0, 10),
    ],
)
def test_encoder_calculates_coverage_bucket(
    coverage_percentage: float,
    expected_bucket: int,
) -> None:
    encoder = StateEncoder()

    state = create_state(
        coverage_percentage=coverage_percentage,
    )

    result = encoder.encode(state)

    assert result.coverage_bucket == expected_bucket


@pytest.mark.parametrize(
    ("missing_line_count", "expected_bucket"),
    [
        (0, 0),
        (4, 0),
        (5, 1),
        (9, 1),
        (10, 2),
        (14, 2),
        (15, 3),
    ],
)
def test_encoder_calculates_missing_lines_bucket(
    missing_line_count: int,
    expected_bucket: int,
) -> None:
    encoder = StateEncoder()

    missing_lines = tuple(
        range(1, missing_line_count + 1)
    )

    state = create_state(
        coverage_percentage=50.0,
        missing_lines=missing_lines,
    )

    result = encoder.encode(state)

    assert result.missing_lines_bucket == expected_bucket


@pytest.mark.parametrize(
    ("uncovered_branches", "expected_bucket"),
    [
        (0, 0),
        (4, 0),
        (5, 1),
        (9, 1),
        (10, 2),
        (15, 3),
    ],
)
def test_encoder_calculates_uncovered_branches_bucket(
    uncovered_branches: int,
    expected_bucket: int,
) -> None:
    encoder = StateEncoder()

    state = create_state(
        coverage_percentage=50.0,
        uncovered_branches=uncovered_branches,
    )

    result = encoder.encode(state)

    assert result.uncovered_branches_bucket == expected_bucket


@pytest.mark.parametrize(
    ("executed_tests", "expected_bucket"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (5, 5),
        (10, 10),
    ],
)
def test_encoder_calculates_executed_tests_bucket(
    executed_tests: int,
    expected_bucket: int,
) -> None:
    encoder = StateEncoder()

    state = create_state(
        coverage_percentage=50.0,
        executed_tests=executed_tests,
    )

    result = encoder.encode(state)

    assert (
        result.executed_tests_bucket
        == expected_bucket
    )


def test_same_coverage_with_different_test_counts_produces_different_state_key() -> None:
    encoder = StateEncoder()

    first_state = create_state(
        coverage_percentage=80.0,
        missing_lines=(1, 2),
        uncovered_branches=1,
        executed_tests=4,
    )

    second_state = create_state(
        coverage_percentage=80.0,
        missing_lines=(1, 2),
        uncovered_branches=1,
        executed_tests=5,
    )

    assert (
        encoder.encode(first_state)
        != encoder.encode(second_state)
    )


def test_similar_states_can_produce_same_state_key() -> None:
    encoder = StateEncoder(
        executed_tests_bucket_size=2,
    )

    first_state = create_state(
        coverage_percentage=71.0,
        missing_lines=(1, 2, 3),
        uncovered_branches=2,
        executed_tests=2,
    )

    second_state = create_state(
        coverage_percentage=79.9,
        missing_lines=(4,),
        uncovered_branches=4,
        executed_tests=3,
    )

    assert (
        encoder.encode(first_state)
        == encoder.encode(second_state)
    )


def test_custom_bucket_sizes_are_used() -> None:
    encoder = StateEncoder(
        coverage_bucket_size=5.0,
        missing_lines_bucket_size=2,
        uncovered_branches_bucket_size=3,
        executed_tests_bucket_size=2,
    )

    state = create_state(
        coverage_percentage=72.5,
        missing_lines=(1, 2, 3, 4, 5),
        uncovered_branches=7,
        executed_tests=5,
    )

    result = encoder.encode(state)

    assert result == StateKey(
        coverage_bucket=14,
        missing_lines_bucket=2,
        uncovered_branches_bucket=2,
        executed_tests_bucket=2,
    )


def test_encoder_rejects_invalid_state() -> None:
    encoder = StateEncoder()

    with pytest.raises(
        TypeError,
        match="state must be a CoverageState instance.",
    ):
        encoder.encode("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("coverage_bucket_size", 0),
        ("coverage_bucket_size", -1.0),
        ("missing_lines_bucket_size", 0),
        ("missing_lines_bucket_size", -1),
        ("uncovered_branches_bucket_size", 0),
        ("uncovered_branches_bucket_size", -1),
        ("executed_tests_bucket_size", 0),
        ("executed_tests_bucket_size", -1),
    ],
)
def test_encoder_rejects_non_positive_bucket_size(
    field_name: str,
    invalid_value: float | int,
) -> None:
    values = {
        "coverage_bucket_size": 10.0,
        "missing_lines_bucket_size": 5,
        "uncovered_branches_bucket_size": 5,
        "executed_tests_bucket_size": 1,
    }

    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be greater than zero\.",
    ):
        StateEncoder(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("coverage_bucket_size", "10"),
        ("coverage_bucket_size", True),
        ("missing_lines_bucket_size", 2.5),
        ("missing_lines_bucket_size", False),
        ("uncovered_branches_bucket_size", "5"),
        ("uncovered_branches_bucket_size", None),
        ("executed_tests_bucket_size", 1.5),
        ("executed_tests_bucket_size", "1"),
        ("executed_tests_bucket_size", False),
        ("executed_tests_bucket_size", None),
    ],
)
def test_encoder_rejects_invalid_bucket_type(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "coverage_bucket_size": 10.0,
        "missing_lines_bucket_size": 5,
        "uncovered_branches_bucket_size": 5,
        "executed_tests_bucket_size": 1,
    }

    values[field_name] = invalid_value

    expected_message = (
        f"{field_name} must be a number."
        if field_name == "coverage_bucket_size"
        else f"{field_name} must be an integer."
    )

    with pytest.raises(
        TypeError,
        match=rf"{expected_message}",
    ):
        StateEncoder(**values)  # type: ignore[arg-type]


def test_state_encoder_is_immutable() -> None:
    encoder = StateEncoder()

    with pytest.raises(AttributeError):
        encoder.coverage_bucket_size = 5.0  # type: ignore[misc]
