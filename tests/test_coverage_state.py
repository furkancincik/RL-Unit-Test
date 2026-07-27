from __future__ import annotations

import pytest

from rl.coverage_state import CoverageState


def test_coverage_state_stores_coverage_information() -> None:
    state = CoverageState(
        coverage_percentage=82.5,
        executed_tests=8,
        missing_lines=(12, 18, 27),
        uncovered_branches=5,
    )

    assert state.coverage_percentage == 82.5
    assert state.executed_tests == 8
    assert state.missing_lines == (12, 18, 27)
    assert state.uncovered_branches == 5


def test_coverage_state_is_immutable() -> None:
    state = CoverageState(
        coverage_percentage=75.0,
        executed_tests=4,
        missing_lines=(10,),
        uncovered_branches=2,
    )

    with pytest.raises(AttributeError):
        state.coverage_percentage = 90.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "coverage_percentage",
    [
        -1.0,
        100.1,
    ],
)
def test_coverage_state_rejects_invalid_coverage_percentage(
    coverage_percentage: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Coverage percentage must be between 0 and 100.",
    ):
        CoverageState(
            coverage_percentage=coverage_percentage,
            executed_tests=1,
            missing_lines=(),
            uncovered_branches=0,
        )


@pytest.mark.parametrize(
    ("executed_tests", "uncovered_branches"),
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_coverage_state_rejects_negative_counts(
    executed_tests: int,
    uncovered_branches: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Counts cannot be negative.",
    ):
        CoverageState(
            coverage_percentage=50.0,
            executed_tests=executed_tests,
            missing_lines=(),
            uncovered_branches=uncovered_branches,
        )


def test_coverage_state_rejects_non_positive_missing_lines() -> None:
    with pytest.raises(
        ValueError,
        match="Missing line numbers must be positive.",
    ):
        CoverageState(
            coverage_percentage=50.0,
            executed_tests=1,
            missing_lines=(0, 5),
            uncovered_branches=1,
        )


def test_is_fully_covered_returns_true_for_complete_coverage() -> None:
    state = CoverageState(
        coverage_percentage=100.0,
        executed_tests=10,
        missing_lines=(),
        uncovered_branches=0,
    )

    assert state.is_fully_covered is True


def test_is_fully_covered_returns_false_when_lines_are_missing() -> None:
    state = CoverageState(
        coverage_percentage=100.0,
        executed_tests=10,
        missing_lines=(15,),
        uncovered_branches=0,
    )

    assert state.is_fully_covered is False


def test_missing_line_count_returns_number_of_missing_lines() -> None:
    state = CoverageState(
        coverage_percentage=80.0,
        executed_tests=5,
        missing_lines=(10, 15, 20),
        uncovered_branches=2,
    )

    assert state.missing_line_count == 3