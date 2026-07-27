from __future__ import annotations

from pathlib import Path

from models.coverage_result import CoverageResult
from rl.coverage_state_mapper import CoverageStateMapper


def create_coverage_result(
    *,
    line_coverage_percent: float,
    branch_coverage_percent: float,
    covered_line_count: int,
    missing_line_count: int,
    total_line_count: int,
    covered_branch_count: int,
    missing_branch_count: int,
    total_branch_count: int,
) -> CoverageResult:
    """Testlerde kullanılacak geçerli CoverageResult nesnesini oluşturur."""

    return CoverageResult(
        source_file=Path("sample.py"),
        test_file=Path("test_sample.py"),
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=0,
        duration_seconds=0.25,
    )


def test_map_creates_coverage_state_from_coverage_result() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=82.5,
        branch_coverage_percent=75.0,
        covered_line_count=33,
        missing_line_count=7,
        total_line_count=40,
        covered_branch_count=6,
        missing_branch_count=2,
        total_branch_count=8,
    )

    state = CoverageStateMapper().map(
        coverage_result=coverage_result,
        executed_tests=8,
        missing_lines=(12, 18, 27),
    )

    assert state.coverage_percentage == 82.5
    assert state.executed_tests == 8
    assert state.missing_lines == (12, 18, 27)
    assert state.uncovered_branches == 2


def test_map_creates_fully_covered_state() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=40,
        missing_line_count=0,
        total_line_count=40,
        covered_branch_count=8,
        missing_branch_count=0,
        total_branch_count=8,
    )

    state = CoverageStateMapper().map(
        coverage_result=coverage_result,
        executed_tests=10,
    )

    assert state.is_fully_covered is True


def test_map_uses_missing_branch_count_as_uncovered_branches() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=90.0,
        branch_coverage_percent=60.0,
        covered_line_count=36,
        missing_line_count=4,
        total_line_count=40,
        covered_branch_count=3,
        missing_branch_count=2,
        total_branch_count=5,
    )

    state = CoverageStateMapper().map(
        coverage_result=coverage_result,
        executed_tests=5,
    )

    assert state.uncovered_branches == 2