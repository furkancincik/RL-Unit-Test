from __future__ import annotations

from pathlib import Path

import pytest

from models.coverage_result import CoverageResult
from rl.coverage_state import CoverageState
from rl.coverage_state_mapper import CoverageStateMapper


def create_coverage_result(
    *,
    line_coverage_percent: float = 50.0,
    branch_coverage_percent: float = 50.0,
    covered_line_count: int = 3,
    missing_line_count: int = 3,
    total_line_count: int = 6,
    covered_branch_count: int = 2,
    missing_branch_count: int = 2,
    total_branch_count: int = 4,
    test_exit_code: int = 0,
) -> CoverageResult:
    """CoverageStateMapper testleri için CoverageResult oluşturur."""
    return CoverageResult(
        source_file=Path("datasets/sample_code.py"),
        test_file=Path(
            "output/generated_tests/test_calculate_score.py"
        ),
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=test_exit_code,
        duration_seconds=0.25,
    )


def test_map_returns_coverage_state() -> None:
    result = CoverageStateMapper.map(
        coverage_result=create_coverage_result(),
        executed_tests=1,
    )

    assert isinstance(result, CoverageState)


def test_map_preserves_line_coverage_percentage() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=50.0,
    )

    state = CoverageStateMapper.map(
        coverage_result=coverage_result,
        executed_tests=1,
    )

    assert state.coverage_percentage == 50.0


def test_map_preserves_executed_test_count() -> None:
    state = CoverageStateMapper.map(
        coverage_result=create_coverage_result(),
        executed_tests=4,
    )

    assert state.executed_tests == 4


def test_map_preserves_missing_branch_count() -> None:
    coverage_result = create_coverage_result(
        missing_branch_count=2,
    )

    state = CoverageStateMapper.map(
        coverage_result=coverage_result,
        executed_tests=1,
    )

    assert state.uncovered_branches == 2


def test_map_creates_representative_missing_lines() -> None:
    coverage_result = create_coverage_result(
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
    )

    state = CoverageStateMapper.map(
        coverage_result=coverage_result,
        executed_tests=1,
    )

    assert state.missing_lines == (1, 2, 3)
    assert state.missing_line_count == 3


def test_map_creates_empty_missing_lines_for_full_coverage() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=6,
        missing_line_count=0,
        total_line_count=6,
        covered_branch_count=4,
        missing_branch_count=0,
        total_branch_count=4,
    )

    state = CoverageStateMapper.map(
        coverage_result=coverage_result,
        executed_tests=3,
    )

    assert state.missing_lines == ()
    assert state.missing_line_count == 0
    assert state.uncovered_branches == 0
    assert state.is_fully_covered is True


def test_map_uses_supplied_real_missing_lines() -> None:
    coverage_result = create_coverage_result(
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
    )

    state = CoverageStateMapper.map(
        coverage_result=coverage_result,
        executed_tests=1,
        missing_lines=(5, 8, 12),
    )

    assert state.missing_lines == (5, 8, 12)


def test_map_rejects_invalid_coverage_result() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "coverage_result bir CoverageResult "
            "örneği olmalıdır"
        ),
    ):
        CoverageStateMapper.map(
            coverage_result="invalid",  # type: ignore[arg-type]
            executed_tests=1,
        )


def test_map_rejects_non_integer_executed_tests() -> None:
    with pytest.raises(
        TypeError,
        match="executed_tests bir tam sayı olmalıdır",
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1.5,  # type: ignore[arg-type]
        )


def test_map_rejects_boolean_executed_tests() -> None:
    with pytest.raises(
        TypeError,
        match="executed_tests bir tam sayı olmalıdır",
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=True,  # type: ignore[arg-type]
        )


def test_map_rejects_negative_executed_tests() -> None:
    with pytest.raises(
        ValueError,
        match="executed_tests negatif olamaz",
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=-1,
        )


def test_map_rejects_non_tuple_missing_lines() -> None:
    with pytest.raises(
        TypeError,
        match="missing_lines bir tuple veya None olmalıdır",
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1,
            missing_lines=[1, 2, 3],  # type: ignore[arg-type]
        )


def test_map_rejects_non_integer_missing_line() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "missing_lines yalnızca tam sayı "
            "değerler içermelidir"
        ),
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1,
            missing_lines=(1, "2", 3),  # type: ignore[arg-type]
        )


def test_map_rejects_non_positive_missing_line() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Eksik satır numaraları 1 veya "
            "daha büyük olmalıdır"
        ),
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1,
            missing_lines=(0, 2, 3),
        )


def test_map_rejects_duplicate_missing_lines() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "missing_lines tekrar eden satır "
            "numarası içeremez"
        ),
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1,
            missing_lines=(2, 2, 3),
        )


def test_map_rejects_missing_line_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Verilen eksik satır sayısı "
            "CoverageResult ile uyuşmuyor"
        ),
    ):
        CoverageStateMapper.map(
            coverage_result=create_coverage_result(),
            executed_tests=1,
            missing_lines=(5, 8),
        )