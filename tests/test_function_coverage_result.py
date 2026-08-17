from __future__ import annotations

from pathlib import Path

import pytest

from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)


def create_file_coverage() -> CoverageResult:
    """Testlerde kullanılacak dosya bazlı coverage sonucu oluşturur."""
    return CoverageResult(
        source_file=Path("datasets/sample_complex_code.py"),
        test_file=Path(
            "output/complex_tests/"
            "test_calculate_score_scenario_suite.py"
        ),
        line_coverage_percent=33.33,
        branch_coverage_percent=40.0,
        covered_line_count=25,
        missing_line_count=50,
        total_line_count=75,
        covered_branch_count=8,
        missing_branch_count=12,
        total_branch_count=20,
        test_exit_code=0,
        duration_seconds=0.25,
    )


def create_function_coverage(
    **overrides: object,
) -> FunctionCoverageResult:
    """Geçerli bir FunctionCoverageResult nesnesi oluşturur."""
    values: dict[str, object] = {
        "source_file": Path(
            "datasets/sample_complex_code.py"
        ),
        "test_file": Path(
            "output/complex_tests/"
            "test_calculate_score_scenario_suite.py"
        ),
        "function_name": "calculate_score",
        "start_line": 4,
        "end_line": 38,
        "line_coverage_percent": 100.0,
        "branch_coverage_percent": 100.0,
        "covered_lines": (
            4,
            9,
            10,
            12,
            13,
            15,
            16,
            18,
            19,
            21,
            22,
            24,
            25,
            27,
            28,
            30,
            31,
            33,
            34,
            36,
            38,
        ),
        "missing_lines": (),
        "covered_branch_count": 18,
        "missing_branch_count": 0,
        "test_exit_code": 0,
        "duration_seconds": 0.25,
        "file_coverage": create_file_coverage(),
    }

    values.update(overrides)

    return FunctionCoverageResult(
        **values,  # type: ignore[arg-type]
    )


def test_function_coverage_result_stores_values() -> None:
    result = create_function_coverage()

    assert result.function_name == "calculate_score"
    assert result.start_line == 4
    assert result.end_line == 38
    assert result.line_coverage_percent == 100.0
    assert result.branch_coverage_percent == 100.0


def test_function_coverage_result_calculates_line_counts() -> None:
    result = create_function_coverage(
        covered_lines=(4, 5, 6),
        missing_lines=(7, 8),
    )

    assert result.covered_line_count == 3
    assert result.missing_line_count == 2
    assert result.total_line_count == 5


def test_function_coverage_result_calculates_branch_count() -> None:
    result = create_function_coverage(
        covered_branch_count=6,
        missing_branch_count=2,
    )

    assert result.total_branch_count == 8


def test_function_coverage_result_preserves_legacy_branch_counts() -> None:
    result = create_function_coverage(
        covered_branch_count=6,
        missing_branch_count=2,
    )

    assert result.has_branch_details is False
    assert result.covered_branches is None
    assert result.missing_branches is None


def test_function_coverage_result_stores_branch_details() -> None:
    result = create_function_coverage(
        covered_branch_count=2,
        missing_branch_count=1,
        covered_branches=(
            (10, 12),
            (12, 13),
        ),
        missing_branches=(
            (10, 15),
        ),
    )

    assert result.has_branch_details is True
    assert result.covered_branches == (
        (10, 12),
        (12, 13),
    )
    assert result.missing_branches == (
        (10, 15),
    )


def test_function_coverage_result_accepts_special_branch_target() -> None:
    result = create_function_coverage(
        covered_branch_count=1,
        missing_branch_count=1,
        covered_branches=(
            (10, -1),
        ),
        missing_branches=(
            (10, 12),
        ),
    )

    assert result.has_branch_details is True
    assert result.covered_branches == (
        (10, -1),
    )


def test_function_coverage_result_success() -> None:
    result = create_function_coverage(
        test_exit_code=0,
    )

    assert result.success is True


def test_function_coverage_result_not_successful() -> None:
    result = create_function_coverage(
        test_exit_code=1,
    )

    assert result.success is False


def test_function_coverage_result_detects_full_coverage() -> None:
    result = create_function_coverage()

    assert result.has_full_line_coverage is True
    assert result.has_full_branch_coverage is True
    assert result.has_full_coverage is True


def test_function_coverage_result_detects_incomplete_line_coverage() -> None:
    result = create_function_coverage(
        line_coverage_percent=80.0,
        covered_lines=(4, 5, 6, 7),
        missing_lines=(8,),
    )

    assert result.has_full_line_coverage is False
    assert result.has_full_coverage is False


def test_function_coverage_result_detects_incomplete_branch_coverage() -> None:
    result = create_function_coverage(
        branch_coverage_percent=75.0,
        covered_branch_count=3,
        missing_branch_count=1,
    )

    assert result.has_full_branch_coverage is False
    assert result.has_full_coverage is False


def test_function_coverage_result_exposes_file_percentages() -> None:
    result = create_function_coverage()

    assert result.file_line_coverage_percent == 33.33
    assert result.file_branch_coverage_percent == 40.0


def test_function_coverage_result_rejects_invalid_function_name() -> None:
    with pytest.raises(
        ValueError,
        match="function_name boş olamaz",
    ):
        create_function_coverage(
            function_name="",
        )


def test_function_coverage_result_rejects_invalid_line_range() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Fonksiyon bitiş satırı başlangıç "
            "satırından küçük olamaz"
        ),
    ):
        create_function_coverage(
            start_line=20,
            end_line=10,
            covered_lines=(),
            missing_lines=(),
        )


def test_function_coverage_result_rejects_line_outside_range() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "covered_lines yalnızca fonksiyon satır "
            "aralığındaki değerleri içermelidir"
        ),
    ):
        create_function_coverage(
            covered_lines=(3, 4, 5),
        )


def test_function_coverage_result_rejects_duplicate_lines() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "covered_lines tekrar eden satır "
            "numarası içeremez"
        ),
    ):
        create_function_coverage(
            covered_lines=(4, 5, 5),
        )


def test_function_coverage_result_rejects_unsorted_lines() -> None:
    with pytest.raises(
        ValueError,
        match="covered_lines artan sırada olmalıdır",
    ):
        create_function_coverage(
            covered_lines=(6, 4, 5),
        )


def test_function_coverage_result_rejects_overlapping_lines() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Çalıştırılan ve eksik satırlar "
            "kesişemez"
        ),
    ):
        create_function_coverage(
            covered_lines=(4, 5, 6),
            missing_lines=(6, 7),
        )


def test_function_coverage_result_requires_both_branch_details() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "covered_branches ve missing_branches "
            "birlikte verilmelidir"
        ),
    ):
        create_function_coverage(
            covered_branch_count=1,
            missing_branch_count=0,
            covered_branches=((10, 12),),
        )


def test_function_coverage_result_rejects_invalid_branch_record() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "covered_branches yalnızca iki elemanlı "
            "tuple değerler içermelidir"
        ),
    ):
        create_function_coverage(
            covered_branch_count=1,
            missing_branch_count=0,
            covered_branches=((10, 12, 13),),
            missing_branches=(),
        )


def test_function_coverage_result_rejects_branch_source_outside_range() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "covered_branches branch kaynak satırları "
            "fonksiyon aralığında olmalıdır"
        ),
    ):
        create_function_coverage(
            covered_branch_count=1,
            missing_branch_count=0,
            covered_branches=((3, 10),),
            missing_branches=(),
        )


def test_function_coverage_result_rejects_duplicate_branches() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "covered_branches tekrar eden branch "
            "içeremez"
        ),
    ):
        create_function_coverage(
            covered_branch_count=2,
            missing_branch_count=0,
            covered_branches=(
                (10, 12),
                (10, 12),
            ),
            missing_branches=(),
        )


def test_function_coverage_result_rejects_unsorted_branches() -> None:
    with pytest.raises(
        ValueError,
        match="covered_branches artan sırada olmalıdır",
    ):
        create_function_coverage(
            covered_branch_count=2,
            missing_branch_count=0,
            covered_branches=(
                (12, 13),
                (10, 12),
            ),
            missing_branches=(),
        )


def test_function_coverage_result_rejects_overlapping_branches() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Çalıştırılan ve eksik branchler "
            "kesişemez"
        ),
    ):
        create_function_coverage(
            covered_branch_count=1,
            missing_branch_count=1,
            covered_branches=((10, 12),),
            missing_branches=((10, 12),),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {
                "covered_branch_count": 2,
                "missing_branch_count": 0,
                "covered_branches": ((10, 12),),
                "missing_branches": (),
            },
            (
                "covered_branches uzunluğu, çalıştırılan "
                "branch sayısına eşit olmalıdır"
            ),
        ),
        (
            {
                "covered_branch_count": 0,
                "missing_branch_count": 2,
                "covered_branches": (),
                "missing_branches": ((10, 12),),
            },
            (
                "missing_branches uzunluğu, eksik branch "
                "sayısına eşit olmalıdır"
            ),
        ),
    ),
)
def test_function_coverage_result_rejects_branch_count_mismatch(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        create_function_coverage(
            **overrides,
        )


def test_function_coverage_result_rejects_invalid_file_coverage() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "file_coverage bir CoverageResult "
            "örneği olmalıdır"
        ),
    ):
        create_function_coverage(
            file_coverage="invalid",
        )


def test_function_coverage_result_rejects_different_source_file() -> None:
    file_coverage = create_file_coverage()

    different_file_coverage = CoverageResult(
        source_file=Path("datasets/other.py"),
        test_file=file_coverage.test_file,
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=5,
        missing_line_count=5,
        total_line_count=10,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
        test_exit_code=0,
        duration_seconds=0.1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Fonksiyon coverage ve dosya coverage "
            "kaynak dosyaları eşleşmelidir"
        ),
    ):
        create_function_coverage(
            file_coverage=different_file_coverage,
        )


def test_function_coverage_result_rejects_different_test_file() -> None:
    file_coverage = create_file_coverage()

    different_file_coverage = CoverageResult(
        source_file=file_coverage.source_file,
        test_file=Path("output/other_test.py"),
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=5,
        missing_line_count=5,
        total_line_count=10,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
        test_exit_code=0,
        duration_seconds=0.1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Fonksiyon coverage ve dosya coverage "
            "test dosyaları eşleşmelidir"
        ),
    ):
        create_function_coverage(
            file_coverage=different_file_coverage,
        )
