from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models.coverage_result import FunctionCoverageResult
from services.coverage_service import CoverageService


def create_source_file(
    tmp_path: Path,
) -> Path:
    source_file = tmp_path / "sample.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 50:\n"
        "        return 'Başarılı'\n"
        "    return 'Başarısız'\n"
        "\n"
        "\n"
        "def other_function():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    return source_file


def create_test_file(
    tmp_path: Path,
) -> Path:
    test_file = tmp_path / "test_sample.py"

    test_file.write_text(
        "def test_placeholder():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    return test_file


def create_report_data(
    source_file: Path,
) -> dict[str, object]:
    return {
        "files": {
            str(source_file.resolve()): {
                "executed_lines": [
                    1,
                    2,
                    3,
                    7,
                ],
                "missing_lines": [
                    4,
                    8,
                ],
                "executed_branches": [
                    [2, 3],
                ],
                "missing_branches": [
                    [2, 4],
                ],
                "summary": {
                    "covered_lines": 4,
                    "num_statements": 6,
                    "percent_covered": 66.67,
                    "percent_covered_display": "67",
                    "missing_lines": 2,
                    "excluded_lines": 0,
                    "covered_branches": 1,
                    "num_branches": 2,
                    "partial_branches": 1,
                    "missing_branches": 1,
                },
            }
        },
        "totals": {
            "covered_lines": 4,
            "num_statements": 6,
            "percent_covered": 66.67,
            "missing_lines": 2,
            "excluded_lines": 0,
            "covered_branches": 1,
            "num_branches": 2,
            "partial_branches": 1,
            "missing_branches": 1,
        },
    }


def create_completed_process(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Mock:
    process = Mock()

    process.returncode = returncode
    process.stdout = stdout
    process.stderr = stderr

    return process


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_returns_function_coverage_result(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    report_data = create_report_data(
        source_file
    )

    original_read_report = CoverageService._read_report

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=report_data,
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert isinstance(
        result,
        FunctionCoverageResult,
    )


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_filters_lines_by_function_range(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=create_report_data(
            source_file
        ),
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert result.covered_lines == (
        1,
        2,
        3,
    )

    assert result.missing_lines == (
        4,
    )


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_calculates_line_coverage(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=create_report_data(
            source_file
        ),
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert result.covered_line_count == 3
    assert result.missing_line_count == 1
    assert result.total_line_count == 4
    assert result.line_coverage_percent == 75.0


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_filters_branches_by_source_line(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=create_report_data(
            source_file
        ),
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert result.covered_branch_count == 1
    assert result.missing_branch_count == 1
    assert result.total_branch_count == 2
    assert result.branch_coverage_percent == 50.0


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_preserves_file_coverage(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=create_report_data(
            source_file
        ),
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert result.file_coverage.line_coverage_percent == 66.67
    assert result.file_coverage.branch_coverage_percent == 50.0


@patch(
    "services.coverage_service.subprocess.run",
)
def test_measure_function_ignores_other_function_lines(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    test_file = create_test_file(tmp_path)

    mock_run.side_effect = [
        create_completed_process(),
        create_completed_process(),
    ]

    with patch.object(
        CoverageService,
        "_read_report",
        return_value=create_report_data(
            source_file
        ),
    ):
        result = CoverageService().measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name="calculate_score",
            start_line=1,
            end_line=4,
        )

    assert 7 not in result.covered_lines
    assert 8 not in result.missing_lines


def test_measure_function_rejects_invalid_function_name(
    tmp_path: Path,
) -> None:
    service = CoverageService()

    with pytest.raises(
        ValueError,
        match="function_name boş olamaz",
    ):
        service.measure_function(
            source_file=create_source_file(tmp_path),
            test_file=create_test_file(tmp_path),
            function_name="",
            start_line=1,
            end_line=4,
        )


def test_measure_function_rejects_invalid_start_line(
    tmp_path: Path,
) -> None:
    service = CoverageService()

    with pytest.raises(
        ValueError,
        match=(
            "start_line 1 veya "
            "daha büyük olmalıdır"
        ),
    ):
        service.measure_function(
            source_file=create_source_file(tmp_path),
            test_file=create_test_file(tmp_path),
            function_name="calculate_score",
            start_line=0,
            end_line=4,
        )


def test_measure_function_rejects_invalid_end_line(
    tmp_path: Path,
) -> None:
    service = CoverageService()

    with pytest.raises(
        ValueError,
        match=(
            "end_line start_line değerinden "
            "küçük olamaz"
        ),
    ):
        service.measure_function(
            source_file=create_source_file(tmp_path),
            test_file=create_test_file(tmp_path),
            function_name="calculate_score",
            start_line=4,
            end_line=2,
        )


def test_filter_lines_by_range() -> None:
    result = CoverageService._filter_lines_by_range(
        lines=(
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        start_line=2,
        end_line=5,
    )

    assert result == (
        2,
        3,
        4,
        5,
    )


def test_filter_branches_by_source_range() -> None:
    result = (
        CoverageService
        ._filter_branches_by_source_range(
            branches=(
                (2, 3),
                (2, 4),
                (7, 8),
            ),
            start_line=1,
            end_line=4,
        )
    )

    assert result == (
        (2, 3),
        (2, 4),
    )


def test_read_line_collection_normalizes_values() -> None:
    result = CoverageService._read_line_collection(
        file_data={
            "executed_lines": [
                4,
                2,
                2,
                3,
            ]
        },
        field_name="executed_lines",
    )

    assert result == (
        2,
        3,
        4,
    )


def test_read_line_collection_rejects_invalid_data() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Coverage raporundaki satır alanı "
            "geçersiz"
        ),
    ):
        CoverageService._read_line_collection(
            file_data={
                "executed_lines": "invalid",
            },
            field_name="executed_lines",
        )


def test_read_branch_collection_normalizes_values() -> None:
    result = CoverageService._read_branch_collection(
        file_data={
            "executed_branches": [
                [4, 5],
                [2, 3],
                [2, 3],
            ]
        },
        field_name="executed_branches",
    )

    assert result == (
        (2, 3),
        (4, 5),
    )


def test_read_branch_collection_rejects_invalid_record() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Coverage raporundaki branch kaydı "
            "geçersiz"
        ),
    ):
        CoverageService._read_branch_collection(
            file_data={
                "executed_branches": [
                    [2, 3, 4],
                ]
            },
            field_name="executed_branches",
        )