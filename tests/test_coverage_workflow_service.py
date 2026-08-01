from __future__ import annotations

from pathlib import Path

import pytest

from services.coverage_workflow_service import (
    CoverageArtifactResult,
    CoverageWorkflowService,
    CoverageWorkflowSummary,
)

SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"


def test_generate_and_measure_creates_single_result(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    assert isinstance(
        summary,
        CoverageWorkflowSummary,
    )

    assert summary.measured_file_count == 1
    assert len(summary.results) == 1

    result = summary.results[0]

    assert isinstance(
        result,
        CoverageArtifactResult,
    )

    assert result.artifact.function_name == (
        "calculate_score"
    )

    assert result.artifact.scenario_count == 3


def test_generate_and_measure_runs_generated_tests_successfully(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    result = summary.results[0]

    assert result.success is True
    assert result.coverage.test_exit_code == 0

    assert summary.success is True
    assert summary.successful_file_count == 1
    assert summary.failed_file_count == 0


def test_generate_and_measure_reaches_full_line_coverage(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    coverage = summary.results[0].coverage

    assert coverage.line_coverage_percent == 100.0
    assert coverage.covered_line_count == (
        coverage.total_line_count
    )
    assert coverage.missing_line_count == 0


def test_generate_and_measure_reaches_full_branch_coverage(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    coverage = summary.results[0].coverage

    assert coverage.branch_coverage_percent == 100.0
    assert coverage.covered_branch_count == (
        coverage.total_branch_count
    )
    assert coverage.missing_branch_count == 0


def test_generate_and_measure_marks_full_coverage(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    coverage = summary.results[0].coverage

    assert coverage.has_full_coverage is True
    assert summary.full_coverage_file_count == 1


def test_generate_and_measure_calculates_average_coverage(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    assert (
        summary.average_line_coverage_percent
        == 100.0
    )

    assert (
        summary.average_branch_coverage_percent
        == 100.0
    )


def test_generate_and_measure_writes_real_test_file(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    generated_file = (
        summary.results[0]
        .artifact.output_path
    )

    assert generated_file.exists()

    generated_source = generated_file.read_text(
        encoding="utf-8",
    )

    assert (
        "result = calculate_score("
        in generated_source
    )

    assert "assert callable" not in generated_source
    assert "TODO" not in generated_source


def test_generate_and_measure_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    missing_source = (
        tmp_path / "missing.py"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        service.generate_and_measure(
            source_file=missing_source,
            module_path=MODULE_PATH,
            output_directory=tmp_path,
            overwrite=True,
            timeout_seconds=30.0,
        )


def test_generate_and_measure_rejects_invalid_timeout(
    tmp_path: Path,
) -> None:
    service = CoverageWorkflowService()

    with pytest.raises(
        ValueError,
        match=(
            "Coverage zaman aşımı sıfırdan "
            "büyük olmalıdır"
        ),
    ):
        service.generate_and_measure(
            source_file=SOURCE_FILE,
            module_path=MODULE_PATH,
            output_directory=tmp_path,
            overwrite=True,
            timeout_seconds=0.0,
        )