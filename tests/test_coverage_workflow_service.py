from pathlib import Path
from typing import Final

from services.coverage_workflow_service import (
    CoverageArtifactResult,
    CoverageWorkflowService,
    CoverageWorkflowSummary,
)


SOURCE_FILE: Final[str] = "datasets/sample_code.py"
MODULE_PATH: Final[str] = "datasets.sample_code"


def test_generate_and_measure_returns_summary(
    tmp_path: Path,
) -> None:
    """Coverage iş akışının toplu sonuç döndürdüğünü doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert isinstance(summary, CoverageWorkflowSummary)
    assert len(summary.results) == 1


def test_generate_and_measure_returns_artifact_result(
    tmp_path: Path,
) -> None:
    """Üretim ve coverage bilgilerinin ilişkilendirildiğini doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    result = summary.results[0]

    assert isinstance(result, CoverageArtifactResult)
    assert result.artifact.function_name == "calculate_score"
    assert result.artifact.output_path.exists()


def test_generate_and_measure_runs_coverage(
    tmp_path: Path,
) -> None:
    """Üretilen test dosyası için gerçek coverage ölçümü yapıldığını doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    coverage = summary.results[0].coverage

    assert coverage.success is True
    assert coverage.total_line_count > 0
    assert coverage.total_branch_count > 0
    assert coverage.duration_seconds >= 0


def test_generated_test_currently_has_partial_coverage(
    tmp_path: Path,
) -> None:
    """
    Mevcut taslak testlerin henüz fonksiyon yollarını çalıştırmadığını doğrular.

    Bu test gelecekte otomatik girdi ve assertion üretimi eklendiğinde
    güncellenecektir.
    """
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    coverage = summary.results[0].coverage

    assert coverage.line_coverage_percent < 100.0
    assert coverage.branch_coverage_percent < 100.0
    assert coverage.has_full_coverage is False


def test_summary_reports_file_counts(
    tmp_path: Path,
) -> None:
    """Özetin dosya sayılarını doğru hesapladığını doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.measured_file_count == 1
    assert summary.successful_file_count == 1
    assert summary.failed_file_count == 0
    assert summary.full_coverage_file_count == 0


def test_summary_reports_average_coverage(
    tmp_path: Path,
) -> None:
    """Tek dosyalı özette ortalamanın ölçüm sonucuna eşit olduğunu doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    coverage = summary.results[0].coverage

    assert (
        summary.average_line_coverage_percent
        == coverage.line_coverage_percent
    )
    assert (
        summary.average_branch_coverage_percent
        == coverage.branch_coverage_percent
    )


def test_summary_reports_success(
    tmp_path: Path,
) -> None:
    """Coverage altında pytest başarılıysa özetin başarılı olduğunu doğrular."""
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.success is True
    assert summary.results[0].success is True


def test_generate_and_measure_overwrites_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite etkin olduğunda test dosyasının yeniden üretildiğini doğrular."""
    service = CoverageWorkflowService()

    first_summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    output_path = first_summary.results[0].artifact.output_path

    output_path.write_text(
        "temporary_content = True\n",
        encoding="utf-8",
    )

    second_summary = service.generate_and_measure(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    regenerated_content = (
        second_summary.results[0]
        .artifact.output_path
        .read_text(encoding="utf-8")
    )

    assert "temporary_content" not in regenerated_content
    assert second_summary.success is True