from pathlib import Path
from typing import Final

from services.automation_service import (
    AutomatedTestResult,
    AutomationService,
    AutomationSummary,
)


SOURCE_FILE: Final[str] = "datasets/sample_code.py"
MODULE_PATH: Final[str] = "datasets.sample_code"


def test_generate_and_execute_returns_summary(
    tmp_path: Path,
) -> None:
    """Servisin toplu otomasyon sonucu dÃ¶ndÃ¼rdÃ¼ÄŸÃ¼nÃ¼ doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert isinstance(summary, AutomationSummary)
    assert len(summary.results) == 1


def test_generate_and_execute_returns_automated_result(
    tmp_path: Path,
) -> None:
    """Her fonksiyon iÃ§in Ã¼retim ve Ã§alÄ±ÅŸtÄ±rma sonucu oluÅŸtuÄŸunu doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    result = summary.results[0]

    assert isinstance(result, AutomatedTestResult)
    assert result.artifact.function_name == "calculate_score"
    assert result.artifact.output_path.exists()


def test_generate_and_execute_runs_generated_tests(
    tmp_path: Path,
) -> None:
    """OluÅŸturulan pytest dosyasÄ±nÄ±n gerÃ§ekten Ã§alÄ±ÅŸtÄ±rÄ±ldÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    result = summary.results[0]

    assert result.execution.exit_code == 0
    assert result.execution.success is True
    assert result.execution.passed_count == 3
    assert result.execution.failed_count == 0


def test_summary_reports_generated_file_count(
    tmp_path: Path,
) -> None:
    """Ã–zetin oluÅŸturulan test dosyasÄ± sayÄ±sÄ±nÄ± doÄŸru verdiÄŸini doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.generated_file_count == 1
    assert summary.successful_file_count == 1
    assert summary.failed_file_count == 0


def test_summary_reports_scenario_and_test_counts(
    tmp_path: Path,
) -> None:
    """Senaryo ve test toplamlarÄ±nÄ±n doÄŸru hesaplandÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.total_generated_scenario_count == 3
    assert summary.total_passed_test_count == 3
    assert summary.total_failed_test_count == 0


def test_summary_reports_success(
    tmp_path: Path,
) -> None:
    """BÃ¼tÃ¼n test dosyalarÄ± baÅŸarÄ±lÄ± olduÄŸunda Ã¶zetin baÅŸarÄ±lÄ± olduÄŸunu doÄŸrular."""
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.success is True
    assert summary.results[0].success is True


def test_generate_and_execute_overwrites_existing_test(
    tmp_path: Path,
) -> None:
    """Overwrite etkinleÅŸtirildiÄŸinde test dosyasÄ±nÄ±n yeniden Ã¼retildiÄŸini doÄŸrular."""
    service = AutomationService()

    first_summary = service.generate_and_execute(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    output_path = first_summary.results[0].artifact.output_path

    output_path.write_text(
        "temporary_content = True\n",
        encoding="utf-8",
    )

    second_summary = service.generate_and_execute(
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

