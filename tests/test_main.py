import ast
from pathlib import Path

from main import (
    MODULE_PATH,
    SOURCE_FILE,
    run_automated_test_pipeline,
)
from services.automation_service import (
    AutomationSummary as SummaryModel,
)


def test_run_automated_test_pipeline_returns_summary(
    tmp_path: Path,
) -> None:
    """CLI otomasyon fonksiyonunun sonuç özeti döndürdüğünü doğrular."""
    summary = run_automated_test_pipeline(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert isinstance(summary, SummaryModel)
    assert summary.success is True


def test_run_automated_test_pipeline_creates_test_file(
    tmp_path: Path,
) -> None:
    """CLI otomasyon fonksiyonunun pytest dosyası ürettiğini doğrular."""
    summary = run_automated_test_pipeline(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    output_path = summary.results[0].artifact.output_path

    assert output_path.exists()
    assert output_path.name == "test_calculate_score.py"


def test_run_automated_test_pipeline_creates_valid_python(
    tmp_path: Path,
) -> None:
    """CLI üzerinden oluşturulan dosyanın geçerli Python kodu olduğunu doğrular."""
    summary = run_automated_test_pipeline(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    output_path = summary.results[0].artifact.output_path
    generated_code = output_path.read_text(
        encoding="utf-8",
    )

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)


def test_run_automated_test_pipeline_reports_expected_counts(
    tmp_path: Path,
) -> None:
    """CLI otomasyon özetindeki test ve senaryo sayılarını doğrular."""
    summary = run_automated_test_pipeline(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert summary.generated_file_count == 1
    assert summary.total_generated_scenario_count == 3
    assert summary.total_passed_test_count == 3
    assert summary.total_failed_test_count == 0