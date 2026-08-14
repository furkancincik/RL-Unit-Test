import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from main import (
    FUNCTION_NAME,
    GENERATED_TEST_DIRECTORY,
    MODULE_PATH,
    SOURCE_FILE,
    ApplicationConfiguration,
    parse_cli_arguments,
    run_automated_test_pipeline,
    run_configured_operation,
    run_real_rl_training,
)
from services.automation_service import (
    AutomationSummary as SummaryModel,
)


def test_run_automated_test_pipeline_returns_summary(
    tmp_path: Path,
) -> None:
    """CLI otomasyonunun sonuç özeti döndürdüğünü doğrular."""
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
    """Üretilen dosyanın geçerli Python olduğunu doğrular."""
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


def test_parse_cli_arguments_preserves_default_configuration(
) -> None:
    """Argüman verilmediğinde eski örnek hedefin korunduğunu doğrular."""
    configuration = parse_cli_arguments([])

    assert configuration.source_file == Path(SOURCE_FILE)
    assert configuration.module_path == MODULE_PATH
    assert configuration.function_name == FUNCTION_NAME
    assert configuration.output_directory == GENERATED_TEST_DIRECTORY
    assert configuration.operation == "menu"
    assert configuration.max_visits_per_node == 3
    assert configuration.episode_count == 3
    assert configuration.epsilon == 0.0
    assert configuration.overwrite is True
    assert configuration.timeout_seconds == 30.0


def test_parse_cli_arguments_builds_custom_configuration(
    tmp_path: Path,
) -> None:
    """Özel hedef ve RL ayarlarının ayrıştırıldığını doğrular."""
    source_file = tmp_path / "custom_target.py"
    source_file.write_text(
        "def process(value: int) -> int:\n    return value\n",
        encoding="utf-8",
    )

    configuration = parse_cli_arguments(
        [
            "--source-file",
            str(source_file),
            "--module-path",
            "package.custom_target",
            "--function-name",
            "process",
            "--output-directory",
            str(tmp_path / "generated"),
            "--operation",
            "rl",
            "--max-visits-per-node",
            "5",
            "--episode-count",
            "7",
            "--epsilon",
            "0.2",
            "--learning-rate",
            "0.4",
            "--discount-factor",
            "0.8",
            "--random-seed",
            "9",
            "--timeout-seconds",
            "45",
            "--no-overwrite",
        ]
    )

    assert configuration.source_file == source_file
    assert configuration.module_path == "package.custom_target"
    assert configuration.function_name == "process"
    assert configuration.output_directory == tmp_path / "generated"
    assert configuration.operation == "rl"
    assert configuration.max_visits_per_node == 5
    assert configuration.episode_count == 7
    assert configuration.epsilon == 0.2
    assert configuration.learning_rate == 0.4
    assert configuration.discount_factor == 0.8
    assert configuration.random_seed == 9
    assert configuration.overwrite is False
    assert configuration.timeout_seconds == 45.0


@pytest.mark.parametrize(
    "arguments",
    (
        ("--source-file", "target.txt"),
        ("--module-path", "invalid-module"),
        ("--function-name", "invalid-name"),
        ("--max-visits-per-node", "0"),
        ("--episode-count", "0"),
        ("--epsilon", "1.1"),
        ("--learning-rate", "-0.1"),
        ("--discount-factor", "2.0"),
        ("--timeout-seconds", "0"),
    ),
)
def test_parse_cli_arguments_rejects_invalid_values(
    arguments: tuple[str, str],
) -> None:
    """Geçersiz CLI değerlerinin reddedildiğini doğrular."""
    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(list(arguments))

    assert error.value.code == 2


def test_parse_cli_arguments_rejects_missing_python_file(
    tmp_path: Path,
) -> None:
    """Var olmayan Python kaynak dosyasının reddedildiğini doğrular."""
    missing_source = tmp_path / "missing.py"

    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(
            [
                "--source-file",
                str(missing_source),
            ]
        )

    assert error.value.code == 2


def test_run_configured_operation_forwards_rl_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doğrudan RL işleminin yapılandırmayı aktardığını doğrular."""
    source_file = tmp_path / "target.py"
    output_directory = tmp_path / "generated"
    rl_runner = Mock(return_value="training-result")

    monkeypatch.setattr(
        "main.run_real_rl_training",
        rl_runner,
    )

    configuration = ApplicationConfiguration(
        source_file=source_file,
        module_path="package.target",
        function_name="process",
        output_directory=output_directory,
        operation="rl",
        max_visits_per_node=5,
        episode_count=7,
        epsilon=0.2,
        learning_rate=0.4,
        discount_factor=0.8,
        random_seed=9,
        overwrite=False,
        timeout_seconds=45.0,
    )

    result = run_configured_operation(configuration)

    assert result == "training-result"

    rl_runner.assert_called_once_with(
        source_file=source_file,
        module_path="package.target",
        function_name="process",
        output_directory=output_directory,
        max_visits_per_node=5,
        episode_count=7,
        epsilon=0.2,
        learning_rate=0.4,
        discount_factor=0.8,
        random_seed=9,
        overwrite=False,
        timeout_seconds=45.0,
    )


def test_run_real_rl_training_forwards_path_expansion_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main wrapper'ının path ayarını servise ilettiğini doğrular."""
    service = Mock()
    result = Mock(
        report="training report",
        scenario_count=4,
        completed_episode_count=2,
        q_table_state_count=3,
        success=True,
    )
    service.run.return_value = result

    service_factory = Mock(return_value=service)
    monkeypatch.setattr(
        "main.RealRLTrainingService",
        service_factory,
    )

    returned_result = run_real_rl_training(
        source_file=tmp_path / "target.py",
        module_path="package.target",
        function_name="process",
        output_directory=tmp_path / "generated",
        max_visits_per_node=5,
        episode_count=2,
        epsilon=0.0,
        learning_rate=0.4,
        discount_factor=0.8,
        random_seed=9,
        overwrite=False,
        timeout_seconds=45.0,
    )

    assert returned_result is result
    service_factory.assert_called_once_with()
    service.run.assert_called_once_with(
        source_file=tmp_path / "target.py",
        module_path="package.target",
        function_name="process",
        output_directory=tmp_path / "generated",
        max_visits_per_node=5,
        episode_count=2,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        learning_rate=0.4,
        discount_factor=0.8,
        random_seed=9,
        overwrite=False,
        timeout_seconds=45.0,
    )
