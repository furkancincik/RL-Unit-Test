import ast
import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from main import (
    FUNCTION_NAME,
    GENERATED_TEST_DIRECTORY,
    MODULE_PATH,
    SOURCE_FILE,
    ApplicationConfiguration,
    _interactive_target_selection,
    main,
    parse_cli_arguments,
    print_menu,
    project_exit_code,
    run_automated_test_pipeline,
    run_configured_operation,
    run_real_rl_training,
    run_source_analysis,
)
from models.project_analysis_result import ProjectRunStatus
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
    """İnteraktif menünün gizli örnek hedef taşımadığını doğrular."""
    configuration = parse_cli_arguments([])

    assert configuration.source_file is None
    assert configuration.module_path is None
    assert configuration.function_name is None
    assert configuration.output_directory == GENERATED_TEST_DIRECTORY
    assert configuration.operation == "menu"
    assert configuration.max_visits_per_node == 3
    assert configuration.episode_count == 3
    assert configuration.epsilon == 0.0
    assert configuration.overwrite is True
    assert configuration.timeout_seconds == 30.0


def test_interactive_menu_exposes_production_preview_and_external_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_menu()

    output = capsys.readouterr().out
    assert "RL-UNIT-TEST DEVELOPER TOOL" in output
    assert "1. Kaynak Kod / Proje Analizi" in output
    assert "2. Hızlı Statik Ön İnceleme" in output
    assert "3. Dış Kaynak Analizi" in output
    assert "0. Çıkış" in output
    assert "Control Flow Graph" not in output
    assert "DQM" not in output
    assert "Q-Learning" not in output
    assert "8 -" not in output


def test_non_interactive_production_requires_explicit_source() -> None:
    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(["--operation", "rl", "--all-functions"])

    assert error.value.code == 2


def test_production_wrapper_has_no_implicit_sample_target() -> None:
    parameters = inspect.signature(run_source_analysis).parameters

    assert parameters["source_file"].default is inspect.Parameter.empty
    assert parameters["module_path"].default is inspect.Parameter.empty
    assert parameters["function_name"].default is inspect.Parameter.empty


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

    monkeypatch.setattr("main.run_source_analysis", rl_runner)

    configuration = ApplicationConfiguration(
        source_file=source_file,
        module_path="package.target",
        function_name="process",
        all_functions=False,
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
        pipeline_timeout_seconds=None,
    )

    result = run_configured_operation(configuration)

    assert result == "training-result"

    rl_runner.assert_called_once_with(
        source_file=source_file,
        module_path="package.target",
        function_name="process",
        all_functions=False,
        output_directory=output_directory,
        max_visits_per_node=5,
        episode_count=7,
        epsilon=0.2,
        learning_rate=0.4,
        discount_factor=0.8,
        random_seed=9,
        overwrite=False,
        timeout_seconds=45.0,
        pipeline_timeout_seconds=None,
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


def test_parse_cli_arguments_supports_all_functions_and_pipeline_timeout(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")

    configuration = parse_cli_arguments(
        [
            "--source-file",
            str(source_file),
            "--module-path",
            "target",
            "--all-functions",
            "--operation",
            "rl",
            "--pipeline-timeout-seconds",
            "12.5",
        ]
    )

    assert configuration.function_name is None
    assert configuration.all_functions is True
    assert configuration.pipeline_timeout_seconds == pytest.approx(12.5)


def test_parse_cli_arguments_accepts_qualified_instance_method(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "renamed_source.py"
    source_file.write_text(
        "class Renamed:\n"
        "    def execute(self, value: int) -> int:\n"
        "        return value\n",
        encoding="utf-8",
    )

    configuration = parse_cli_arguments(
        [
            "--source-file",
            str(source_file),
            "--module-path",
            "renamed_source",
            "--function-name",
            "Renamed.execute",
            "--operation",
            "rl",
        ]
    )

    assert configuration.function_name == "Renamed.execute"
    assert configuration.all_functions is False


@pytest.mark.parametrize(
    "value",
    ("Owner.method.extra", ".method", "Owner.", "Owner/method", "Owner.method()"),
)
def test_parse_cli_arguments_rejects_malformed_qualified_target(
    tmp_path: Path,
    value: str,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def target():\n    return 1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(
            [
                "--source-file",
                str(source_file),
                "--module-path",
                "target",
                "--function-name",
                value,
                "--operation",
                "rl",
            ]
        )

    assert error.value.code == 2


def test_direct_rl_unknown_qualified_target_returns_exit_two(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "known_target.py"
    source_file.write_text(
        "class Known:\n"
        "    def run(self, value: int) -> int:\n"
        "        return value\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--source-file",
            str(source_file),
            "--module-path",
            "known_target",
            "--function-name",
            "Missing.run",
            "--operation",
            "rl",
            "--output-directory",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2


def test_interactive_target_selection_displays_and_returns_qualified_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "interactive_source.py"
    source_file.write_text(
        "class Vessel:\n"
        "    def inspect(self, enabled: bool) -> str:\n"
        "        return 'yes' if enabled else 'no'\n",
        encoding="utf-8",
    )
    analysis = PythonAnalyzer().analyze_file(source_file)
    answers = iter(("1", "Vessel.inspect"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    function_name, all_functions = _interactive_target_selection(analysis)

    assert function_name == "Vessel.inspect"
    assert all_functions is False
    assert "Vessel.inspect" in capsys.readouterr().out


def test_parse_cli_arguments_rejects_conflicting_target_modes(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(
            [
                "--source-file",
                str(source_file),
                "--function-name",
                "first",
                "--all-functions",
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf", "-inf"))
def test_parse_cli_arguments_rejects_invalid_pipeline_timeout(
    value: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        parse_cli_arguments(["--pipeline-timeout-seconds", value])

    assert error.value.code == 2


def test_run_configured_operation_uses_project_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_runner = Mock(return_value="project-result")
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    configuration = ApplicationConfiguration(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name=None,
        all_functions=True,
        output_directory=tmp_path / "output",
        operation="rl",
        episode_count=1,
        pipeline_timeout_seconds=4.0,
    )

    assert run_configured_operation(configuration) == "project-result"
    source_runner.assert_called_once_with(
        source_file=configuration.source_file,
        module_path="target",
        function_name=None,
        all_functions=True,
        output_directory=configuration.output_directory,
        max_visits_per_node=3,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        overwrite=True,
        timeout_seconds=30.0,
        pipeline_timeout_seconds=4.0,
    )


@pytest.mark.parametrize(
    ("status", "usable", "expected"),
    (
        (ProjectRunStatus.COMPLETED, True, 0),
        (ProjectRunStatus.PARTIAL, True, 0),
        (ProjectRunStatus.FAILED, False, 3),
        (ProjectRunStatus.TIMED_OUT, False, 4),
    ),
)
def test_project_exit_code_policy(
    status: ProjectRunStatus,
    usable: bool,
    expected: int,
) -> None:
    result = Mock(status=status, has_usable_result=usable)
    assert project_exit_code(result) == expected


def test_menu_project_analysis_calls_the_same_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    project = Mock(status=ProjectRunStatus.COMPLETED, has_usable_result=True)
    source_runner = Mock(return_value=project)
    responses = iter(("1", str(source_file), "target", "2", "", "", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_called_once()
    assert source_runner.call_args.kwargs["source_file"] == source_file.resolve()
    assert source_runner.call_args.kwargs["all_functions"] is True
    assert source_runner.call_args.kwargs["function_name"] is None


@pytest.mark.parametrize(
    "error",
    (TypeError("bug"), RuntimeError("bug"), AssertionError("bug")),
)
def test_menu_project_analysis_propagates_unexpected_programming_errors(
    error: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock(side_effect=error)
    responses = iter(("1", str(source_file), "target", "2", "", ""))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    with pytest.raises(type(error), match="bug"):
        main([])

    assert "Proje analizi başlatılamadı" not in capsys.readouterr().out


def test_direct_rl_operation_reports_source_selection_validation_as_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def available():\n    return 1\n", encoding="utf-8")

    assert main(
        [
            "--operation",
            "rl",
            "--source-file",
            str(source_file),
            "--module-path",
            "target",
            "--function-name",
            "missing",
            "--output-directory",
            str(tmp_path / "output"),
        ]
    ) == 2
    assert "CLI doğrulama hatası" in capsys.readouterr().out


def test_direct_rl_operation_returns_project_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    project = Mock(status=ProjectRunStatus.TIMED_OUT, has_usable_result=False)
    monkeypatch.setattr("main.run_configured_operation", lambda config: project)

    assert main(
        [
            "--operation",
            "rl",
            "--source-file",
            str(source_file),
            "--module-path",
            "target",
            "--all-functions",
        ]
    ) == 4


def test_run_source_analysis_formats_orchestrator_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Mock()
    orchestrator = Mock()
    orchestrator.run.return_value = project
    formatter = Mock()
    formatter.format.return_value = "project report"
    monkeypatch.setattr(
        "main.SourceAnalysisOrchestrator", Mock(return_value=orchestrator)
    )
    monkeypatch.setattr(
        "main.ProjectAnalysisReportFormatter", Mock(return_value=formatter)
    )

    result = run_source_analysis(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="first",
        all_functions=False,
        output_directory=tmp_path / "output",
        pipeline_timeout_seconds=3.0,
    )

    assert result is project
    assert formatter.format.called
    assert orchestrator.run.call_args.kwargs["per_function_timeout_seconds"] == 3.0


def test_menu_project_analysis_rejects_empty_source_without_sample_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_runner = Mock()
    responses = iter(("1", "", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_not_called()
    output = capsys.readouterr().out
    assert "boş bırakılamaz" in output
    assert SOURCE_FILE not in output


@pytest.mark.parametrize("kind", ("missing", "text", "syntax"))
def test_menu_project_analysis_rejects_invalid_source(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / ("target.txt" if kind == "text" else "target.py")
    if kind == "text":
        source_file.write_text("not python", encoding="utf-8")
    elif kind == "syntax":
        source_file.write_text("def broken(:\n", encoding="utf-8")
    source_runner = Mock()
    responses = iter(("1", str(source_file), "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_not_called()
    assert "Proje analizi başlatılamadı" in capsys.readouterr().out


def test_menu_single_function_lists_and_forwards_selected_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    source_runner = Mock(return_value=Mock())
    responses = iter(
        ("1", str(source_file), "target", "1", "second", "", "", "0")
    )
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    assert source_runner.call_args.kwargs["function_name"] == "second"
    assert source_runner.call_args.kwargs["all_functions"] is False
    output = capsys.readouterr().out
    assert "- first" in output
    assert "- second" in output
    assert str(source_file.resolve()) in output


def test_menu_single_function_rejects_unknown_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock()
    responses = iter(("1", str(source_file), "target", "1", "missing", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_not_called()
    assert "Desteklenen fonksiyon bulunamadı" in capsys.readouterr().out


def test_menu_project_defaults_and_safe_output_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "order-rules.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock(return_value=Mock())
    responses = iter(("1", str(source_file), "target", "2", "", "", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    values = source_runner.call_args.kwargs
    assert values["output_directory"].name == "order_rules_analysis"
    assert values["max_visits_per_node"] == 3
    assert values["episode_count"] == 3
    assert values["epsilon"] == 0.0
    assert values["learning_rate"] == 0.5
    assert values["discount_factor"] == 0.9
    assert values["random_seed"] == 42
    assert values["timeout_seconds"] == 30.0
    assert values["pipeline_timeout_seconds"] is None


def test_menu_project_validated_advanced_settings_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock(return_value=Mock())
    responses = iter(
        (
            "1", str(source_file), "target", "2", "", "e",
            "4", "2", "0.1", "0.4", "0.8", "9", "45", "12", "0",
        )
    )
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    values = source_runner.call_args.kwargs
    assert values["max_visits_per_node"] == 4
    assert values["episode_count"] == 2
    assert values["epsilon"] == pytest.approx(0.1)
    assert values["learning_rate"] == pytest.approx(0.4)
    assert values["discount_factor"] == pytest.approx(0.8)
    assert values["random_seed"] == 9
    assert values["timeout_seconds"] == pytest.approx(45.0)
    assert values["pipeline_timeout_seconds"] == pytest.approx(12.0)


def test_menu_invalid_advanced_setting_does_not_reach_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock()
    responses = iter(("1", str(source_file), "target", "2", "", "e", "0", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_not_called()
    assert "Değer 1 veya daha büyük olmalıdır" in capsys.readouterr().out


def test_menu_static_preview_uses_real_source_without_running_rl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    source_runner = Mock()
    analyzer_report = Mock()
    responses = iter(("2", str(source_file), "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("main.print_analyzer_report", analyzer_report)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    analyzer_report.assert_called_once_with(source_file.resolve())
    source_runner.assert_not_called()


def test_old_option_eight_is_invalid_and_does_not_start_analysis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_runner = Mock()
    responses = iter(("8", "0"))
    monkeypatch.setattr("main.run_source_analysis", source_runner)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0
    source_runner.assert_not_called()
    assert "Lütfen 0, 1, 2 veya 3 girin" in capsys.readouterr().out


def test_menu_external_analysis_uses_terminal_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = Mock()
    adapter_type = Mock(return_value=adapter)
    responses = iter(("3", "0"))
    monkeypatch.setattr("main.ExternalSourceTerminalAdapter", adapter_type)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert main([]) == 0

    adapter_type.assert_called_once()
    adapter.run_menu.assert_called_once_with()


@pytest.mark.parametrize(
    "error",
    (TypeError("bug"), RuntimeError("bug"), AssertionError("bug")),
)
def test_menu_external_analysis_propagates_programming_errors(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = Mock()
    adapter.run_menu.side_effect = error
    responses = iter(("3",))
    monkeypatch.setattr("main.ExternalSourceTerminalAdapter", Mock(return_value=adapter))
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    with pytest.raises(type(error), match="bug"):
        main([])


def test_advanced_cli_operations_retain_explicit_demo_defaults() -> None:
    configuration = parse_cli_arguments(["--operation", "cfg"])

    assert configuration.source_file == Path(SOURCE_FILE)
    assert configuration.module_path == MODULE_PATH
    assert configuration.function_name == FUNCTION_NAME


def test_explicit_sample_source_remains_available_for_production() -> None:
    configuration = parse_cli_arguments(
        [
            "--operation",
            "rl",
            "--source-file",
            SOURCE_FILE,
            "--module-path",
            MODULE_PATH,
            "--function-name",
            FUNCTION_NAME,
        ]
    )

    assert configuration.source_file == Path(SOURCE_FILE)
    assert configuration.function_name == FUNCTION_NAME
