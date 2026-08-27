from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleSelectionMode,
    ExternalModuleStatus,
    ExternalSourceAnalysisResult,
    ExternalSourceKind,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    LocalProjectDirectory,
    PublicGitHubRepository,
    UploadedPythonFile,
)
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)
from models.project_analysis_result import FunctionRunStatus
from models.project_analysis_result import TargetSelectionMode
from services.external_source_terminal_service import (
    ExternalSourceInteractiveValidationError,
    ExternalSourceTerminalAdapter,
)
from services.external_source_analysis_service import (
    ExternalSourceAnalysisValidationError,
)


def _adapter(responses: tuple[str, ...], service: Mock) -> tuple[ExternalSourceTerminalAdapter, list[str]]:
    values = iter(responses)
    output: list[str] = []
    return (
        ExternalSourceTerminalAdapter(
            input_fn=lambda prompt: next(values),
            output_fn=output.append,
            service_factory=Mock(return_value=service),
        ),
        output,
    )


def _successful_service() -> Mock:
    service = Mock()
    service.run.return_value = _result()
    return service


def _result() -> ExternalSourceAnalysisResult:
    output_root = Path("output/external_source_analysis").resolve()
    return ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.INLINE_PYTHON_SOURCE,
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        status=ExternalAnalysisStatus.STATIC_COMPLETED,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(),
        output_root=output_root,
        report_path=output_root / "report.json",
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


def test_submenu_keeps_four_source_kinds_separate() -> None:
    service = Mock()
    adapter, output = _adapter(("0",), service)

    adapter.run_menu()

    text = "\n".join(output)
    assert "1. Python kodu yapıştır" in text
    assert "2. Python dosyası seç" in text
    assert "3. Yerel proje klasörü seç" in text
    assert "4. Public GitHub URL gir" in text
    service.run.assert_not_called()


def test_terminal_explicit_qualified_target_selection_uses_exact_public_identity() -> None:
    service = _successful_service()
    adapter, _ = _adapter(
        (
            "1", "class Vessel:", "    def inspect(self):", "        return 1",
            "__END__", "1", "", "1", "2", "1", "Vessel.inspect", "", "", "", "0",
        ),
        service,
    )

    adapter.run_menu()

    selection = service.run.call_args.args[0].configuration.target_selection
    assert selection.mode is TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS
    assert selection.selectors[0].to_dict() == {
        "module_identity": "inline_source",
        "qualified_name": "Vessel.inspect",
    }


def test_inline_multiline_static_request_excludes_end_marker() -> None:
    service = _successful_service()
    adapter, output = _adapter(
        ("1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", "", "0"),
        service,
    )

    adapter.run_menu()

    request = service.run.call_args.args[0]
    assert isinstance(request.source, InlinePythonSource)
    assert request.source.source_text == "def first():\n    return 1"
    assert "__END__" not in request.source.source_text
    assert request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY
    assert "def first" not in "\n".join(output)


def test_terminal_forwards_optional_total_project_timeout_separately() -> None:
    service = _successful_service()
    adapter, _ = _adapter(
        (
            "1", "def target():", "    return 1", "__END__", "1", "",
            "1", "1", "", "", "18.5", "0",
        ),
        service,
    )

    adapter.run_menu()

    configuration = service.run.call_args.args[0].configuration
    assert configuration.project_timeout_seconds == pytest.approx(18.5)
    assert configuration.per_function_pipeline_timeout_seconds == pytest.approx(120.0)


def test_empty_inline_source_is_controlled() -> None:
    service = Mock()
    adapter, output = _adapter(("1", "__END__", "0"), service)

    adapter.run_menu()

    service.run.assert_not_called()
    assert "boş olamaz" in "\n".join(output)


def test_uploaded_python_file_is_read_without_mutating_original(tmp_path: Path) -> None:
    source = tmp_path / "uploaded.py"
    original = b"def first():\n    return 1\n"
    source.write_bytes(original)
    service = _successful_service()
    adapter, _ = _adapter(("2", str(source), "1", "", "1", "1", "", "", "", "0"), service)

    adapter.run_menu()

    request = service.run.call_args.args[0]
    assert isinstance(request.source, UploadedPythonFile)
    assert request.source.original_filename == "uploaded.py"
    assert request.source.file_bytes == original
    assert source.read_bytes() == original


@pytest.mark.parametrize("name", ("missing.py", "target.txt"))
def test_uploaded_python_file_validation_is_controlled(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if path.suffix == ".txt":
        path.write_text("not python", encoding="utf-8")
    service = Mock()
    adapter, output = _adapter(("2", str(path), "0"), service)

    adapter.run_menu()

    service.run.assert_not_called()
    assert "Python dosyası" in "\n".join(output)


def test_local_directory_remains_user_owned_payload(tmp_path: Path) -> None:
    service = _successful_service()
    adapter, _ = _adapter(("3", str(tmp_path), "1", "", "1", "1", "", "", "", "0"), service)

    adapter.run_menu()

    request = service.run.call_args.args[0]
    assert isinstance(request.source, LocalProjectDirectory)
    assert request.source.directory == tmp_path.resolve()
    assert tmp_path.is_dir()


def test_github_url_is_forwarded_without_terminal_revalidation() -> None:
    service = _successful_service()
    adapter, _ = _adapter(
        ("4", "https://github.com/owner/repository", "1", "", "1", "1", "", "", "", "0"),
        service,
    )

    adapter.run_menu()

    request = service.run.call_args.args[0]
    assert isinstance(request.source, PublicGitHubRepository)
    assert request.source.repository_url == "https://github.com/owner/repository"


def test_trusted_dynamic_requires_confirmation_and_forwards_configuration() -> None:
    service = _successful_service()
    adapter, output = _adapter(
        (
            "1", "def first():", "    return 1", "__END__", "2", "EVET",
            "", "3", " pkg.alpha ,pkg.alpha, pkg.beta ", "1", "4", "5", "45", "7", "9",
            "e", "e", "", "0",
        ),
        service,
    )

    adapter.run_menu()

    request = service.run.call_args.args[0]
    config = request.configuration
    assert request.execution_policy is ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS
    assert config.module_selection.mode is ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES
    assert config.module_selection.values == ("pkg.alpha", "pkg.beta")
    assert config.maximum_selected_modules == 4
    assert config.maximum_functions_per_module == 5
    assert config.per_function_pipeline_timeout_seconds == pytest.approx(45.0)
    assert config.episode_count == 7
    assert config.random_seed == 9
    assert config.run_greedy_baseline is True
    assert config.run_strategy_comparison is True
    assert "sandbox değildir" in "\n".join(output)


def test_trusted_dynamic_cancel_does_not_run_service() -> None:
    service = Mock()
    adapter, _ = _adapter(
        ("1", "def first():", "    return 1", "__END__", "2", "hayır", "0"),
        service,
    )

    adapter.run_menu()

    service.run.assert_not_called()


def test_controlled_validation_returns_to_submenu() -> None:
    service = Mock()
    adapter, output = _adapter(("9", "0"), service)

    adapter.run_menu()

    assert "Geçersiz dış kaynak seçimi" in "\n".join(output)


def test_service_domain_value_error_returns_to_submenu() -> None:
    service = Mock()
    service.run.side_effect = ExternalSourceAnalysisValidationError(
        "output root source içinde olamaz"
    )
    adapter, output = _adapter(
        ("1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", "", "0"),
        service,
    )

    adapter.run_menu()

    assert "Dış kaynak doğrulama hatası" in "\n".join(output)


def test_unexpected_internal_value_error_propagates() -> None:
    service = Mock()
    service.run.side_effect = ValueError("internal bug")
    adapter, _ = _adapter(
        ("1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", ""),
        service,
    )

    with pytest.raises(ValueError, match="internal bug"):
        adapter.run_menu()


@pytest.mark.parametrize("error", (TypeError("bug"), RuntimeError("bug"), AssertionError("bug")))
def test_programming_errors_propagate(error: BaseException) -> None:
    service = Mock()
    service.run.side_effect = error
    adapter, _ = _adapter(
        ("1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", ""),
        service,
    )

    with pytest.raises(type(error), match="bug"):
        adapter.run_menu()


def test_keyboard_interrupt_is_reported_as_controlled_cancel() -> None:
    service = Mock()
    service.run.side_effect = KeyboardInterrupt
    adapter, output = _adapter(
        ("1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", "", "0"),
        service,
    )

    adapter.run_menu()

    assert "iptal edildi" in "\n".join(output).lower()


def test_real_inline_syntax_error_is_controlled_and_source_is_not_printed(
    tmp_path: Path,
) -> None:
    marker = "secret_syntax_marker"
    service = __import__(
        "services.external_source_analysis_service",
        fromlist=["ExternalSourceAnalysisService"],
    ).ExternalSourceAnalysisService()
    adapter, output = _adapter(
        (
            "1", f"def {marker}(:", "__END__", "1", str(tmp_path / "output"),
            "1", "1", "", "", "", "0",
        ),
        service,
    )

    adapter.run_menu()

    text = "\n".join(output)
    assert "SYNTAX_ERROR" in text
    assert marker not in text
    assert not any(tmp_path.glob("rl-unit-test-inline-*"))


def test_consecutive_sources_build_isolated_requests(tmp_path: Path) -> None:
    source_file = tmp_path / "second.py"
    source_file.write_text("def second():\n    return 2\n", encoding="utf-8")
    service = Mock()
    service.run.side_effect = (_result(), _result())
    adapter, _ = _adapter(
        (
            "1", "def first():", "    return 1", "__END__", "1", "", "1", "1", "", "", "",
            "2", str(source_file), "1", "", "1", "1", "", "", "", "0",
        ),
        service,
    )

    adapter.run_menu()

    first_request = service.run.call_args_list[0].args[0]
    second_request = service.run.call_args_list[1].args[0]
    assert isinstance(first_request.source, InlinePythonSource)
    assert isinstance(second_request.source, UploadedPythonFile)
    assert first_request is not second_request
    assert first_request.configuration is not second_request.configuration


def test_real_terminal_inline_trusted_dynamic_runs_comparison(tmp_path: Path) -> None:
    from services.external_source_analysis_service import ExternalSourceAnalysisService

    service = ExternalSourceAnalysisService()
    output_root = tmp_path / "inline_dynamic"
    adapter, output = _adapter(
        (
            "1", "def classify(value: int) -> str:", "    if value > 0:",
            "        return 'positive'", "    return 'other'", "__END__",
            "2", "EVET", str(output_root), "1", "1", "1", "2", "60", "1", "42",
            "e", "e", "", "0",
        ),
        service,
    )

    adapter.run_menu()

    report = output_root / "external_source_analysis_report.json"
    assert report.is_file()
    text = "\n".join(output)
    assert "TRUSTED_DYNAMIC_ANALYSIS" in text
    assert "RL test" in text
    assert "Greedy seçilen" in text
    assert "Winner" in text
    assert "def classify" not in text


def test_real_terminal_local_multimodule_trusted_dynamic(tmp_path: Path) -> None:
    from services.external_source_analysis_service import ExternalSourceAnalysisService

    project = tmp_path / "project"
    project.mkdir()
    (project / "alpha.py").write_text(
        "def alpha(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    (project / "beta.py").write_text(
        "def beta(value: int) -> int:\n    return value - 1\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "local_dynamic"
    adapter, output = _adapter(
        (
            "3", str(project), "2", "EVET", str(output_root), "1", "1", "2", "1",
            "60", "1", "42", "h", "h", "", "0",
        ),
        ExternalSourceAnalysisService(),
    )

    adapter.run_menu()

    report = output_root / "external_source_analysis_report.json"
    assert report.is_file()
    assert project.is_dir()
    text = "\n".join(output)
    assert "Keşfedilen modül     : 2" in text
    assert "Cleanup durumu        : NOT_REQUIRED" in text


def test_terminal_result_exposes_normalized_input_rejections() -> None:
    diagnostic = PipelineDiagnosticResult(
        status=PipelineRunStatus.COMPLETED,
        source_file=Path("target.py"),
        function_name="target",
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=1.0,
        funnel=PipelineFunnelSnapshot(
            bounded_path_count=19,
            input_generation_accepted_count=1,
            input_generation_rejected_count=18,
            pre_concrete_scenario_count=1,
            concrete_validation_accepted_count=1,
            concrete_validation_rejected_count=0,
            final_scenario_count=1,
        ),
        scenario_rejection_counts=(
            ("UNSUPPORTED_INPUT_SYNTHESIS", 18),
        ),
        line_coverage_percent=18.75,
        branch_coverage_percent=12.5,
    )
    function = SimpleNamespace(
        target=SimpleNamespace(qualified_name="target"),
        status=FunctionRunStatus.COMPLETED,
        diagnostic=diagnostic,
        scenario_count=1,
        concrete_accepted_count=1,
        concrete_rejected_count=0,
        rl_test_count=1,
        strategy_comparison=None,
    )
    project = SimpleNamespace(function_results=(function,))
    module = ExternalModuleAnalysisResult(
        relative_path="target.py",
        module_name="target",
        status=ExternalModuleStatus.COMPLETED,
        discovered_function_count=1,
        project_result=project,
        issue_category=None,
        issue_message=None,
        artifact_paths=(),
        discovered_function_names=("target",),
    )
    result = ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.INLINE_PYTHON_SOURCE,
        execution_policy=ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
        status=ExternalAnalysisStatus.COMPLETED,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
        output_root=Path("output"),
        report_path=Path("output/report.json"),
        duration_seconds=1.0,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )

    report = ExternalSourceTerminalAdapter.format_result(result)

    assert "Genel durum          : COMPLETED" in report
    assert "Sınırlandırılmış yol : 19" in report
    assert "Girdi üretimi        : 1 kabul / 18 red" in report
    assert "Red kategorileri     : UNSUPPORTED_INPUT_SYNTHESIS: 18" in report
    assert "Scenario           : 1" in report
    assert "Concrete           : 1 kabul / 0 red" in report
    assert "Line coverage      : %18.75" in report
    assert "Branch coverage    : %12.50" in report
