from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalModuleStatus,
    ExternalSourceAnalysisRequest,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    LocalProjectDirectory,
    PublicGitHubRepository,
    UploadedPythonFile,
)
from models.project_analysis_result import ProjectRunStatus
from services.external_source_analysis_service import ExternalSourceAnalysisService


def _configuration(tmp_path: Path, **kwargs: object) -> ExternalAnalysisConfiguration:
    return ExternalAnalysisConfiguration(output_root=tmp_path / "output", **kwargs)


def test_inline_static_discovery_writes_no_source_to_json_and_cleans(tmp_path: Path) -> None:
    marker = "never_serialize_this_source"
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            source=InlinePythonSource(f"def target():\n    return {marker!r}\n"),
            execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            configuration=_configuration(tmp_path),
        )
    )
    assert result.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert result.module_results[0].status is ExternalModuleStatus.STATIC_ONLY
    assert result.module_results[0].line_coverage_percent is None
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    report = result.report_path.read_text(encoding="utf-8")
    assert marker not in report
    assert "rl-unit-test-inline-" not in report


def test_keyboard_interrupt_cleans_inline_workspace() -> None:
    acquisition = Mock()
    acquisition.resolve.side_effect = KeyboardInterrupt
    original_mkdtemp = tempfile.mkdtemp
    created: list[Path] = []

    def tracked_mkdtemp(*, prefix: str) -> str:
        path = Path(original_mkdtemp(prefix=prefix)).resolve()
        created.append(path)
        return str(path)

    request = ExternalSourceAnalysisRequest(
        source=InlinePythonSource("def target():\n    return 1\n"),
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        configuration=ExternalAnalysisConfiguration(output_root=Path("output/interrupt")),
    )
    with patch(
        "services.external_source_analysis_service.tempfile.mkdtemp",
        side_effect=tracked_mkdtemp,
    ):
        with pytest.raises(KeyboardInterrupt):
            ExternalSourceAnalysisService(acquisition_service=acquisition).run(request)

    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.parametrize(
    "source,maximum_payload_bytes,category",
    (
        (InlinePythonSource("x='a' * 100\n"), 8, "SOURCE_LIMIT_EXCEEDED"),
        (InlinePythonSource("def broken(:\n"), 100, "SYNTAX_ERROR"),
        (InlinePythonSource("value='\x00'\n"), 100, "INVALID_CONTROL_CHARACTER"),
    ),
)
def test_inline_validation_is_controlled(
    tmp_path: Path,
    source: InlinePythonSource,
    maximum_payload_bytes: int,
    category: str,
) -> None:
    configuration = _configuration(
        tmp_path,
        maximum_payload_bytes=maximum_payload_bytes,
    )
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(source, ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY, configuration)
    )
    assert result.status is ExternalAnalysisStatus.FAILED
    assert category in result.issues


@pytest.mark.parametrize("filename", ("../target.py", r"C:\\target.py", r"folder\\target.py", "target.txt"))
def test_uploaded_filename_validation(tmp_path: Path, filename: str) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile(filename, b"value=1\n"),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            _configuration(tmp_path),
        )
    )
    assert result.status is ExternalAnalysisStatus.FAILED
    assert "INVALID_UPLOAD_FILENAME" in result.issues


def test_upload_workspace_is_unique_and_payload_is_not_serialized(tmp_path: Path) -> None:
    service = ExternalSourceAnalysisService()
    request = lambda output: ExternalSourceAnalysisRequest(
        UploadedPythonFile("same.py", b"secret_upload=1\n"),
        ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        ExternalAnalysisConfiguration(output_root=output),
    )
    first = service.run(request(tmp_path / "first"))
    second = service.run(request(tmp_path / "second"))
    assert first.report_path != second.report_path
    assert "secret_upload" not in first.report_path.read_text(encoding="utf-8")
    assert first.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert second.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED


def test_static_local_directory_never_calls_orchestrator(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    factory = Mock()
    result = ExternalSourceAnalysisService(orchestrator_factory=factory).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            _configuration(tmp_path),
        )
    )
    factory.assert_not_called()
    assert result.status is ExternalAnalysisStatus.STATIC_COMPLETED


def test_static_discovery_keeps_function_inventory_despite_execution_limit(
    tmp_path: Path,
) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(
                "def first():\n return 1\n\n"
                "def second():\n return 2\n\n"
                "def third():\n return 3\n"
            ),
            configuration=_configuration(tmp_path, maximum_functions_per_module=1),
        )
    )

    assert result.module_results[0].discovered_function_count == 3
    assert result.module_results[0].analyzed_function_count == 0
    assert result.module_results[0].limit_skipped_function_count == 0


def test_github_is_static_by_default(tmp_path: Path) -> None:
    request = ExternalSourceAnalysisRequest(
        PublicGitHubRepository("https://github.com/owner/repo"),
        configuration=_configuration(tmp_path),
    )
    assert request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY


def test_dynamic_analysis_requires_explicit_trust(tmp_path: Path) -> None:
    request = ExternalSourceAnalysisRequest(
        InlinePythonSource("def target():\n    return 1\n"),
        configuration=_configuration(tmp_path),
    )
    assert request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY


def test_trusted_dynamic_analysis_requires_worker_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per-function pipeline timeout"):
        ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("def target():\n return 1\n"),
                ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
                _configuration(
                    tmp_path,
                    per_function_pipeline_timeout_seconds=None,
                ),
            )
        )


@pytest.mark.parametrize(
    "selection,expected",
    (
        (ExternalModuleSelection(ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS, ("a.py",)), "a.py"),
        (ExternalModuleSelection(ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES, ("b",)), "b.py"),
    ),
)
def test_explicit_module_selection(tmp_path: Path, selection: ExternalModuleSelection, expected: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("a.py", "b.py"):
        (project / name).write_text("def target():\n    return 1\n", encoding="utf-8")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            _configuration(tmp_path, module_selection=selection),
        )
    )
    selected = [item.relative_path for item in result.module_results if item.status is ExternalModuleStatus.STATIC_ONLY]
    assert selected == [expected]


def test_all_eligible_limit_is_deterministic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("c.py", "a.py", "b.py"):
        (project / name).write_text("def target():\n    return 1\n", encoding="utf-8")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            _configuration(tmp_path, maximum_selected_modules=2),
        )
    )
    assert [item.relative_path for item in result.module_results] == ["a.py", "b.py", "c.py"]
    assert result.module_results[-1].status is ExternalModuleStatus.SKIPPED_LIMIT


def test_ambiguous_and_syntax_invalid_modules_are_isolated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "__init__.py").write_text("", encoding="utf-8")
    (project / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project / "src" / "pkg" / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    (project / "good.py").write_text("def good():\n return 1\n", encoding="utf-8")
    (project / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(LocalProjectDirectory(project), configuration=_configuration(tmp_path))
    )
    statuses = {item.relative_path: item.status for item in result.module_results}
    assert statuses["good.py"] is ExternalModuleStatus.STATIC_ONLY
    assert statuses["broken.py"] is ExternalModuleStatus.UNSUPPORTED
    assert statuses["src/pkg/module.py"] is ExternalModuleStatus.UNSUPPORTED


def test_parent_sys_path_is_never_modified(tmp_path: Path) -> None:
    before = tuple(sys.path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(LocalProjectDirectory(project), configuration=_configuration(tmp_path))
    )
    assert tuple(sys.path) == before


def test_dynamic_forwards_import_root_limits_and_strategy_flags(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    orchestrator = Mock()
    project_result = Mock(status=ProjectRunStatus.COMPLETED, function_results=(), discovered_targets=(), report_path=tmp_path / "report.json")
    orchestrator.run.return_value = project_result
    result = ExternalSourceAnalysisService(orchestrator_factory=lambda: orchestrator).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(
                tmp_path,
                maximum_functions_per_module=2,
                run_greedy_baseline=True,
                run_strategy_comparison=True,
            ),
        )
    )
    call = orchestrator.run.call_args.kwargs
    assert call["import_root"] == project.resolve()
    assert call["maximum_functions"] == 2
    assert call["run_greedy_baseline"] is True
    assert call["run_strategy_comparison"] is True
    assert result.status is ExternalAnalysisStatus.COMPLETED


def test_unexpected_orchestrator_runtime_error_propagates_and_cleans(tmp_path: Path) -> None:
    temporary_root = Path(os.environ["TEMP"])
    before = frozenset(temporary_root.glob("rl-unit-test-inline-*"))
    orchestrator = Mock()
    orchestrator.run.side_effect = RuntimeError("unexpected")
    with pytest.raises(RuntimeError, match="unexpected"):
        ExternalSourceAnalysisService(orchestrator_factory=lambda: orchestrator).run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("def target():\n return 1\n"),
                ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
                _configuration(tmp_path),
            )
        )
    after = frozenset(temporary_root.glob("rl-unit-test-inline-*"))
    assert after == before


def test_output_cannot_be_inside_user_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("value=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output"):
        ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                LocalProjectDirectory(project),
                configuration=ExternalAnalysisConfiguration(output_root=project / "output"),
            )
        )


@pytest.mark.parametrize(
    "payload,category",
    (
        (b"x = 'too large'\n", "SOURCE_LIMIT_EXCEEDED"),
        (b"# coding: unknown-codec\nvalue=1\n", "UNSUPPORTED_ENCODING"),
        (b"def broken(:\n", "SYNTAX_ERROR"),
    ),
)
def test_uploaded_payload_validation_is_controlled(
    tmp_path: Path,
    payload: bytes,
    category: str,
) -> None:
    maximum = 4 if category == "SOURCE_LIMIT_EXCEEDED" else 100
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile("module.py", payload),
            configuration=_configuration(tmp_path, maximum_payload_bytes=maximum),
        )
    )
    assert result.status is ExternalAnalysisStatus.FAILED
    assert category in result.issues


def test_src_layout_import_root_is_forwarded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "src" / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    orchestrator = Mock()
    orchestrator.run.return_value = Mock(
        status=ProjectRunStatus.COMPLETED,
        function_results=(),
        discovered_targets=(),
        report_path=tmp_path / "missing.json",
    )
    selection = ExternalModuleSelection(
        ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES,
        ("package.module",),
    )
    ExternalSourceAnalysisService(orchestrator_factory=lambda: orchestrator).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, module_selection=selection),
        )
    )
    assert orchestrator.run.call_args.kwargs["import_root"] == (project / "src").resolve()


def test_missing_dependency_is_controlled_per_module(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    orchestrator = Mock()
    orchestrator.run.side_effect = ModuleNotFoundError("private_dependency_name")
    result = ExternalSourceAnalysisService(orchestrator_factory=lambda: orchestrator).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path),
        )
    )
    assert result.status is ExternalAnalysisStatus.FAILED
    assert result.module_results[0].issue_category == "DEPENDENCY_UNAVAILABLE"
    assert "private_dependency_name" not in result.report_path.read_text(encoding="utf-8")


def test_import_failure_does_not_erase_later_module_result(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("a.py", "b.py"):
        (project / name).write_text("def target():\n return 1\n", encoding="utf-8")
    orchestrators = []
    for outcome in (
        ImportError("unsafe import detail"),
        Mock(status=ProjectRunStatus.COMPLETED, function_results=(), discovered_targets=(), report_path=tmp_path / "none"),
    ):
        orchestrator = Mock()
        orchestrator.run.side_effect = outcome if isinstance(outcome, Exception) else None
        if not isinstance(outcome, Exception):
            orchestrator.run.return_value = outcome
        orchestrators.append(orchestrator)
    result = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrators.pop(0)
    ).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path),
        )
    )
    assert result.status is ExternalAnalysisStatus.PARTIAL
    assert [item.status for item in result.module_results] == [
        ExternalModuleStatus.FAILED,
        ExternalModuleStatus.COMPLETED,
    ]


def test_timeout_status_is_preserved_without_project_average(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("def target():\n return 1\n", encoding="utf-8")
    orchestrator = Mock()
    orchestrator.run.return_value = Mock(
        status=ProjectRunStatus.TIMED_OUT,
        function_results=(),
        discovered_targets=(),
        report_path=tmp_path / "none",
    )
    result = ExternalSourceAnalysisService(orchestrator_factory=lambda: orchestrator).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path),
        )
    )
    assert result.status is ExternalAnalysisStatus.TIMED_OUT
    assert result.to_dict()["aggregate_project_coverage"]["line_percent"] is None


def test_cleanup_failure_is_reported_without_source_leak(tmp_path: Path) -> None:
    temporary_root = Path(os.environ["TEMP"])
    before = frozenset(temporary_root.glob("rl-unit-test-inline-*"))
    with patch.object(
        ExternalSourceAnalysisService,
        "_remove_temporary_workspace",
        side_effect=OSError("private cleanup path"),
    ):
        result = ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("secret_marker = 1\n"),
                configuration=_configuration(tmp_path),
            )
        )
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.FAILED
    report = result.report_path.read_text(encoding="utf-8")
    assert "CLEANUP_FAILED" in report
    assert "secret_marker" not in report
    assert "private cleanup path" not in report
    created = frozenset(temporary_root.glob("rl-unit-test-inline-*")) - before
    assert len(created) == 1
    ExternalSourceAnalysisService._remove_temporary_workspace(next(iter(created)))


def test_parallel_inline_workspaces_and_outputs_are_isolated(tmp_path: Path) -> None:
    def run(ordinal: int):
        return ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("value = 1\n"),
                configuration=ExternalAnalysisConfiguration(
                    output_root=tmp_path / f"output_{ordinal}"
                ),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = tuple(executor.map(run, (1, 2)))
    assert first.report_path != second.report_path
    assert first.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert second.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED


def test_external_report_is_written_atomically(tmp_path: Path) -> None:
    with patch("services.external_source_analysis_service.os.replace", wraps=os.replace) as replace_call:
        result = ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("value = 1\n"),
                configuration=_configuration(tmp_path),
            )
        )
    assert result.report_path.is_file()
    assert replace_call.call_count >= 1
    assert not tuple(result.output_root.glob("*.tmp"))
