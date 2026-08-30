from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from analyzer import python_source_reader
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
from models.project_analysis_result import (
    ProjectRunStatus,
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)
from services.external_source_analysis_service import ExternalSourceAnalysisService
from services.source_acquisition_service import SourceAcquisitionService


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


def test_normal_finalization_attempts_external_workspace_cleanup_exactly_once(
    tmp_path: Path,
) -> None:
    service = ExternalSourceAnalysisService()
    with patch.object(service, "_cleanup", wraps=service._cleanup) as cleanup:
        result = service.run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("def target():\n    return 1\n"),
                configuration=_configuration(tmp_path),
            )
        )

    assert result.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert cleanup.call_count == 1


def test_unexpected_external_failure_still_attempts_cleanup_exactly_once(
    tmp_path: Path,
) -> None:
    service = ExternalSourceAnalysisService()
    with (
        patch.object(
            service,
            "_target_inventories",
            side_effect=RuntimeError("private worker detail"),
        ),
        patch.object(service, "_cleanup", wraps=service._cleanup) as cleanup,
    ):
        with pytest.raises(RuntimeError, match="private worker detail"):
            service.run(
                ExternalSourceAnalysisRequest(
                    InlinePythonSource("def target():\n    return 1\n"),
                    configuration=_configuration(tmp_path),
                )
            )

    assert cleanup.call_count == 1


def test_inline_source_canonicalization_is_idempotent_and_repeated_bom_safe() -> None:
    canonicalize = python_source_reader.canonicalize_inline_python_source
    source = "\ufeffdef hedef():\r\n    return 'Türkçe'\r\n"

    canonical = canonicalize(source)

    assert canonical == "def hedef():\r\n    return 'Türkçe'\r\n"
    assert canonicalize(canonical) == canonical
    for count in (2, 3):
        with pytest.raises(ValueError, match="BOM"):
            canonicalize("\ufeff" * count + "value = 1\n")
    middle = "value = '\ufeff'\n"
    assert canonicalize(middle) == middle
    prefixed = " \ufeffvalue = 1\n"
    assert canonicalize(prefixed) == prefixed


@pytest.mark.parametrize("bom_count", (2, 3))
def test_direct_inline_service_rejects_repeated_bom_as_syntax_error(
    tmp_path: Path,
    bom_count: int,
) -> None:
    marker = "never_publish_repeated_bom_source"
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(
                "\ufeff" * bom_count
                + f"def target():\n    return {marker!r}\n"
            ),
            configuration=_configuration(tmp_path),
        )
    )

    assert result.status is ExternalAnalysisStatus.FAILED
    assert result.issues == ("SYNTAX_ERROR",)
    assert marker not in result.report_path.read_text(encoding="utf-8")


def test_inline_payload_limit_uses_raw_transport_size_before_bom_normalization(
    tmp_path: Path,
) -> None:
    source = InlinePythonSource("\ufeffvalue = 1\n")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            source,
            configuration=_configuration(
                tmp_path,
                maximum_payload_bytes=len(source.source_text.encode("utf-8")),
            ),
        )
    )

    assert result.status is ExternalAnalysisStatus.FAILED
    assert result.issues == ("SOURCE_LIMIT_EXCEEDED",)


def test_inline_service_does_not_decode_canonical_source_again(
    tmp_path: Path,
) -> None:
    with patch(
        "services.external_source_analysis_service.decode_python_source_bytes",
        side_effect=AssertionError("inline source yeniden decode edildi"),
    ):
        result = ExternalSourceAnalysisService().run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("\ufeffdef target():\n    return 1\n"),
                configuration=_configuration(tmp_path),
            )
        )

    assert result.status is ExternalAnalysisStatus.STATIC_COMPLETED


def test_static_discovery_preserves_qualified_method_inventory_with_selection(
    tmp_path: Path,
) -> None:
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("inline_source", "Engine.start"),),
    )
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(
                "class Engine:\n"
                "    def start(self, enabled: bool) -> bool:\n"
                "        return enabled\n\n"
                "def helper() -> int:\n"
                "    return 1\n"
            ),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            _configuration(tmp_path, target_selection=selection),
        )
    )

    assert result.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert result.module_results[0].discovered_function_names == (
        "Engine.start",
        "helper",
    )


def test_dynamic_forwards_explicit_target_selection_separately_from_module_selection(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "alpha.py").write_text(
        "class Shared:\n"
        "    def run(self, value: int) -> int:\n"
        "        return value\n",
        encoding="utf-8",
    )
    orchestrator = Mock()
    project_result = Mock(
        status=ProjectRunStatus.COMPLETED,
        function_results=(),
        discovered_targets=(),
        report_path=tmp_path / "report.json",
    )
    orchestrator.run.return_value = project_result
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("alpha", "Shared.run"),),
    )

    ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator
    ).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, target_selection=selection),
        )
    )

    call = orchestrator.run.call_args.kwargs
    assert call["all_functions"] is False
    assert call["function_name"] is None
    assert call["target_selection"].for_module("alpha") == ("Shared.run",)


def test_same_qualified_target_in_different_modules_keeps_selector_identity(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for module_name, literal in (("first_module", 5), ("second_module", 11)):
        (project / f"{module_name}.py").write_text(
            "class Shared:\n"
            "    def run(self, value: int) -> int:\n"
            f"        return value + {literal}\n",
            encoding="utf-8",
        )
    orchestrator = Mock()
    orchestrator.run.return_value = Mock(
        status=ProjectRunStatus.COMPLETED,
        function_results=(),
        discovered_targets=(),
        report_path=tmp_path / "report.json",
    )
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (
            QualifiedTargetSelector("first_module", "Shared.run"),
            QualifiedTargetSelector("second_module", "Shared.run"),
        ),
    )

    ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator
    ).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, target_selection=selection),
        )
    )

    calls = {
        call.kwargs["module_path"]: call.kwargs["target_selection"]
        for call in orchestrator.run.call_args_list
    }
    assert calls["first_module"].for_module("first_module") == ("Shared.run",)
    assert calls["second_module"].for_module("second_module") == ("Shared.run",)


def test_unknown_external_target_is_controlled_before_dynamic_orchestrator(
    tmp_path: Path,
) -> None:
    orchestrator = Mock()
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("inline_source", "Missing.run"),),
    )

    result = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator
    ).run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def available() -> int:\n    return 1\n"),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, target_selection=selection),
        )
    )

    assert result.status is ExternalAnalysisStatus.FAILED
    assert result.issues == ("UNKNOWN_TARGET_SELECTION",)
    orchestrator.run.assert_not_called()
    assert "available" not in result.report_path.read_text(encoding="utf-8")


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
    assert result.module_results[0].discovered_function_names == (
        "first",
        "second",
        "third",
    )
    assert result.module_results[0].analyzed_function_count == 0
    assert result.module_results[0].limit_skipped_function_count == 0


def test_successful_github_acquisition_without_python_is_partial_and_never_executes(
    tmp_path: Path,
) -> None:
    def runner(
        arguments: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "README.md").write_text("public fixture\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "a" * 40, "")

    acquisition = SourceAcquisitionService(
        subprocess_runner=runner,
        git_executable="git",
    )
    orchestrator_factory = Mock()
    result = ExternalSourceAnalysisService(
        acquisition_service=acquisition,
        orchestrator_factory=orchestrator_factory,
    ).run(
        ExternalSourceAnalysisRequest(
            PublicGitHubRepository("https://github.com/owner/repository"),
            configuration=_configuration(tmp_path),
        )
    )

    assert result.status is ExternalAnalysisStatus.PARTIAL
    assert result.acquisition_status == "PARTIAL"
    assert result.discovered_module_count == 0
    assert result.discovered_function_count == 0
    assert result.module_results == ()
    assert result.to_dict()["aggregate_project_coverage"] == {
        "line_percent": None,
        "branch_percent": None,
        "status": "UNMEASURED",
    }
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert result.issues == ("NO_PYTHON_FILES",)
    orchestrator_factory.assert_not_called()


def test_github_clone_failure_remains_failed(tmp_path: Path) -> None:
    runner = Mock(
        return_value=subprocess.CompletedProcess(("git",), 128, "", "private detail")
    )
    result = ExternalSourceAnalysisService(
        acquisition_service=SourceAcquisitionService(
            subprocess_runner=runner,
            git_executable="git",
        )
    ).run(
        ExternalSourceAnalysisRequest(
            PublicGitHubRepository("https://github.com/owner/repository"),
            configuration=_configuration(tmp_path),
        )
    )

    assert result.status is ExternalAnalysisStatus.FAILED
    assert result.issues == ("CLONE_FAILED",)


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
    assert result.status is ExternalAnalysisStatus.FAILED
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


class _ExternalDeadlineClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.mark.parametrize(
    "value",
    (True, False, 0, -1, float("nan"), float("inf"), -float("inf")),
)
def test_external_configuration_rejects_invalid_project_timeout(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="project_timeout_seconds"):
        _configuration(tmp_path, project_timeout_seconds=value)


def test_project_deadline_none_preserves_unlimited_external_behavior(
    tmp_path: Path,
) -> None:
    orchestrator = Mock()
    orchestrator.run.return_value = Mock(
        status=ProjectRunStatus.COMPLETED,
        function_results=(),
        discovered_targets=(),
        coverage_candidates=(),
        report_path=tmp_path / "none.json",
    )

    result = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator
    ).run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def target(value: int) -> int:\n    return value\n"),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, project_timeout_seconds=None),
        )
    )

    assert result.status is ExternalAnalysisStatus.COMPLETED
    assert result.project_timeout_seconds is None
    assert result.project_deadline_exceeded is False
    assert orchestrator.run.call_args.kwargs["project_deadline"].timeout_seconds is None


def test_project_deadline_before_acquisition_returns_controlled_timeout(
    tmp_path: Path,
) -> None:
    values = iter((0.0, 2.0, 2.0, 2.0, 2.0, 2.0))
    acquisition = Mock()

    result = ExternalSourceAnalysisService(
        acquisition_service=acquisition,
        clock=lambda: next(values, 2.0),
    ).run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def target() -> int:\n    return 1\n"),
            configuration=_configuration(tmp_path, project_timeout_seconds=1.0),
        )
    )

    acquisition.resolve.assert_not_called()
    assert result.status is ExternalAnalysisStatus.TIMED_OUT
    assert result.acquisition_status == "NOT_STARTED"
    assert result.project_deadline_exceeded is True
    assert result.deadline_stage == "SOURCE_PREPARATION"
    assert result.project_coverage is None
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.NOT_REQUIRED


def test_project_deadline_after_discovery_preserves_inventory_and_cleans(
    tmp_path: Path,
) -> None:
    clock = _ExternalDeadlineClock()
    real_acquisition = SourceAcquisitionService()
    acquisition = Mock()

    def resolve(request, *, project_deadline=None):
        acquired = real_acquisition.resolve(
            request,
            project_deadline=project_deadline,
        )
        clock.value = 2.0
        return acquired

    acquisition.resolve.side_effect = resolve
    result = ExternalSourceAnalysisService(
        acquisition_service=acquisition,
        clock=clock,
    ).run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(
                "def first() -> int:\n    return 1\n\n"
                "def second() -> int:\n    return 2\n"
            ),
            configuration=_configuration(tmp_path, project_timeout_seconds=1.0),
        )
    )

    assert result.status is ExternalAnalysisStatus.TIMED_OUT
    assert result.discovered_module_count == 1
    assert result.discovered_function_count == 2
    assert result.deadline_skipped_function_count == 2
    assert result.module_results[0].status is ExternalModuleStatus.SKIPPED_DEADLINE
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert result.to_dict()["aggregate_project_coverage"]["status"] == "UNMEASURED"


def test_project_deadline_preserves_completed_module_and_skips_later_module(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("alpha.py", "beta.py"):
        (project / name).write_text(
            "def target(value: int) -> int:\n    return value\n",
            encoding="utf-8",
        )
    clock = _ExternalDeadlineClock()
    orchestrator = Mock()

    def run(**kwargs):
        clock.value = 3.0
        return Mock(
            status=ProjectRunStatus.COMPLETED,
            function_results=(),
            discovered_targets=(),
            coverage_candidates=(),
            report_path=tmp_path / "none.json",
        )

    orchestrator.run.side_effect = run
    result = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator,
        clock=clock,
    ).run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(tmp_path, project_timeout_seconds=2.0),
        )
    )

    assert [item.status for item in result.module_results] == [
        ExternalModuleStatus.COMPLETED,
        ExternalModuleStatus.SKIPPED_DEADLINE,
    ]
    assert result.status is ExternalAnalysisStatus.PARTIAL
    assert result.project_deadline_exceeded is True
    assert orchestrator.run.call_count == 1
    assert result.project_coverage is None


def test_project_coverage_receives_remaining_project_budget(
    tmp_path: Path,
) -> None:
    clock = _ExternalDeadlineClock()
    orchestrator = Mock()
    orchestrator.run.return_value = Mock(
        status=ProjectRunStatus.COMPLETED,
        function_results=(),
        discovered_targets=(),
        coverage_candidates=(),
        report_path=tmp_path / "none.json",
    )
    coverage_service = Mock()
    coverage_service.measure_and_minimize.return_value = None

    result = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: orchestrator,
        project_coverage_service=coverage_service,
        clock=clock,
    ).run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def target() -> int:\n    return 1\n"),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _configuration(
                tmp_path,
                project_timeout_seconds=5.0,
                pytest_coverage_timeout_seconds=30.0,
            ),
        )
    )

    assert result.status is ExternalAnalysisStatus.COMPLETED
    assert coverage_service.measure_and_minimize.call_args.kwargs[
        "timeout_seconds"
    ] == pytest.approx(5.0)


def test_public_clone_timeout_is_clamped_by_remaining_project_budget(
    tmp_path: Path,
) -> None:
    acquisition = Mock()
    acquisition.resolve.side_effect = RuntimeError("stop after request capture")

    with pytest.raises(RuntimeError, match="request capture"):
        ExternalSourceAnalysisService(acquisition_service=acquisition).run(
            ExternalSourceAnalysisRequest(
                PublicGitHubRepository("https://github.com/owner/repository"),
                configuration=_configuration(
                    tmp_path,
                    project_timeout_seconds=4.0,
                ),
            )
        )

    source_request = acquisition.resolve.call_args.args[0]
    assert 0.0 < source_request.limits.clone_timeout_seconds <= 4.0
    assert acquisition.resolve.call_args.kwargs["project_deadline"].timeout_seconds == 4.0


def test_deadline_state_is_not_reused_by_later_external_run(tmp_path: Path) -> None:
    clock = _ExternalDeadlineClock()
    service = ExternalSourceAnalysisService(clock=clock)
    first = service.run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def first() -> int:\n    return 1\n"),
            configuration=_configuration(
                tmp_path / "first",
                project_timeout_seconds=0.5,
            ),
        )
    )
    clock.value = 100.0
    second = service.run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource("def second() -> int:\n    return 2\n"),
            configuration=_configuration(
                tmp_path / "second",
                project_timeout_seconds=5.0,
            ),
        )
    )

    assert first.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert second.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert first.project_deadline_exceeded is False
    assert second.project_deadline_exceeded is False


def test_deadline_crossed_during_cooperative_finalization_preserves_report_and_cleanup(
    tmp_path: Path,
) -> None:
    clock = _ExternalDeadlineClock()
    service = ExternalSourceAnalysisService(clock=clock)
    real_cleanup = service._cleanup

    def cleanup(acquired, temporary_workspace):
        status = real_cleanup(acquired, temporary_workspace)
        clock.value = 2.0
        return status

    with patch.object(service, "_cleanup", side_effect=cleanup):
        result = service.run(
            ExternalSourceAnalysisRequest(
                InlinePythonSource("def target() -> int:\n    return 1\n"),
                configuration=_configuration(
                    tmp_path,
                    project_timeout_seconds=1.0,
                ),
            )
        )

    assert result.status is ExternalAnalysisStatus.PARTIAL
    assert result.project_deadline_exceeded is True
    assert result.deadline_stage == "REPORT_FINALIZATION_OR_CLEANUP"
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert result.report_path.is_file()
    assert not tuple(result.output_root.glob("*.tmp"))
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["project_deadline_exceeded"] is True
    assert payload["aggregate_project_coverage"]["line_percent"] is None
