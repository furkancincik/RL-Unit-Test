from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalModuleStatus,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalSourceKind,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    LocalProjectDirectory,
    PublicGitHubRepository,
    UploadedPythonFile,
)
from models.project_analysis_result import (
    FunctionAnalysisResult,
    FunctionRunStatus,
    FunctionSelectionMode,
    FunctionTarget,
    ProjectAnalysisResult,
    ProjectRunStatus,
)


def _configuration(tmp_path: Path) -> ExternalAnalysisConfiguration:
    return ExternalAnalysisConfiguration(output_root=tmp_path / "output")


def test_execution_policy_and_request_are_immutable_and_json_safe(tmp_path: Path) -> None:
    request = ExternalSourceAnalysisRequest(
        source=InlinePythonSource("def target():\n    return 1\n", "target.py"),
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        configuration=_configuration(tmp_path),
    )
    with pytest.raises(FrozenInstanceError):
        request.execution_policy = ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS  # type: ignore[misc]
    payload = request.to_dict()
    serialized = json.dumps(payload)
    assert payload["source_kind"] == "INLINE_PYTHON_SOURCE"
    assert "def target" not in serialized


@pytest.mark.parametrize("source", ("", "   "))
def test_inline_request_rejects_empty_source(source: str) -> None:
    with pytest.raises(ValueError, match="source_text"):
        InlinePythonSource(source)


def test_uploaded_payload_is_not_serialized(tmp_path: Path) -> None:
    request = ExternalSourceAnalysisRequest(
        source=UploadedPythonFile("target.py", b"secret_payload = 1\n"),
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        configuration=_configuration(tmp_path),
    )
    assert "secret_payload" not in json.dumps(request.to_dict())


@pytest.mark.parametrize(
    "source,kind",
    (
        (InlinePythonSource("value=1\n"), ExternalSourceKind.INLINE_PYTHON_SOURCE),
        (UploadedPythonFile("value.py", b"value=1\n"), ExternalSourceKind.UPLOADED_PYTHON_FILE),
        (LocalProjectDirectory(Path("project")), ExternalSourceKind.LOCAL_PROJECT_DIRECTORY),
        (PublicGitHubRepository("https://github.com/owner/repo"), ExternalSourceKind.PUBLIC_GITHUB_REPOSITORY),
    ),
)
def test_four_source_models_remain_distinct(source: object, kind: ExternalSourceKind) -> None:
    assert source.source_kind is kind  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mode,values",
    (
        (ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT, ()),
        (ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS, ("package/service.py",)),
        (ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES, ("package.service",)),
    ),
)
def test_module_selection_modes(mode: ExternalModuleSelectionMode, values: tuple[str, ...]) -> None:
    selection = ExternalModuleSelection(mode=mode, values=values)
    assert selection.mode is mode


@pytest.mark.parametrize("field,value", (("maximum_selected_modules", 0), ("maximum_functions_per_module", True)))
def test_configuration_rejects_invalid_limits(tmp_path: Path, field: str, value: object) -> None:
    arguments = {"output_root": tmp_path / "out", field: value}
    with pytest.raises((TypeError, ValueError)):
        ExternalAnalysisConfiguration(**arguments)  # type: ignore[arg-type]


def test_result_serialization_uses_none_for_unmeasured_dynamic_fields(tmp_path: Path) -> None:
    module = ExternalModuleAnalysisResult(
        relative_path="module.py",
        module_name="module",
        status=ExternalModuleStatus.STATIC_ONLY,
        discovered_function_count=1,
        project_result=None,
        issue_category=None,
        issue_message=None,
        artifact_paths=(),
    )
    result = ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.INLINE_PYTHON_SOURCE,
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        status=ExternalAnalysisStatus.STATIC_COMPLETED,
        acquisition_status="COMPLETED",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
        output_root=tmp_path,
        report_path=tmp_path / "external_analysis_report.json",
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
        issues=(),
    )
    payload = result.to_dict()
    assert payload["modules"][0]["line_coverage_percent"] is None
    assert payload["modules"][0]["branch_coverage_percent"] is None
    assert payload["modules"][0]["greedy"] is None
    assert payload["modules"][0]["strategy_comparison"] is None


def test_external_json_preserves_function_limit_visibility(tmp_path: Path) -> None:
    targets = tuple(
        FunctionTarget(
            name=name,
            qualified_name=name,
            start_line=index,
            end_line=index,
            parameters=(),
            return_annotation=None,
            is_async=False,
            is_nested=False,
            is_method=False,
            is_supported=True,
        )
        for index, name in enumerate(("first", "second", "third"), start=1)
    )
    functions = (
        FunctionAnalysisResult(
            target=targets[0],
            status=FunctionRunStatus.COMPLETED,
            diagnostic=None,
            output_directory=tmp_path / "first",
        ),
        FunctionAnalysisResult(
            target=targets[1],
            status=FunctionRunStatus.COMPLETED,
            diagnostic=None,
            output_directory=tmp_path / "second",
        ),
        FunctionAnalysisResult(
            target=targets[2],
            status=FunctionRunStatus.SKIPPED_LIMIT,
            diagnostic=None,
            output_directory=tmp_path / "third",
            skip_reason="FUNCTION_LIMIT_EXCEEDED",
        ),
    )
    project = ProjectAnalysisResult(
        source_file=tmp_path / "module.py",
        module_path="module",
        selection_mode=FunctionSelectionMode.ALL,
        discovered_targets=targets,
        function_results=functions,
        total_duration_seconds=0.1,
        status=ProjectRunStatus.PARTIAL,
        output_root=tmp_path,
        report_path=tmp_path / "project_analysis_report.json",
    )
    module = ExternalModuleAnalysisResult(
        relative_path="module.py",
        module_name="module",
        status=ExternalModuleStatus.PARTIAL,
        discovered_function_count=3,
        project_result=project,
        issue_category=None,
        issue_message=None,
        artifact_paths=(),
    )
    result = ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.LOCAL_PROJECT_DIRECTORY,
        execution_policy=ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
        status=ExternalAnalysisStatus.PARTIAL,
        acquisition_status="COMPLETED",
        repository_name="project",
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
        output_root=tmp_path,
        report_path=tmp_path / "external_analysis_report.json",
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
        issues=(),
    )

    payload = result.to_dict()

    assert payload["discovered_function_count"] == 3
    assert payload["analyzed_function_count"] == 2
    assert payload["limit_skipped_function_count"] == 1
    assert payload["modules"][0]["limit_skipped_function_count"] == 1
    assert payload["modules"][0]["functions"][-1]["status"] == "SKIPPED_LIMIT"
    assert payload["modules"][0]["functions"][-1]["skip_reason"] == "FUNCTION_LIMIT_EXCEEDED"
