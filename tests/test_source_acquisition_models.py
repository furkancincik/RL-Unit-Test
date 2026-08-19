from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from models.source_acquisition_result import (
    DiscoveredPythonModule,
    SourceAcquisitionLimits,
    SourceAcquisitionRequest,
    SourceAcquisitionStatus,
    SourceDiscoveryIssue,
    SourceIssueCategory,
    SourceTargetKind,
    SourceWorkspaceOwnership,
    ResolvedSourceTarget,
)


def _module() -> DiscoveredPythonModule:
    return DiscoveredPythonModule(
        relative_path="package/service.py",
        file_size_bytes=24,
        encoding="utf-8",
        syntax_valid=True,
        top_level_function_count=1,
        module_path="package.service",
        module_path_candidates=("package.service",),
        package_root=".",
        supported=True,
    )


def test_source_models_are_immutable_and_json_safe(tmp_path: Path) -> None:
    result = ResolvedSourceTarget(
        source_kind=SourceTargetKind.LOCAL_DIRECTORY,
        normalized_origin=str(tmp_path),
        resolved_project_root=tmp_path,
        workspace_ownership=SourceWorkspaceOwnership.USER_OWNED,
        repository_name=tmp_path.name,
        github_owner=None,
        github_repository=None,
        requested_ref=None,
        resolved_commit_sha=None,
        status=SourceAcquisitionStatus.COMPLETED,
        discovered_modules=(_module(),),
        issues=(),
        total_scanned_bytes=24,
        duration_seconds=0.1,
        cleanup_required=False,
        include_tests=False,
        limits=SourceAcquisitionLimits(),
    )

    with pytest.raises(FrozenInstanceError):
        result.status = SourceAcquisitionStatus.FAILED  # type: ignore[misc]

    payload = result.to_dict()
    serialized = json.dumps(payload)
    assert payload["python_file_count"] == 1
    assert payload["project_root"] == "."
    assert "def " not in serialized
    assert "traceback" not in serialized.lower()
    assert "environment" not in serialized.lower()


@pytest.mark.parametrize(
    "field,value",
    (
        ("clone_timeout_seconds", True),
        ("clone_timeout_seconds", 0),
        ("maximum_repository_bytes", -1),
        ("maximum_python_file_count", 0),
        ("maximum_single_file_bytes", float("inf")),
        ("maximum_total_python_bytes", float("nan")),
        ("maximum_path_depth", False),
    ),
)
def test_limits_reject_invalid_values(field: str, value: object) -> None:
    values = SourceAcquisitionLimits().__dict__ if hasattr(SourceAcquisitionLimits(), "__dict__") else {
        "clone_timeout_seconds": 30.0,
        "maximum_repository_bytes": 100_000_000,
        "maximum_python_file_count": 10_000,
        "maximum_single_file_bytes": 2_000_000,
        "maximum_total_python_bytes": 20_000_000,
        "maximum_path_depth": 30,
    }
    values = dict(values)
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        SourceAcquisitionLimits(**values)


def test_request_validates_types_without_parsing_origin() -> None:
    request = SourceAcquisitionRequest(
        source_kind=SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
        origin="https://github.com/owner/repository",
        ref="main",
        include_tests=True,
    )
    assert request.include_tests is True
    with pytest.raises(TypeError):
        SourceAcquisitionRequest(
            source_kind=SourceTargetKind.LOCAL_FILE,
            origin="target.py",
            include_tests=1,  # type: ignore[arg-type]
        )


def test_issue_serialization_does_not_include_exception_details() -> None:
    issue = SourceDiscoveryIssue(
        category=SourceIssueCategory.SYNTAX_ERROR,
        message="Python syntax validation failed.",
        relative_path="broken.py",
    )
    assert issue.to_dict() == {
        "category": "SYNTAX_ERROR",
        "message": "Python syntax validation failed.",
        "relative_path": "broken.py",
    }
