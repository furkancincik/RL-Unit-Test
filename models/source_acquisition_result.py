from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class SourceTargetKind(str, Enum):
    LOCAL_FILE = "LOCAL_FILE"
    LOCAL_DIRECTORY = "LOCAL_DIRECTORY"
    PUBLIC_GITHUB_REPOSITORY = "PUBLIC_GITHUB_REPOSITORY"


class SourceWorkspaceOwnership(str, Enum):
    USER_OWNED = "USER_OWNED"
    TOOL_TEMPORARY = "TOOL_TEMPORARY"


class SourceAcquisitionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"


class SourceIssueCategory(str, Enum):
    INVALID_SOURCE = "INVALID_SOURCE"
    INVALID_GITHUB_URL = "INVALID_GITHUB_URL"
    GIT_NOT_AVAILABLE = "GIT_NOT_AVAILABLE"
    CLONE_TIMEOUT = "CLONE_TIMEOUT"
    CLONE_FAILED = "CLONE_FAILED"
    REPOSITORY_LIMIT_EXCEEDED = "REPOSITORY_LIMIT_EXCEEDED"
    FILE_LIMIT_EXCEEDED = "FILE_LIMIT_EXCEEDED"
    PATH_DEPTH_EXCEEDED = "PATH_DEPTH_EXCEEDED"
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    SYMLINK_SKIPPED = "SYMLINK_SKIPPED"
    UNSUPPORTED_ENCODING = "UNSUPPORTED_ENCODING"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    MODULE_PATH_UNRESOLVED = "MODULE_PATH_UNRESOLVED"
    MODULE_PATH_AMBIGUOUS = "MODULE_PATH_AMBIGUOUS"
    NO_PYTHON_FILES = "NO_PYTHON_FILES"
    CLEANUP_FAILED = "CLEANUP_FAILED"


@dataclass(frozen=True, slots=True)
class SourceAcquisitionLimits:
    clone_timeout_seconds: float = 30.0
    maximum_repository_bytes: int = 100_000_000
    maximum_python_file_count: int = 10_000
    maximum_single_file_bytes: int = 2_000_000
    maximum_total_python_bytes: int = 20_000_000
    maximum_path_depth: int = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.clone_timeout_seconds, bool)
            or not isinstance(self.clone_timeout_seconds, (int, float))
        ):
            raise TypeError("clone_timeout_seconds sayısal olmalıdır.")
        if not math.isfinite(float(self.clone_timeout_seconds)) or (
            float(self.clone_timeout_seconds) <= 0.0
        ):
            raise ValueError("clone_timeout_seconds pozitif ve sonlu olmalıdır.")
        for name in (
            "maximum_repository_bytes",
            "maximum_python_file_count",
            "maximum_single_file_bytes",
            "maximum_total_python_bytes",
            "maximum_path_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} tam sayı olmalıdır.")
            if value <= 0:
                raise ValueError(f"{name} pozitif olmalıdır.")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "clone_timeout_seconds": float(self.clone_timeout_seconds),
            "maximum_repository_bytes": self.maximum_repository_bytes,
            "maximum_python_file_count": self.maximum_python_file_count,
            "maximum_single_file_bytes": self.maximum_single_file_bytes,
            "maximum_total_python_bytes": self.maximum_total_python_bytes,
            "maximum_path_depth": self.maximum_path_depth,
        }


@dataclass(frozen=True, slots=True)
class SourceAcquisitionRequest:
    source_kind: SourceTargetKind
    origin: str
    ref: str | None = None
    include_tests: bool = False
    limits: SourceAcquisitionLimits = SourceAcquisitionLimits()

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, SourceTargetKind):
            raise TypeError("source_kind geçersiz.")
        if not isinstance(self.origin, str):
            raise TypeError("origin string olmalıdır.")
        if self.ref is not None and not isinstance(self.ref, str):
            raise TypeError("ref string veya None olmalıdır.")
        if not isinstance(self.include_tests, bool):
            raise TypeError("include_tests bool olmalıdır.")
        if not isinstance(self.limits, SourceAcquisitionLimits):
            raise TypeError("limits SourceAcquisitionLimits olmalıdır.")


@dataclass(frozen=True, slots=True)
class SourceDiscoveryIssue:
    category: SourceIssueCategory
    message: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, SourceIssueCategory):
            raise TypeError("issue category geçersiz.")
        if not isinstance(self.message, str) or not self.message:
            raise TypeError("issue message boş olmayan string olmalıdır.")
        if self.relative_path is not None and (
            not isinstance(self.relative_path, str) or not self.relative_path
        ):
            raise TypeError("relative_path string veya None olmalıdır.")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "category": self.category.value,
            "message": self.message,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True, slots=True)
class DiscoveredPythonModule:
    relative_path: str
    file_size_bytes: int
    encoding: str | None
    syntax_valid: bool
    top_level_function_count: int
    module_path: str | None
    module_path_candidates: tuple[str, ...]
    package_root: str | None
    supported: bool

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise TypeError("relative_path boş olmayan string olmalıdır.")
        for name in ("file_size_bytes", "top_level_function_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if not isinstance(self.syntax_valid, bool) or not isinstance(self.supported, bool):
            raise TypeError("syntax_valid ve supported bool olmalıdır.")
        if not isinstance(self.module_path_candidates, tuple) or any(
            not isinstance(value, str) or not value
            for value in self.module_path_candidates
        ):
            raise TypeError("module_path_candidates string tuple'ı olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "file_size_bytes": self.file_size_bytes,
            "encoding": self.encoding,
            "syntax_valid": self.syntax_valid,
            "top_level_function_count": self.top_level_function_count,
            "module_path": self.module_path,
            "module_path_candidates": list(self.module_path_candidates),
            "package_root": self.package_root,
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSourceTarget:
    source_kind: SourceTargetKind
    normalized_origin: str
    resolved_project_root: Path
    workspace_ownership: SourceWorkspaceOwnership
    repository_name: str | None
    github_owner: str | None
    github_repository: str | None
    requested_ref: str | None
    resolved_commit_sha: str | None
    status: SourceAcquisitionStatus
    discovered_modules: tuple[DiscoveredPythonModule, ...]
    issues: tuple[SourceDiscoveryIssue, ...]
    total_scanned_bytes: int
    duration_seconds: float
    cleanup_required: bool
    include_tests: bool
    limits: SourceAcquisitionLimits

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, SourceTargetKind):
            raise TypeError("source_kind geçersiz.")
        if not isinstance(self.resolved_project_root, Path):
            raise TypeError("resolved_project_root Path olmalıdır.")
        if not isinstance(self.workspace_ownership, SourceWorkspaceOwnership):
            raise TypeError("workspace_ownership geçersiz.")
        if not isinstance(self.status, SourceAcquisitionStatus):
            raise TypeError("status geçersiz.")
        if isinstance(self.total_scanned_bytes, bool) or not isinstance(
            self.total_scanned_bytes, int
        ) or self.total_scanned_bytes < 0:
            raise ValueError("total_scanned_bytes negatif olmayan tam sayı olmalıdır.")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise TypeError("duration_seconds sayısal olmalıdır.")
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0:
            raise ValueError("duration_seconds negatif olmayan sonlu sayı olmalıdır.")
        if not isinstance(self.cleanup_required, bool) or not isinstance(
            self.include_tests, bool
        ):
            raise TypeError("cleanup_required/include_tests bool olmalıdır.")
        if self.cleanup_required and (
            self.workspace_ownership is not SourceWorkspaceOwnership.TOOL_TEMPORARY
        ):
            raise ValueError("Yalnız tool-owned workspace cleanup gerektirebilir.")

    @property
    def python_file_count(self) -> int:
        return len(self.discovered_modules)

    @property
    def is_available(self) -> bool:
        return self.status in {
            SourceAcquisitionStatus.COMPLETED,
            SourceAcquisitionStatus.PARTIAL,
        } and self.resolved_project_root.exists()

    def _public_origin(self) -> str:
        if self.source_kind is SourceTargetKind.PUBLIC_GITHUB_REPOSITORY:
            return self.normalized_origin
        return Path(self.normalized_origin).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source_kind": self.source_kind.value,
            "normalized_origin": self._public_origin(),
            "project_root": ".",
            "workspace_ownership": self.workspace_ownership.value,
            "repository_name": self.repository_name,
            "github_owner": self.github_owner,
            "github_repository": self.github_repository,
            "requested_ref": self.requested_ref,
            "resolved_commit_sha": self.resolved_commit_sha,
            "discovery_status": self.status.value,
            "python_file_count": self.python_file_count,
            "total_scanned_bytes": self.total_scanned_bytes,
            "duration_seconds": self.duration_seconds,
            "cleanup_required": self.cleanup_required,
            "include_tests": self.include_tests,
            "limits": self.limits.to_dict(),
            "python_modules": [module.to_dict() for module in self.discovered_modules],
            "issues": [issue.to_dict() for issue in self.issues],
        }
