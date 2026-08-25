from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from generator.scenario_generator import Scenario


class ProjectCoverageStatus(str, Enum):
    """Birleşik project coverage/minimization çalışmasının durumudur."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    NON_ADDITIVE_COVERAGE = "NON_ADDITIVE_COVERAGE"


class ProjectCoverageScope(str, Enum):
    """Yüzdelerin hangi kaynak kapsamını ölçtüğünü belirtir."""

    ANALYZED_PROJECT_SCOPE_COVERAGE = "ANALYZED_PROJECT_SCOPE_COVERAGE"


def _normalize_relative_module_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("relative_module_path boş olmayan string olmalıdır.")
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or normalized.startswith("/")
        or not posix.parts
    ):
        raise ValueError("relative_module_path güvenli relative path olmalıdır.")
    return posix.as_posix()


@dataclass(frozen=True, slots=True, order=True)
class ProjectLineIdentity:
    relative_module_path: str
    line_number: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_module_path",
            _normalize_relative_module_path(self.relative_module_path),
        )
        if (
            isinstance(self.line_number, bool)
            or not isinstance(self.line_number, int)
            or self.line_number < 1
        ):
            raise ValueError("line_number pozitif tam sayı olmalıdır.")

    def to_tuple(self) -> tuple[str, int]:
        return self.relative_module_path, self.line_number


@dataclass(frozen=True, slots=True, order=True)
class ProjectBranchIdentity:
    relative_module_path: str
    from_line: int
    to_line: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_module_path",
            _normalize_relative_module_path(self.relative_module_path),
        )
        if (
            isinstance(self.from_line, bool)
            or not isinstance(self.from_line, int)
            or self.from_line < 1
        ):
            raise ValueError("from_line pozitif tam sayı olmalıdır.")
        if isinstance(self.to_line, bool) or not isinstance(self.to_line, int):
            raise TypeError("to_line tam sayı olmalıdır.")

    def to_tuple(self) -> tuple[str, int, int]:
        return self.relative_module_path, self.from_line, self.to_line


@dataclass(frozen=True, slots=True)
class ProjectCoverageScopeSummary:
    discovered_module_count: int
    selected_module_count: int
    completed_module_count: int
    discovered_function_count: int
    eligible_function_count: int
    analyzed_function_count: int
    completed_function_count: int
    partial_function_count: int
    failed_function_count: int
    timed_out_function_count: int
    unsupported_function_count: int
    skipped_limit_function_count: int
    scope_complete: bool
    skipped_selection_function_count: int = 0
    skipped_deadline_function_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "discovered_module_count",
            "selected_module_count",
            "completed_module_count",
            "discovered_function_count",
            "eligible_function_count",
            "analyzed_function_count",
            "completed_function_count",
            "partial_function_count",
            "failed_function_count",
            "timed_out_function_count",
            "unsupported_function_count",
            "skipped_limit_function_count",
            "skipped_selection_function_count",
            "skipped_deadline_function_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if not isinstance(self.scope_complete, bool):
            raise TypeError("scope_complete bool olmalıdır.")
        if self.selected_module_count > self.discovered_module_count:
            raise ValueError("selected module sayısı discovered sayısını aşamaz.")
        if self.completed_module_count > self.selected_module_count:
            raise ValueError("completed module sayısı selected sayısını aşamaz.")
        if self.eligible_function_count > self.discovered_function_count:
            raise ValueError("eligible function sayısı discovered sayısını aşamaz.")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        payload["incomplete_function_count"] = max(
            0,
            self.eligible_function_count - self.completed_function_count,
        )
        return payload


@dataclass(frozen=True, slots=True)
class ProjectTestCandidate:
    """Public çıktıya serialize edilmeyen concrete-valid project test adayıdır."""

    project_test_id: str
    relative_module_path: str
    module_path: str
    function_name: str
    function_start_line: int
    function_end_line: int
    source_file: Path = field(repr=False, compare=False)
    import_root: Path = field(repr=False, compare=False)
    scenario: Scenario = field(repr=False, compare=False)
    original_order: int
    precomputed_line_identities: tuple[int, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    precomputed_branch_identities: tuple[tuple[int, int], ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    precomputed_execution_success: bool | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.project_test_id, str) or not self.project_test_id:
            raise ValueError("project_test_id boş olamaz.")
        object.__setattr__(
            self,
            "relative_module_path",
            _normalize_relative_module_path(self.relative_module_path),
        )
        if not self.module_path or any(
            not part.isidentifier() for part in self.module_path.split(".")
        ):
            raise ValueError("module_path geçersiz.")
        if not self.function_name.isidentifier():
            raise ValueError("function_name geçersiz.")
        if (
            self.function_start_line < 1
            or self.function_end_line < self.function_start_line
        ):
            raise ValueError("Fonksiyon satır aralığı geçersiz.")
        if not isinstance(self.source_file, Path) or not self.source_file.is_file():
            raise ValueError("source_file mevcut Python dosyası olmalıdır.")
        if not isinstance(self.import_root, Path) or not self.import_root.is_dir():
            raise ValueError("import_root mevcut directory olmalıdır.")
        if not self.source_file.resolve().is_relative_to(self.import_root.resolve()):
            raise ValueError("source_file import_root dışında olamaz.")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario Scenario olmalıdır.")
        if (
            isinstance(self.original_order, bool)
            or not isinstance(self.original_order, int)
            or self.original_order < 1
        ):
            raise ValueError("original_order pozitif tam sayı olmalıdır.")
        supplied = (
            self.precomputed_line_identities is not None,
            self.precomputed_branch_identities is not None,
            self.precomputed_execution_success is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("Precomputed coverage alanları birlikte verilmelidir.")

    @property
    def qualified_function_name(self) -> str:
        class_name = self.scenario.target_class_name
        if class_name is None:
            return self.function_name
        return f"{class_name}.{self.function_name}"


@dataclass(frozen=True, slots=True)
class ProjectCoverageResult:
    status: ProjectCoverageStatus
    scope: ProjectCoverageScopeSummary
    executable_line_identities: tuple[ProjectLineIdentity, ...]
    executable_branch_identities: tuple[ProjectBranchIdentity, ...]
    target_line_identities: tuple[ProjectLineIdentity, ...]
    target_branch_identities: tuple[ProjectBranchIdentity, ...]
    minimized_covered_line_identities: tuple[ProjectLineIdentity, ...]
    minimized_covered_branch_identities: tuple[ProjectBranchIdentity, ...]
    full_line_coverage_percent: float | None
    full_branch_coverage_percent: float | None
    minimized_line_coverage_percent: float | None
    minimized_branch_coverage_percent: float | None
    full_pytest_exit_code: int | None
    minimized_pytest_exit_code: int | None
    full_scenario_count: int
    greedy_initially_selected_count: int
    redundancy_elimination_removed_count: int
    selected_project_test_ids: tuple[str, ...]
    removed_project_test_ids: tuple[str, ...]
    coverage_preserved: bool
    duration_seconds: float
    output_root: Path = field(repr=False)
    full_test_file: Path = field(repr=False)
    minimized_test_file: Path = field(repr=False)
    report_path: Path = field(repr=False)
    artifact_paths: tuple[Path, ...] = field(default=(), repr=False)
    failure_category: str | None = None
    failure_message: str | None = None
    unmeasured_reason: str | None = None
    coverage_scope: ProjectCoverageScope = (
        ProjectCoverageScope.ANALYZED_PROJECT_SCOPE_COVERAGE
    )
    globally_minimal: bool = False
    algorithm: str = "GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION"

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProjectCoverageStatus):
            raise TypeError("status ProjectCoverageStatus olmalıdır.")
        if not isinstance(self.scope, ProjectCoverageScopeSummary):
            raise TypeError("scope ProjectCoverageScopeSummary olmalıdır.")
        for name, expected in (
            ("executable_line_identities", ProjectLineIdentity),
            ("executable_branch_identities", ProjectBranchIdentity),
            ("target_line_identities", ProjectLineIdentity),
            ("target_branch_identities", ProjectBranchIdentity),
            ("minimized_covered_line_identities", ProjectLineIdentity),
            ("minimized_covered_branch_identities", ProjectBranchIdentity),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, expected) for value in values
            ):
                raise TypeError(f"{name} geçersiz identity tuple'ıdır.")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} sıralı ve benzersiz olmalıdır.")
        for name in (
            "full_line_coverage_percent",
            "full_branch_coverage_percent",
            "minimized_line_coverage_percent",
            "minimized_branch_coverage_percent",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 100.0
            ):
                raise ValueError(f"{name} geçersiz yüzde değeridir.")
        for name in (
            "full_scenario_count",
            "greedy_initially_selected_count",
            "redundancy_elimination_removed_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0:
            raise ValueError("duration_seconds geçersiz.")
        if self.globally_minimal is not False:
            raise ValueError("Greedy project suite globally minimal değildir.")
        if not isinstance(self.coverage_preserved, bool):
            raise TypeError("coverage_preserved bool olmalıdır.")
        if self.final_selected_count > self.full_scenario_count:
            raise ValueError("Selected test sayısı full pool'u aşamaz.")
        if set(self.selected_project_test_ids) & set(self.removed_project_test_ids):
            raise ValueError("Selected ve removed project test ID kümeleri ayrık olmalıdır.")

    @property
    def final_selected_count(self) -> int:
        return len(self.selected_project_test_ids)

    @property
    def removal_count(self) -> int:
        return len(self.removed_project_test_ids)

    @property
    def reduction_percentage(self) -> float:
        if self.full_scenario_count == 0:
            return 0.0
        return self.removal_count / self.full_scenario_count * 100.0

    def _relative_artifact(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.output_root.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError("Project coverage artifact output root dışında.")
        return resolved.relative_to(root).as_posix()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_status": self.status.value,
            "coverage_scope": self.coverage_scope.value,
            "scope": self.scope.to_dict(),
            "whole_repository_line_coverage_percent": None,
            "whole_repository_branch_coverage_percent": None,
            "executable_line_identities": [
                list(value.to_tuple()) for value in self.executable_line_identities
            ],
            "executable_branch_identities": [
                list(value.to_tuple()) for value in self.executable_branch_identities
            ],
            "target_line_identities": [
                list(value.to_tuple()) for value in self.target_line_identities
            ],
            "target_line_identity_count": len(self.target_line_identities),
            "target_branch_identities": [
                list(value.to_tuple()) for value in self.target_branch_identities
            ],
            "target_branch_identity_count": len(self.target_branch_identities),
            "minimized_covered_line_identities": [
                list(value.to_tuple())
                for value in self.minimized_covered_line_identities
            ],
            "minimized_covered_line_identity_count": len(
                self.minimized_covered_line_identities
            ),
            "minimized_covered_branch_identities": [
                list(value.to_tuple())
                for value in self.minimized_covered_branch_identities
            ],
            "minimized_covered_branch_identity_count": len(
                self.minimized_covered_branch_identities
            ),
            "full_line_coverage_percent": self.full_line_coverage_percent,
            "full_branch_coverage_percent": self.full_branch_coverage_percent,
            "minimized_line_coverage_percent": self.minimized_line_coverage_percent,
            "minimized_branch_coverage_percent": self.minimized_branch_coverage_percent,
            "full_pytest_exit_code": self.full_pytest_exit_code,
            "minimized_pytest_exit_code": self.minimized_pytest_exit_code,
            "full_scenario_count": self.full_scenario_count,
            "greedy_initially_selected_count": self.greedy_initially_selected_count,
            "redundancy_elimination_removed_count": self.redundancy_elimination_removed_count,
            "selected_project_test_ids": list(self.selected_project_test_ids),
            "final_selected_count": self.final_selected_count,
            "removed_project_test_ids": list(self.removed_project_test_ids),
            "removal_count": self.removal_count,
            "reduction_percentage": self.reduction_percentage,
            "coverage_preserved": self.coverage_preserved,
            "globally_minimal": self.globally_minimal,
            "algorithm": self.algorithm,
            "duration_seconds": self.duration_seconds,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "unmeasured_reason": self.unmeasured_reason,
            "full_test_file": self._relative_artifact(self.full_test_file),
            "minimized_test_file": self._relative_artifact(self.minimized_test_file),
            "report_path": self._relative_artifact(self.report_path),
            "artifact_paths": [
                self._relative_artifact(path) for path in self.artifact_paths
            ],
        }
