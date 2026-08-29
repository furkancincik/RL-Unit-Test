from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from analyzer.python_source_reader import (
    InlinePythonSourceNormalizationError,
    canonicalize_inline_python_source,
)
from models.project_analysis_result import (
    FunctionRunStatus,
    ProjectAnalysisResult,
    TargetSelection,
)
from models.project_coverage_result import ProjectCoverageResult
from models.source_acquisition_result import SourceAcquisitionLimits


class ExternalSourceKind(str, Enum):
    INLINE_PYTHON_SOURCE = "INLINE_PYTHON_SOURCE"
    UPLOADED_PYTHON_FILE = "UPLOADED_PYTHON_FILE"
    LOCAL_PROJECT_DIRECTORY = "LOCAL_PROJECT_DIRECTORY"
    PUBLIC_GITHUB_REPOSITORY = "PUBLIC_GITHUB_REPOSITORY"


class ExternalExecutionPolicy(str, Enum):
    STATIC_DISCOVERY_ONLY = "STATIC_DISCOVERY_ONLY"
    TRUSTED_DYNAMIC_ANALYSIS = "TRUSTED_DYNAMIC_ANALYSIS"


class ExternalModuleSelectionMode(str, Enum):
    ALL_ELIGIBLE_WITH_LIMIT = "ALL_ELIGIBLE_WITH_LIMIT"
    EXPLICIT_RELATIVE_PATHS = "EXPLICIT_RELATIVE_PATHS"
    EXPLICIT_MODULE_NAMES = "EXPLICIT_MODULE_NAMES"


class ExternalAnalysisStatus(str, Enum):
    STATIC_COMPLETED = "STATIC_COMPLETED"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class ExternalModuleStatus(str, Enum):
    STATIC_ONLY = "STATIC_ONLY"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    UNSUPPORTED = "UNSUPPORTED"
    SKIPPED_LIMIT = "SKIPPED_LIMIT"
    SKIPPED_DEADLINE = "SKIPPED_DEADLINE"


class ExternalWorkspaceCleanupStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


PUBLIC_GITHUB_STATIC_ONLY_MESSAGE = (
    "Public GitHub kaynakları yalnız statik keşif destekler."
)


class ExternalSourcePolicyValidationError(ValueError):
    """Kaynak türü ile execution policy arasındaki güvenlik invariant'ı."""


@dataclass(frozen=True, slots=True)
class InlinePythonSource:
    source_text: str
    display_name: str | None = None
    transport_size_bytes: int = field(init=False, repr=False)
    normalization_issue: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_text, str):
            raise TypeError("source_text string olmalıdır.")
        transport_size = len(self.source_text.encode("utf-8"))
        try:
            canonical_source = canonicalize_inline_python_source(
                self.source_text
            )
        except InlinePythonSourceNormalizationError:
            canonical_source = self.source_text
            normalization_issue = "SYNTAX_ERROR"
        else:
            normalization_issue = None
        if not canonical_source.strip():
            raise ValueError("source_text boş olamaz.")
        if self.display_name is not None and not isinstance(self.display_name, str):
            raise TypeError("display_name string veya None olmalıdır.")
        object.__setattr__(self, "source_text", canonical_source)
        object.__setattr__(self, "transport_size_bytes", transport_size)
        object.__setattr__(self, "normalization_issue", normalization_issue)

    @property
    def source_kind(self) -> ExternalSourceKind:
        return ExternalSourceKind.INLINE_PYTHON_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return {"source_kind": self.source_kind.value, "display_name": self.display_name}


@dataclass(frozen=True, slots=True)
class UploadedPythonFile:
    original_filename: str
    file_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.original_filename, str) or not self.original_filename:
            raise TypeError("original_filename boş olmayan string olmalıdır.")
        if not isinstance(self.file_bytes, bytes):
            raise TypeError("file_bytes bytes olmalıdır.")

    @property
    def source_kind(self) -> ExternalSourceKind:
        return ExternalSourceKind.UPLOADED_PYTHON_FILE

    def to_dict(self) -> dict[str, Any]:
        return {"source_kind": self.source_kind.value, "original_filename": Path(self.original_filename).name}


@dataclass(frozen=True, slots=True)
class LocalProjectDirectory:
    directory: Path
    include_tests: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise TypeError("directory Path olmalıdır.")
        if not isinstance(self.include_tests, bool):
            raise TypeError("include_tests bool olmalıdır.")

    @property
    def source_kind(self) -> ExternalSourceKind:
        return ExternalSourceKind.LOCAL_PROJECT_DIRECTORY

    def to_dict(self) -> dict[str, Any]:
        return {"source_kind": self.source_kind.value, "display_name": self.directory.name, "include_tests": self.include_tests}


@dataclass(frozen=True, slots=True)
class PublicGitHubRepository:
    repository_url: str
    ref: str | None = None
    include_tests: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository_url, str) or not self.repository_url:
            raise TypeError("repository_url boş olmayan string olmalıdır.")
        if self.ref is not None and not isinstance(self.ref, str):
            raise TypeError("ref string veya None olmalıdır.")
        if not isinstance(self.include_tests, bool):
            raise TypeError("include_tests bool olmalıdır.")

    @property
    def source_kind(self) -> ExternalSourceKind:
        return ExternalSourceKind.PUBLIC_GITHUB_REPOSITORY

    def to_dict(self) -> dict[str, Any]:
        return {"source_kind": self.source_kind.value, "repository_url": self.repository_url, "requested_ref": self.ref, "include_tests": self.include_tests}


ExternalSourcePayload = (
    InlinePythonSource
    | UploadedPythonFile
    | LocalProjectDirectory
    | PublicGitHubRepository
)


def validate_external_source_execution_policy(
    source: ExternalSourcePayload,
    execution_policy: ExternalExecutionPolicy,
) -> None:
    if (
        isinstance(source, PublicGitHubRepository)
        and execution_policy
        is not ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY
    ):
        raise ExternalSourcePolicyValidationError(
            PUBLIC_GITHUB_STATIC_ONLY_MESSAGE
        )


@dataclass(frozen=True, slots=True)
class ExternalModuleSelection:
    mode: ExternalModuleSelectionMode = ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExternalModuleSelectionMode):
            raise TypeError("module selection mode geçersiz.")
        if not isinstance(self.values, tuple) or any(
            not isinstance(value, str) or not value for value in self.values
        ):
            raise TypeError("module selection values string tuple'ı olmalıdır.")
        explicit = self.mode is not ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
        if explicit != bool(self.values):
            raise ValueError("Explicit module selection values ile birlikte kullanılmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class ExternalAnalysisConfiguration:
    output_root: Path
    module_selection: ExternalModuleSelection = ExternalModuleSelection()
    target_selection: TargetSelection = TargetSelection()
    maximum_selected_modules: int = 10
    maximum_functions_per_module: int = 10
    maximum_payload_bytes: int = 2_000_000
    max_visits_per_node: int = 3
    episode_count: int = 3
    epsilon: float = 0.0
    learning_rate: float = 0.5
    discount_factor: float = 0.9
    random_seed: int | None = 42
    pytest_coverage_timeout_seconds: float = 30.0
    per_function_pipeline_timeout_seconds: float | None = 120.0
    run_greedy_baseline: bool = False
    run_strategy_comparison: bool = False
    comparison_timeout_seconds: float | None = None
    project_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root Path olmalıdır.")
        if not isinstance(self.module_selection, ExternalModuleSelection):
            raise TypeError("module_selection geçersiz.")
        if not isinstance(self.target_selection, TargetSelection):
            raise TypeError("target_selection geçersiz.")
        for name in (
            "maximum_selected_modules",
            "maximum_functions_per_module",
            "maximum_payload_bytes",
            "max_visits_per_node",
            "episode_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} tam sayı olmalıdır.")
            if value <= 0:
                raise ValueError(f"{name} pozitif olmalıdır.")
        for name in (
            "epsilon",
            "learning_rate",
            "discount_factor",
            "pytest_coverage_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} sayısal olmalıdır.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} sonlu olmalıdır.")
        if self.pytest_coverage_timeout_seconds <= 0:
            raise ValueError("pytest_coverage_timeout_seconds pozitif olmalıdır.")
        for name in (
            "per_function_pipeline_timeout_seconds",
            "comparison_timeout_seconds",
            "project_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} pozitif sonlu sayı veya None olmalıdır.")
        if not isinstance(self.run_greedy_baseline, bool) or not isinstance(
            self.run_strategy_comparison, bool
        ):
            raise TypeError("greedy/comparison flag bool olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_selection": self.module_selection.to_dict(),
            "target_selection": self.target_selection.to_dict(),
            "maximum_selected_modules": self.maximum_selected_modules,
            "maximum_functions_per_module": self.maximum_functions_per_module,
            "maximum_payload_bytes": self.maximum_payload_bytes,
            "max_visits_per_node": self.max_visits_per_node,
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "random_seed": self.random_seed,
            "pytest_coverage_timeout_seconds": self.pytest_coverage_timeout_seconds,
            "per_function_pipeline_timeout_seconds": self.per_function_pipeline_timeout_seconds,
            "run_greedy_baseline": self.run_greedy_baseline,
            "run_strategy_comparison": self.run_strategy_comparison,
            "comparison_timeout_seconds": self.comparison_timeout_seconds,
            "project_timeout_seconds": self.project_timeout_seconds,
        }


@dataclass(frozen=True, slots=True, init=False)
class ExternalSourceAnalysisRequest:
    source: ExternalSourcePayload
    execution_policy: ExternalExecutionPolicy
    configuration: ExternalAnalysisConfiguration
    acquisition_limits: SourceAcquisitionLimits

    def __init__(
        self,
        source: ExternalSourcePayload,
        execution_policy: ExternalExecutionPolicy = ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        configuration: ExternalAnalysisConfiguration | None = None,
        acquisition_limits: SourceAcquisitionLimits = SourceAcquisitionLimits(),
    ) -> None:
        if not isinstance(source, (InlinePythonSource, UploadedPythonFile, LocalProjectDirectory, PublicGitHubRepository)):
            raise TypeError("source external source payload olmalıdır.")
        if not isinstance(execution_policy, ExternalExecutionPolicy):
            raise TypeError("execution_policy geçersiz.")
        if not isinstance(configuration, ExternalAnalysisConfiguration):
            raise TypeError("configuration gereklidir.")
        if not isinstance(acquisition_limits, SourceAcquisitionLimits):
            raise TypeError("acquisition_limits geçersiz.")
        validate_external_source_execution_policy(source, execution_policy)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "execution_policy", execution_policy)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "acquisition_limits", acquisition_limits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source.source_kind.value,
            "source": self.source.to_dict(),
            "execution_policy": self.execution_policy.value,
            "configuration": self.configuration.to_dict(),
            "acquisition_limits": self.acquisition_limits.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExternalModuleAnalysisResult:
    relative_path: str
    module_name: str | None
    status: ExternalModuleStatus
    discovered_function_count: int
    project_result: ProjectAnalysisResult | Any | None
    issue_category: str | None
    issue_message: str | None
    artifact_paths: tuple[Path, ...]
    discovered_function_names: tuple[str, ...] = ()

    @property
    def analyzed_function_count(self) -> int:
        if self.project_result is None:
            return 0
        return sum(
            item.status
            not in {
                FunctionRunStatus.SKIPPED,
                FunctionRunStatus.SKIPPED_SELECTION,
                FunctionRunStatus.SKIPPED_LIMIT,
                FunctionRunStatus.SKIPPED_DEADLINE,
                FunctionRunStatus.UNSUPPORTED,
            }
            for item in getattr(self.project_result, "function_results", ())
        )

    @property
    def limit_skipped_function_count(self) -> int:
        if self.project_result is None:
            return 0
        return sum(
            item.status is FunctionRunStatus.SKIPPED_LIMIT
            for item in getattr(self.project_result, "function_results", ())
        )

    @property
    def selection_skipped_function_count(self) -> int:
        if self.project_result is None:
            return 0
        return sum(
            item.status is FunctionRunStatus.SKIPPED_SELECTION
            for item in self.project_result.function_results
        )

    @property
    def deadline_skipped_function_count(self) -> int:
        if self.project_result is None:
            return (
                self.discovered_function_count
                if self.status is ExternalModuleStatus.SKIPPED_DEADLINE
                else 0
            )
        return sum(
            item.status is FunctionRunStatus.SKIPPED_DEADLINE
            for item in self.project_result.function_results
        )

    @property
    def line_coverage_percent(self) -> float | None:
        values = self._diagnostic_values("line_coverage_percent")
        return values[0] if len(values) == 1 else None

    @property
    def branch_coverage_percent(self) -> float | None:
        values = self._diagnostic_values("branch_coverage_percent")
        return values[0] if len(values) == 1 else None

    def _diagnostic_values(self, name: str) -> tuple[float, ...]:
        if self.project_result is None:
            return ()
        values: list[float] = []
        for function in getattr(self.project_result, "function_results", ()):
            diagnostic = getattr(function, "diagnostic", None)
            value = getattr(diagnostic, name, None) if diagnostic is not None else None
            if value is not None:
                values.append(value)
        return tuple(values)

    def to_dict(self, output_root: Path) -> dict[str, Any]:
        functions = []
        greedy_summaries = []
        comparison_summaries = []
        for item in getattr(self.project_result, "function_results", ()) if self.project_result is not None else ():
            diagnostic = getattr(item, "diagnostic", None)
            comparison = getattr(item, "strategy_comparison", None)
            scenario_pool = getattr(item, "scenario_pool_coverage", None)
            minimization = getattr(item, "minimization_result", None)
            best_rl = getattr(item, "best_rl_coverage", None)
            verified_rl_line = getattr(
                comparison, "rl_verified_line_percentage", None
            )
            verified_rl_branch = getattr(
                comparison, "rl_verified_branch_percentage", None
            )
            scenario_pool_line = getattr(
                scenario_pool,
                "line_coverage_percent",
                getattr(diagnostic, "line_coverage_percent", None),
            )
            scenario_pool_branch = getattr(
                scenario_pool,
                "branch_coverage_percent",
                getattr(diagnostic, "branch_coverage_percent", None),
            )
            functions.append(
                {
                    "qualified_name": item.target.qualified_name,
                    "status": item.status.value,
                    "skip_reason": item.skip_reason,
                    "line_coverage_percent": scenario_pool_line,
                    "branch_coverage_percent": scenario_pool_branch,
                    "scenario_pool_line_coverage_percent": scenario_pool_line,
                    "scenario_pool_branch_coverage_percent": scenario_pool_branch,
                    "greedy_line_coverage_percent": getattr(
                        minimization, "final_verified_line_percentage", None
                    ),
                    "greedy_branch_coverage_percent": getattr(
                        minimization, "final_verified_branch_percentage", None
                    ),
                    "greedy_coverage_preserved": getattr(
                        minimization, "coverage_preserved", None
                    ),
                    "best_rl_line_coverage_percent": getattr(
                        best_rl, "line_coverage_percent", None
                    ) if verified_rl_line is None else verified_rl_line,
                    "best_rl_branch_coverage_percent": getattr(
                        best_rl, "branch_coverage_percent", None
                    ) if verified_rl_branch is None else verified_rl_branch,
                    "best_rl_coverage_preserved": getattr(
                        comparison,
                        "rl_coverage_preserved",
                        getattr(item, "best_rl_coverage_preserved", None),
                    ),
                    "scenario_count": getattr(item, "scenario_count", None),
                    "rl_test_count": getattr(item, "rl_test_count", None),
                    "strategy_comparison": (
                        {
                            "winner": comparison.winner.value,
                            "greedy_selected_count": comparison.greedy_selected_count,
                            "rl_executed_test_count": comparison.best_rl_executed_test_count,
                            "coverage_equality_verified": comparison.coverage_equality_verified,
                            "target_line_coverage_percent": comparison.target_line_percentage,
                            "target_branch_coverage_percent": comparison.target_branch_percentage,
                            "greedy_line_coverage_percent": comparison.greedy_verified_line_percentage,
                            "greedy_branch_coverage_percent": comparison.greedy_verified_branch_percentage,
                            "best_rl_line_coverage_percent": comparison.rl_verified_line_percentage,
                            "best_rl_branch_coverage_percent": comparison.rl_verified_branch_percentage,
                        }
                        if comparison is not None
                        else None
                    ),
                }
            )
            if comparison is not None:
                comparison_summaries.append(
                    {
                        "winner": comparison.winner.value,
                        "greedy_selected_count": comparison.greedy_selected_count,
                        "rl_executed_test_count": comparison.best_rl_executed_test_count,
                        "coverage_equality_verified": comparison.coverage_equality_verified,
                        "target_line_coverage_percent": comparison.target_line_percentage,
                        "target_branch_coverage_percent": comparison.target_branch_percentage,
                        "greedy_line_coverage_percent": comparison.greedy_verified_line_percentage,
                        "greedy_branch_coverage_percent": comparison.greedy_verified_branch_percentage,
                        "best_rl_line_coverage_percent": comparison.rl_verified_line_percentage,
                        "best_rl_branch_coverage_percent": comparison.rl_verified_branch_percentage,
                    }
                )
        for path in self.artifact_paths:
            if path.name != "minimization_report.json" or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            greedy_summaries.append(
                {
                    "run_status": payload.get("run_status"),
                    "full_pool_scenario_count": payload.get("full_pool_scenario_count"),
                    "final_selected_count": payload.get("final_selected_count"),
                    "coverage_preserved": payload.get("coverage_preserved"),
                }
            )
        return {
            "relative_path": self.relative_path,
            "module_name": self.module_name,
            "status": self.status.value,
            "discovered_function_count": self.discovered_function_count,
            "discovered_function_names": list(self.discovered_function_names),
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "selection_skipped_function_count": self.selection_skipped_function_count,
            "deadline_skipped_function_count": self.deadline_skipped_function_count,
            "line_coverage_percent": self.line_coverage_percent,
            "branch_coverage_percent": self.branch_coverage_percent,
            "greedy": greedy_summaries or None,
            "strategy_comparison": comparison_summaries or None,
            "functions": functions,
            "issue_category": self.issue_category,
            "issue_message": self.issue_message,
            "artifact_paths": [str(path.resolve().relative_to(output_root.resolve())).replace("\\", "/") for path in self.artifact_paths if path.resolve().is_relative_to(output_root.resolve())],
        }


@dataclass(frozen=True, slots=True)
class ExternalSourceAnalysisResult:
    source_kind: ExternalSourceKind
    execution_policy: ExternalExecutionPolicy
    status: ExternalAnalysisStatus
    acquisition_status: str
    repository_name: str | None
    github_owner: str | None
    github_repository: str | None
    resolved_commit_sha: str | None
    discovered_module_count: int
    selected_module_count: int
    module_results: tuple[ExternalModuleAnalysisResult, ...]
    output_root: Path
    report_path: Path
    duration_seconds: float
    cleanup_status: ExternalWorkspaceCleanupStatus
    issues: tuple[str, ...]
    project_coverage: ProjectCoverageResult | None = None
    project_timeout_seconds: float | None = None
    project_deadline_exceeded: bool = False
    last_completed_stage: str | None = None
    deadline_stage: str | None = None

    @property
    def analyzed_module_count(self) -> int:
        return sum(item.status in {ExternalModuleStatus.COMPLETED, ExternalModuleStatus.PARTIAL, ExternalModuleStatus.FAILED, ExternalModuleStatus.TIMED_OUT} for item in self.module_results)

    @property
    def discovered_function_count(self) -> int:
        return sum(item.discovered_function_count for item in self.module_results)

    @property
    def analyzed_function_count(self) -> int:
        return sum(item.analyzed_function_count for item in self.module_results)

    @property
    def limit_skipped_function_count(self) -> int:
        return sum(item.limit_skipped_function_count for item in self.module_results)

    @property
    def selection_skipped_function_count(self) -> int:
        return sum(
            item.selection_skipped_function_count for item in self.module_results
        )

    @property
    def deadline_skipped_function_count(self) -> int:
        return sum(
            item.deadline_skipped_function_count for item in self.module_results
        )

    def _function_status_count(self, status: FunctionRunStatus) -> int:
        return sum(
            function.status is status
            for module in self.module_results
            for function in (
                getattr(module.project_result, "function_results", ())
                if module.project_result is not None
                else ()
            )
        )

    @property
    def completed_function_count(self) -> int:
        return self._function_status_count(FunctionRunStatus.COMPLETED)

    @property
    def partial_function_count(self) -> int:
        return self._function_status_count(FunctionRunStatus.PARTIAL)

    @property
    def timed_out_function_count(self) -> int:
        return self._function_status_count(FunctionRunStatus.TIMED_OUT) + sum(
            item.discovered_function_count
            for item in self.module_results
            if item.status is ExternalModuleStatus.TIMED_OUT
            and item.project_result is None
        )

    def to_dict(self) -> dict[str, Any]:
        counts = {status.value: sum(item.status is status for item in self.module_results) for status in ExternalModuleStatus}
        return {
            "schema_version": "1.0",
            "source_kind": self.source_kind.value,
            "execution_policy": self.execution_policy.value,
            "status": self.status.value,
            "acquisition_status": self.acquisition_status,
            "repository_name": self.repository_name,
            "github_owner": self.github_owner,
            "github_repository": self.github_repository,
            "resolved_commit_sha": self.resolved_commit_sha,
            "discovered_module_count": self.discovered_module_count,
            "selected_module_count": self.selected_module_count,
            "analyzed_module_count": self.analyzed_module_count,
            "discovered_function_count": self.discovered_function_count,
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "selection_skipped_function_count": self.selection_skipped_function_count,
            "deadline_skipped_function_count": self.deadline_skipped_function_count,
            "completed_function_count": self.completed_function_count,
            "partial_function_count": self.partial_function_count,
            "timed_out_function_count": self.timed_out_function_count,
            "module_status_counts": counts,
            "modules": [item.to_dict(self.output_root) for item in self.module_results],
            "output_root": ".",
            "report_path": self.report_path.name,
            "duration_seconds": self.duration_seconds,
            "project_timeout_seconds": self.project_timeout_seconds,
            "project_deadline_exceeded": self.project_deadline_exceeded,
            "last_completed_stage": self.last_completed_stage,
            "deadline_stage": self.deadline_stage,
            "cleanup_status": self.cleanup_status.value,
            "issues": list(self.issues),
            "project_coverage": (
                self.project_coverage.to_dict()
                if self.project_coverage is not None
                else None
            ),
            "aggregate_project_coverage": (
                {
                    "coverage_scope": self.project_coverage.coverage_scope.value,
                    "line_percent": self.project_coverage.full_line_coverage_percent,
                    "branch_percent": self.project_coverage.full_branch_coverage_percent,
                    "status": self.project_coverage.status.value,
                    "scope_complete": self.project_coverage.scope.scope_complete,
                }
                if self.project_coverage is not None
                else {
                    "line_percent": None,
                    "branch_percent": None,
                    "status": "UNMEASURED",
                }
            ),
        }
