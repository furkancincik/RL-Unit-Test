from __future__ import annotations

import math
import keyword
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from models.coverage_result import FunctionCoverageResult
from models.pipeline_diagnostic_result import PipelineDiagnosticResult
from models.project_coverage_result import ProjectTestCandidate
from models.scenario_minimization_result import ScenarioMinimizationResult
from models.strategy_comparison_result import StrategyComparisonResult


class FunctionSelectionMode(str, Enum):
    """Kaynak analizinde hedef fonksiyonların nasıl seçildiğidir."""

    ALL_ELIGIBLE_WITH_LIMIT = "ALL_ELIGIBLE_WITH_LIMIT"
    EXPLICIT_QUALIFIED_TARGETS = "EXPLICIT_QUALIFIED_TARGETS"
    ALL = ALL_ELIGIBLE_WITH_LIMIT
    SINGLE = EXPLICIT_QUALIFIED_TARGETS


TargetSelectionMode = FunctionSelectionMode


def validate_qualified_target_name(value: str) -> str:
    """Top-level function veya iki segmentli instance-method adını doğrular."""
    if not isinstance(value, str):
        raise TypeError("qualified target name string olmalıdır.")
    if not value or value != value.strip():
        raise ValueError("qualified target name geçersiz.")
    parts = value.split(".")
    if len(parts) not in {1, 2} or any(
        not part.isidentifier() or keyword.iskeyword(part)
        for part in parts
    ):
        raise ValueError("qualified target name geçersiz.")
    return value


def validate_module_identity(value: str) -> str:
    """Selector için canonical dotted module identity doğrular."""
    if not isinstance(value, str):
        raise TypeError("module identity string olmalıdır.")
    if not value or value != value.strip():
        raise ValueError("module identity geçersiz.")
    if any(
        not part.isidentifier() or keyword.iskeyword(part)
        for part in value.split(".")
    ):
        raise ValueError("module identity geçersiz.")
    return value


@dataclass(frozen=True, slots=True)
class QualifiedTargetSelector:
    """Bir modüldeki exact discovery target kimliğidir."""

    module_identity: str
    qualified_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module_identity",
            validate_module_identity(self.module_identity),
        )
        object.__setattr__(
            self,
            "qualified_name",
            validate_qualified_target_name(self.qualified_name),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "module_identity": self.module_identity,
            "qualified_name": self.qualified_name,
        }


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """Default veya explicit qualified target seçim sözleşmesidir."""

    mode: TargetSelectionMode = TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    selectors: tuple[QualifiedTargetSelector, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TargetSelectionMode):
            raise TypeError("target selection mode geçersiz.")
        if not isinstance(self.selectors, tuple) or any(
            not isinstance(item, QualifiedTargetSelector)
            for item in self.selectors
        ):
            raise TypeError("target selectors tuple olmalıdır.")
        explicit = self.mode is TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS
        if explicit != bool(self.selectors):
            raise ValueError(
                "Explicit target selection selector değerleriyle kullanılmalıdır."
            )
        object.__setattr__(
            self,
            "selectors",
            tuple(dict.fromkeys(self.selectors)),
        )

    def for_module(self, module_identity: str) -> tuple[str, ...]:
        normalized = validate_module_identity(module_identity)
        return tuple(
            item.qualified_name
            for item in self.selectors
            if item.module_identity == normalized
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "selectors": [item.to_dict() for item in self.selectors],
        }


class FunctionRunStatus(str, Enum):
    """Tek bir keşfedilmiş fonksiyonun production sonucudur."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED = "SKIPPED"
    SKIPPED_SELECTION = "SKIPPED_SELECTION"
    SKIPPED_LIMIT = "SKIPPED_LIMIT"
    SKIPPED_DEADLINE = "SKIPPED_DEADLINE"
    UNSUPPORTED = "UNSUPPORTED"


class ProjectRunStatus(str, Enum):
    """Birleşik kaynak analizi çalışmasının sonucudur."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True, slots=True)
class FunctionTarget:
    """AST üzerinden keşfedilmiş fonksiyon hedefinin güvenli metadata'sı."""

    name: str
    qualified_name: str
    start_line: int
    end_line: int
    parameters: tuple[str, ...]
    return_annotation: str | None
    is_async: bool
    is_nested: bool
    is_method: bool
    is_supported: bool
    unsupported_reason: str | None = None
    class_name: str | None = None
    constructor_parameters: tuple[str, ...] = ()
    constructor_parameter_types: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("Fonksiyon adı geçerli bir identifier olmalıdır.")
        if not self.qualified_name:
            raise ValueError("qualified_name boş olamaz.")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("Fonksiyon kaynak satır aralığı geçersiz.")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters tuple olmalıdır.")
        if not self.is_supported and not self.unsupported_reason:
            raise ValueError("Unsupported function target bir reason taşımalıdır.")
        if self.is_supported and self.unsupported_reason is not None:
            raise ValueError("Supported function target reason taşıyamaz.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parameters": list(self.parameters),
            "return_annotation": self.return_annotation,
            "is_async": self.is_async,
            "is_nested": self.is_nested,
            "is_method": self.is_method,
            "is_supported": self.is_supported,
            "unsupported_reason": self.unsupported_reason,
            "class_name": self.class_name,
            "constructor_parameters": list(self.constructor_parameters),
            "constructor_parameter_types": dict(
                self.constructor_parameter_types
            ),
        }


@dataclass(frozen=True, slots=True)
class FunctionAnalysisResult:
    """Tek fonksiyonun diagnostic ve artifact özetidir."""

    target: FunctionTarget
    status: FunctionRunStatus
    diagnostic: PipelineDiagnosticResult | None
    output_directory: Path
    artifact_paths: tuple[Path, ...] = ()
    skip_reason: str | None = None
    strategy_comparison: StrategyComparisonResult | None = None
    scenario_pool_coverage: FunctionCoverageResult | None = None
    minimization_result: ScenarioMinimizationResult | None = None
    best_rl_coverage: FunctionCoverageResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, FunctionTarget):
            raise TypeError("target bir FunctionTarget olmalıdır.")
        if not isinstance(self.status, FunctionRunStatus):
            raise TypeError("status bir FunctionRunStatus olmalıdır.")
        if self.diagnostic is not None and not isinstance(
            self.diagnostic, PipelineDiagnosticResult
        ):
            raise TypeError("diagnostic PipelineDiagnosticResult olmalıdır.")
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory Path olmalıdır.")
        if self.status in {
            FunctionRunStatus.SKIPPED,
            FunctionRunStatus.SKIPPED_SELECTION,
            FunctionRunStatus.SKIPPED_LIMIT,
            FunctionRunStatus.SKIPPED_DEADLINE,
            FunctionRunStatus.UNSUPPORTED,
        } and not self.skip_reason:
            raise ValueError("Skipped/unsupported function bir reason taşımalıdır.")
        if self.strategy_comparison is not None and not isinstance(
            self.strategy_comparison, StrategyComparisonResult
        ):
            raise TypeError("strategy_comparison StrategyComparisonResult olmalıdır.")
        for value, expected_type, name in (
            (self.scenario_pool_coverage, FunctionCoverageResult, "scenario_pool_coverage"),
            (self.minimization_result, ScenarioMinimizationResult, "minimization_result"),
            (self.best_rl_coverage, FunctionCoverageResult, "best_rl_coverage"),
        ):
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(f"{name} {expected_type.__name__} olmalıdır.")

    @property
    def best_rl_coverage_preserved(self) -> bool | None:
        target = self.scenario_pool_coverage
        actual = self.best_rl_coverage
        if target is None or actual is None:
            return None
        return (
            actual.covered_lines == target.covered_lines
            and tuple(actual.covered_branches or ())
            == tuple(target.covered_branches or ())
        )

    @staticmethod
    def _coverage_summary(
        coverage: FunctionCoverageResult | None,
    ) -> dict[str, Any] | None:
        if coverage is None:
            return None
        return {
            "line_coverage_percent": coverage.line_coverage_percent,
            "branch_coverage_percent": coverage.branch_coverage_percent,
            "covered_line_identities": list(coverage.covered_lines),
            "covered_branch_identities": [
                list(branch) for branch in (coverage.covered_branches or ())
            ],
        }

    @property
    def usable(self) -> bool:
        return self.status in {
            FunctionRunStatus.COMPLETED,
            FunctionRunStatus.PARTIAL,
        }

    @property
    def scenario_count(self) -> int | None:
        return self._funnel_value("final_scenario_count")

    @property
    def concrete_accepted_count(self) -> int | None:
        return self._funnel_value("concrete_validation_accepted_count")

    @property
    def concrete_rejected_count(self) -> int | None:
        return self._funnel_value("concrete_validation_rejected_count")

    @property
    def rl_test_count(self) -> int | None:
        return self._funnel_value("rl_executed_test_count")

    @property
    def q_table_state_count(self) -> int | None:
        return self._funnel_value("q_table_state_count")

    def _funnel_value(self, name: str) -> int | None:
        if self.diagnostic is None:
            return None
        return getattr(self.diagnostic.funnel, name)

    def to_dict(self) -> dict[str, Any]:
        diagnostic = self.diagnostic
        return {
            "target": self.target.to_dict(),
            "status": self.status.value,
            "scenario_count": self.scenario_count,
            "concrete_accepted_count": self.concrete_accepted_count,
            "concrete_rejected_count": self.concrete_rejected_count,
            "rl_test_count": self.rl_test_count,
            "q_table_state_count": self.q_table_state_count,
            "line_coverage_percent": (
                diagnostic.line_coverage_percent
                if diagnostic is not None
                else None
            ),
            "branch_coverage_percent": (
                diagnostic.branch_coverage_percent
                if diagnostic is not None
                else None
            ),
            "reachability_counts": (
                dict(diagnostic.reachability_counts)
                if diagnostic is not None
                else {}
            ),
            "duration_seconds": (
                diagnostic.total_duration_seconds
                if diagnostic is not None
                else None
            ),
            "output_directory": str(self.output_directory),
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "skip_reason": self.skip_reason,
            "diagnostic": (
                diagnostic.to_dict() if diagnostic is not None else None
            ),
            "strategy_comparison": (
                self.strategy_comparison.to_dict()
                if self.strategy_comparison is not None
                else None
            ),
            "scenario_pool_coverage": self._coverage_summary(
                self.scenario_pool_coverage
            ),
            "greedy_minimization": (
                self.minimization_result.to_dict()
                if self.minimization_result is not None
                else None
            ),
            "best_rl_coverage": self._coverage_summary(self.best_rl_coverage),
            "best_rl_coverage_preserved": self.best_rl_coverage_preserved,
        }


@dataclass(frozen=True, slots=True)
class ProjectAnalysisResult:
    """Çok fonksiyonlu production orchestration sonucudur."""

    source_file: Path
    module_path: str
    selection_mode: FunctionSelectionMode
    discovered_targets: tuple[FunctionTarget, ...]
    function_results: tuple[FunctionAnalysisResult, ...]
    total_duration_seconds: float
    status: ProjectRunStatus
    output_root: Path
    report_path: Path
    coverage_candidates: tuple[ProjectTestCandidate, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file Path olmalıdır.")
        if not isinstance(self.selection_mode, FunctionSelectionMode):
            raise TypeError("selection_mode geçersiz.")
        if not isinstance(self.coverage_candidates, tuple) or any(
            not isinstance(item, ProjectTestCandidate)
            for item in self.coverage_candidates
        ):
            raise TypeError("coverage_candidates ProjectTestCandidate tuple'ı olmalıdır.")
        if isinstance(self.total_duration_seconds, bool) or not isinstance(
            self.total_duration_seconds, (int, float)
        ):
            raise TypeError("total_duration_seconds sayısal olmalıdır.")
        if not math.isfinite(float(self.total_duration_seconds)) or (
            self.total_duration_seconds < 0.0
        ):
            raise ValueError("total_duration_seconds negatif olmayan sonlu sayı olmalıdır.")
        derived = self.derive_status(item.status for item in self.function_results)
        if self.status is not derived:
            raise ValueError("Project status function sonuçlarıyla uyuşmuyor.")

    @staticmethod
    def derive_status(
        statuses: Iterable[FunctionRunStatus],
    ) -> ProjectRunStatus:
        values = tuple(statuses)
        relevant = tuple(
            status
            for status in values
            if status is not FunctionRunStatus.SKIPPED_SELECTION
        )
        if values and not relevant:
            return ProjectRunStatus.COMPLETED
        if relevant and all(
            status is FunctionRunStatus.COMPLETED for status in relevant
        ):
            return ProjectRunStatus.COMPLETED
        if any(
            status in {FunctionRunStatus.COMPLETED, FunctionRunStatus.PARTIAL}
            for status in relevant
        ):
            return ProjectRunStatus.PARTIAL
        if any(status is FunctionRunStatus.TIMED_OUT for status in relevant):
            return ProjectRunStatus.TIMED_OUT
        if any(status is FunctionRunStatus.SKIPPED_DEADLINE for status in relevant):
            return ProjectRunStatus.TIMED_OUT
        return ProjectRunStatus.FAILED

    @property
    def selected_function_count(self) -> int:
        return sum(
            item.status is not FunctionRunStatus.SKIPPED_SELECTION
            for item in self.function_results
        )

    @property
    def executed_function_count(self) -> int:
        return sum(
            item.status
            not in {
                FunctionRunStatus.SKIPPED,
                FunctionRunStatus.SKIPPED_SELECTION,
                FunctionRunStatus.SKIPPED_LIMIT,
                FunctionRunStatus.SKIPPED_DEADLINE,
                FunctionRunStatus.UNSUPPORTED,
            }
            for item in self.function_results
        )

    def _count(self, status: FunctionRunStatus) -> int:
        return sum(item.status is status for item in self.function_results)

    @property
    def completed_count(self) -> int:
        return self._count(FunctionRunStatus.COMPLETED)

    @property
    def partial_count(self) -> int:
        return self._count(FunctionRunStatus.PARTIAL)

    @property
    def failed_count(self) -> int:
        return self._count(FunctionRunStatus.FAILED)

    @property
    def timed_out_count(self) -> int:
        return self._count(FunctionRunStatus.TIMED_OUT)

    @property
    def skipped_count(self) -> int:
        return self._count(FunctionRunStatus.SKIPPED)

    @property
    def limit_skipped_count(self) -> int:
        return self.limit_skipped_function_count

    @property
    def limit_skipped_function_count(self) -> int:
        return self._count(FunctionRunStatus.SKIPPED_LIMIT)

    @property
    def selection_skipped_function_count(self) -> int:
        return self._count(FunctionRunStatus.SKIPPED_SELECTION)

    @property
    def deadline_skipped_function_count(self) -> int:
        return self._count(FunctionRunStatus.SKIPPED_DEADLINE)

    @property
    def skipped_function_count(self) -> int:
        return (
            self.skipped_count
            + self.selection_skipped_function_count
            + self.limit_skipped_function_count
            + self.deadline_skipped_function_count
        )

    @property
    def unsupported_count(self) -> int:
        return self._count(FunctionRunStatus.UNSUPPORTED)

    @property
    def has_usable_result(self) -> bool:
        return any(item.usable for item in self.function_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "project_status": self.status.value,
            "source_file": str(self.source_file),
            "module_path": self.module_path,
            "selection_mode": self.selection_mode.value,
            "output_root": str(self.output_root),
            "report_path": str(self.report_path),
            "total_duration_seconds": self.total_duration_seconds,
            "summary": {
                "discovered_function_count": len(self.discovered_targets),
                "selected_function_count": self.selected_function_count,
                "executed_function_count": self.executed_function_count,
                "completed_count": self.completed_count,
                "partial_count": self.partial_count,
                "failed_count": self.failed_count,
                "timed_out_count": self.timed_out_count,
                "skipped_count": self.skipped_count,
                "selection_skipped_function_count": (
                    self.selection_skipped_function_count
                ),
                "skipped_function_count": self.skipped_function_count,
                "limit_skipped_function_count": self.limit_skipped_function_count,
                "deadline_skipped_function_count": (
                    self.deadline_skipped_function_count
                ),
                "unsupported_count": self.unsupported_count,
            },
            "aggregate_project_coverage": {
                "line_percent": None,
                "branch_percent": None,
                "status": "UNMEASURED",
            },
            "discovered_functions": [
                target.to_dict() for target in self.discovered_targets
            ],
            "functions": [item.to_dict() for item in self.function_results],
        }
