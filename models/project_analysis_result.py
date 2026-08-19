from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from models.pipeline_diagnostic_result import PipelineDiagnosticResult


class FunctionSelectionMode(str, Enum):
    """Kaynak analizinde hedef fonksiyonların nasıl seçildiğidir."""

    SINGLE = "SINGLE"
    ALL = "ALL"


class FunctionRunStatus(str, Enum):
    """Tek bir keşfedilmiş fonksiyonun production sonucudur."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED = "SKIPPED"
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
            FunctionRunStatus.UNSUPPORTED,
        } and not self.skip_reason:
            raise ValueError("Skipped/unsupported function bir reason taşımalıdır.")

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

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file Path olmalıdır.")
        if not isinstance(self.selection_mode, FunctionSelectionMode):
            raise TypeError("selection_mode geçersiz.")
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
        if values and all(
            status is FunctionRunStatus.COMPLETED for status in values
        ):
            return ProjectRunStatus.COMPLETED
        if any(
            status in {FunctionRunStatus.COMPLETED, FunctionRunStatus.PARTIAL}
            for status in values
        ):
            return ProjectRunStatus.PARTIAL
        if any(status is FunctionRunStatus.TIMED_OUT for status in values):
            return ProjectRunStatus.TIMED_OUT
        return ProjectRunStatus.FAILED

    @property
    def selected_function_count(self) -> int:
        return len(self.function_results)

    @property
    def executed_function_count(self) -> int:
        return sum(
            item.status
            not in {FunctionRunStatus.SKIPPED, FunctionRunStatus.UNSUPPORTED}
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
