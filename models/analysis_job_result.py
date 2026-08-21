from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from models.external_source_analysis_result import ExternalExecutionPolicy, ExternalSourceKind


class AnalysisJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self not in {AnalysisJobStatus.QUEUED, AnalysisJobStatus.RUNNING}


@dataclass(frozen=True, slots=True)
class AnalysisArtifactSummary:
    artifact_id: str
    filename: str
    content_type: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class AnalysisFunctionSummary:
    qualified_name: str
    status: str
    skip_reason: str | None
    scenario_count: int | None
    rl_test_count: int | None
    line_coverage_percent: float | None
    branch_coverage_percent: float | None
    greedy_selected_count: int | None
    rl_selected_count: int | None
    strategy_winner: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "scenario_count": self.scenario_count,
            "rl_test_count": self.rl_test_count,
            "line_coverage_percent": self.line_coverage_percent,
            "branch_coverage_percent": self.branch_coverage_percent,
            "greedy_selected_count": self.greedy_selected_count,
            "rl_selected_count": self.rl_selected_count,
            "strategy_winner": self.strategy_winner,
        }


@dataclass(frozen=True, slots=True)
class AnalysisModuleSummary:
    relative_path: str
    module_name: str | None
    status: str
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    functions: tuple[AnalysisFunctionSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "module_name": self.module_name,
            "status": self.status,
            "discovered_function_count": self.discovered_function_count,
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "functions": [item.to_dict() for item in self.functions],
        }


@dataclass(frozen=True, slots=True)
class AnalysisJobResultSummary:
    job_id: str
    source_kind: ExternalSourceKind
    analysis_policy: ExternalExecutionPolicy
    status: AnalysisJobStatus
    acquisition_status: str | None
    discovered_module_count: int
    selected_module_count: int
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    project_line_coverage_percent: None
    project_branch_coverage_percent: None
    modules: tuple[AnalysisModuleSummary, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_kind": self.source_kind.value,
            "analysis_policy": self.analysis_policy.value,
            "status": self.status.value,
            "acquisition_status": self.acquisition_status,
            "discovered_module_count": self.discovered_module_count,
            "selected_module_count": self.selected_module_count,
            "discovered_function_count": self.discovered_function_count,
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "project_line_coverage_percent": None,
            "project_branch_coverage_percent": None,
            "modules": [item.to_dict() for item in self.modules],
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class AnalysisJobSummary:
    job_id: str
    source_kind: ExternalSourceKind
    analysis_policy: ExternalExecutionPolicy
    status: AnalysisJobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_stage: str | None = None
    safe_error_category: str | None = None
    cancellation_requested: bool = False
    artifact_count: int = 0

    def __post_init__(self) -> None:
        for value in (self.created_at, self.started_at, self.finished_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("Job timestamp timezone-aware olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_kind": self.source_kind.value,
            "analysis_policy": self.analysis_policy.value,
            "status": self.status.value,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "started_at": self.started_at.astimezone(UTC).isoformat() if self.started_at else None,
            "finished_at": self.finished_at.astimezone(UTC).isoformat() if self.finished_at else None,
            "progress_stage": self.progress_stage,
            "safe_error_category": self.safe_error_category,
            "cancellation_requested": self.cancellation_requested,
            "artifact_count": self.artifact_count,
        }
