from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from generator.scenario_generator import ScenarioRejectionCategory
from models.external_source_analysis_result import ExternalExecutionPolicy, ExternalSourceKind


def normalize_public_input_rejection_categories(
    values: object,
) -> tuple[tuple[str, int], ...]:
    """Internal rejection sayaçlarını güvenli public kategori özetine indirger."""
    if not isinstance(values, tuple):
        return ()
    allowed = {item.value for item in ScenarioRejectionCategory}
    counts: dict[str, int] = {}
    for item in values:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or item[0] not in allowed
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
        ):
            continue
        counts[item[0]] = counts.get(item[0], 0) + item[1]
    return tuple(sorted(counts.items()))


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
    concrete_accepted_count: int | None
    concrete_rejected_count: int | None
    rl_test_count: int | None
    q_table_state_count: int | None
    line_coverage_percent: float | None
    branch_coverage_percent: float | None
    scenario_pool_line_coverage_percent: float | None
    scenario_pool_branch_coverage_percent: float | None
    greedy_line_coverage_percent: float | None
    greedy_branch_coverage_percent: float | None
    greedy_coverage_preserved: bool | None
    best_rl_line_coverage_percent: float | None
    best_rl_branch_coverage_percent: float | None
    best_rl_coverage_preserved: bool | None
    duration_seconds: float | None
    stopped_stage: str | None
    error_category: str | None
    greedy_selected_count: int | None
    rl_selected_count: int | None
    strategy_winner: str | None
    comparison_status: str | None
    comparison_scenario_pool_count: int | None
    greedy_reduction_percentage: float | None
    rl_reduction_percentage: float | None
    coverage_equality_verified: bool | None
    globally_minimal: bool | None
    rl_done_reason: str | None
    bounded_path_count: int = 0
    input_generation_accepted_count: int = 0
    input_generation_rejected_count: int = 0
    input_rejection_categories: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "scenario_count": self.scenario_count,
            "concrete_accepted_count": self.concrete_accepted_count,
            "concrete_rejected_count": self.concrete_rejected_count,
            "rl_test_count": self.rl_test_count,
            "q_table_state_count": self.q_table_state_count,
            "line_coverage_percent": self.line_coverage_percent,
            "branch_coverage_percent": self.branch_coverage_percent,
            "scenario_pool_line_coverage_percent": self.scenario_pool_line_coverage_percent,
            "scenario_pool_branch_coverage_percent": self.scenario_pool_branch_coverage_percent,
            "greedy_line_coverage_percent": self.greedy_line_coverage_percent,
            "greedy_branch_coverage_percent": self.greedy_branch_coverage_percent,
            "greedy_coverage_preserved": self.greedy_coverage_preserved,
            "best_rl_line_coverage_percent": self.best_rl_line_coverage_percent,
            "best_rl_branch_coverage_percent": self.best_rl_branch_coverage_percent,
            "best_rl_coverage_preserved": self.best_rl_coverage_preserved,
            "duration_seconds": self.duration_seconds,
            "stopped_stage": self.stopped_stage,
            "error_category": self.error_category,
            "greedy_selected_count": self.greedy_selected_count,
            "rl_selected_count": self.rl_selected_count,
            "strategy_winner": self.strategy_winner,
            "comparison_status": self.comparison_status,
            "comparison_scenario_pool_count": self.comparison_scenario_pool_count,
            "greedy_reduction_percentage": self.greedy_reduction_percentage,
            "rl_reduction_percentage": self.rl_reduction_percentage,
            "coverage_equality_verified": self.coverage_equality_verified,
            "globally_minimal": self.globally_minimal,
            "rl_done_reason": self.rl_done_reason,
            "bounded_path_count": self.bounded_path_count,
            "input_generation_accepted_count": self.input_generation_accepted_count,
            "input_generation_rejected_count": self.input_generation_rejected_count,
            "input_rejection_categories": [
                {"category": category, "count": count}
                for category, count in normalize_public_input_rejection_categories(
                    self.input_rejection_categories
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class AnalysisModuleSummary:
    relative_path: str
    module_name: str | None
    status: str
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    selection_skipped_function_count: int
    discovered_function_names: tuple[str, ...]
    functions: tuple[AnalysisFunctionSummary, ...]
    deadline_skipped_function_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "module_name": self.module_name,
            "status": self.status,
            "discovered_function_count": self.discovered_function_count,
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "selection_skipped_function_count": self.selection_skipped_function_count,
            "deadline_skipped_function_count": self.deadline_skipped_function_count,
            "discovered_function_names": list(self.discovered_function_names),
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
    selection_skipped_function_count: int
    project_line_coverage_percent: float | None
    project_branch_coverage_percent: float | None
    duration_seconds: float
    cleanup_status: str
    modules: tuple[AnalysisModuleSummary, ...]
    issues: tuple[str, ...]
    project_coverage: dict[str, Any] | None = None
    deadline_skipped_function_count: int = 0
    project_timeout_seconds: float | None = None
    project_deadline_exceeded: bool = False
    last_completed_stage: str | None = None
    deadline_stage: str | None = None
    completed_function_count: int = 0
    partial_function_count: int = 0
    timed_out_function_count: int = 0
    resolved_commit_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_kind": self.source_kind.value,
            "analysis_policy": self.analysis_policy.value,
            "status": self.status.value,
            "acquisition_status": self.acquisition_status,
            "resolved_commit_sha": self.resolved_commit_sha,
            "discovered_module_count": self.discovered_module_count,
            "selected_module_count": self.selected_module_count,
            "discovered_function_count": self.discovered_function_count,
            "analyzed_function_count": self.analyzed_function_count,
            "limit_skipped_function_count": self.limit_skipped_function_count,
            "selection_skipped_function_count": self.selection_skipped_function_count,
            "deadline_skipped_function_count": self.deadline_skipped_function_count,
            "project_line_coverage_percent": self.project_line_coverage_percent,
            "project_branch_coverage_percent": self.project_branch_coverage_percent,
            "project_coverage": self.project_coverage,
            "duration_seconds": self.duration_seconds,
            "cleanup_status": self.cleanup_status,
            "modules": [item.to_dict() for item in self.modules],
            "issues": list(self.issues),
            "project_timeout_seconds": self.project_timeout_seconds,
            "project_deadline_exceeded": self.project_deadline_exceeded,
            "last_completed_stage": self.last_completed_stage,
            "deadline_stage": self.deadline_stage,
            "completed_function_count": self.completed_function_count,
            "partial_function_count": self.partial_function_count,
            "timed_out_function_count": self.timed_out_function_count,
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
    project_timeout_seconds: float | None = None
    project_deadline_exceeded: bool = False
    last_completed_stage: str | None = None
    deadline_stage: str | None = None

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
            "project_timeout_seconds": self.project_timeout_seconds,
            "project_deadline_exceeded": self.project_deadline_exceeded,
            "last_completed_stage": self.last_completed_stage,
            "deadline_stage": self.deadline_stage,
        }
