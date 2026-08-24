from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

from models.analysis_job_result import (
    AnalysisArtifactSummary,
    AnalysisFunctionSummary,
    AnalysisJobResultSummary,
    AnalysisJobStatus,
    AnalysisJobSummary,
    AnalysisModuleSummary,
)
from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
)
from services.external_source_analysis_service import ExternalSourceAnalysisService


class AnalysisJobNotFoundError(LookupError):
    pass


class AnalysisJobQueueFullError(RuntimeError):
    pass


class AnalysisJobStateConflictError(RuntimeError):
    pass


class AnalysisArtifactNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisJobSettings:
    output_root: Path
    maximum_running_jobs: int = 2
    maximum_queued_jobs: int = 20
    retention_seconds: float = 3600.0
    maximum_upload_bytes: int = 2_000_000
    maximum_inline_source_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root Path olmalıdır.")
        for name in ("maximum_running_jobs", "maximum_queued_jobs", "maximum_upload_bytes", "maximum_inline_source_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} pozitif tam sayı olmalıdır.")
        if not isinstance(self.retention_seconds, (int, float)) or self.retention_seconds <= 0:
            raise ValueError("retention_seconds pozitif olmalıdır.")


@dataclass(slots=True)
class _JobEntry:
    summary: AnalysisJobSummary
    result: AnalysisJobResultSummary | None = None
    future: Future[ExternalSourceAnalysisResult] | None = None
    artifacts: dict[str, tuple[AnalysisArtifactSummary, Path]] | None = None


class AnalysisJobService:
    """External analysis işlerini bounded executor ve güvenli registry ile yürütür."""

    def __init__(
        self,
        *,
        settings: AnalysisJobSettings,
        runner_factory: Callable[[], ExternalSourceAnalysisService] = ExternalSourceAnalysisService,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self._runner_factory = runner_factory
        self._clock = clock
        self._lock = threading.RLock()
        self._capacity = threading.BoundedSemaphore(
            settings.maximum_running_jobs + settings.maximum_queued_jobs
        )
        self._executor = ThreadPoolExecutor(
            max_workers=settings.maximum_running_jobs,
            thread_name_prefix="rl-unit-test-analysis",
        )
        self._jobs: dict[str, _JobEntry] = {}
        self._shutdown = False

    def submit(self, request: ExternalSourceAnalysisRequest) -> AnalysisJobSummary:
        if not isinstance(request, ExternalSourceAnalysisRequest):
            raise TypeError("request ExternalSourceAnalysisRequest olmalıdır.")
        if self._shutdown:
            raise AnalysisJobStateConflictError("Job service kapatıldı.")
        self.purge_expired()
        if not self._capacity.acquire(blocking=False):
            raise AnalysisJobQueueFullError("Analysis job kuyruğu dolu.")
        job_id = uuid.uuid4().hex
        now = self._utc_now()
        job_output = (self.settings.output_root.resolve() / job_id).resolve()
        if not job_output.is_relative_to(self.settings.output_root.resolve()):
            self._capacity.release()
            raise RuntimeError("Job output root containment ihlali.")
        safe_request = ExternalSourceAnalysisRequest(
            source=request.source,
            execution_policy=request.execution_policy,
            configuration=replace(request.configuration, output_root=job_output),
            acquisition_limits=request.acquisition_limits,
        )
        summary = AnalysisJobSummary(
            job_id=job_id,
            source_kind=request.source.source_kind,
            analysis_policy=request.execution_policy,
            status=AnalysisJobStatus.QUEUED,
            created_at=now,
            progress_stage="QUEUED",
        )
        entry = _JobEntry(summary=summary, artifacts={})
        with self._lock:
            self._jobs[job_id] = entry
        try:
            future = self._executor.submit(self._run_job, job_id, safe_request)
        except RuntimeError:
            with self._lock:
                self._jobs.pop(job_id, None)
            self._capacity.release()
            raise
        entry.future = future
        future.add_done_callback(lambda completed, current=job_id: self._on_done(current, completed))
        return summary

    def get(self, job_id: str) -> AnalysisJobSummary:
        with self._lock:
            return self._entry(job_id).summary

    def get_result(self, job_id: str) -> AnalysisJobResultSummary:
        with self._lock:
            entry = self._entry(job_id)
            if entry.result is None:
                raise AnalysisJobStateConflictError("Job sonucu henüz hazır değil.")
            return entry.result

    def list_artifacts(self, job_id: str) -> tuple[AnalysisArtifactSummary, ...]:
        with self._lock:
            entry = self._entry(job_id)
            if not entry.summary.status.terminal:
                raise AnalysisJobStateConflictError("Job artifact'leri henüz hazır değil.")
            return tuple(item[0] for item in (entry.artifacts or {}).values())

    def artifact_path(self, job_id: str, artifact_id: str) -> tuple[AnalysisArtifactSummary, Path]:
        with self._lock:
            entry = self._entry(job_id)
            value = (entry.artifacts or {}).get(artifact_id)
            if value is None:
                raise AnalysisArtifactNotFoundError("Artifact bulunamadı.")
            metadata, path = value
        root = (self.settings.output_root.resolve() / job_id).resolve()
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
            raise AnalysisArtifactNotFoundError("Artifact güvenli output root içinde değil.")
        return metadata, resolved

    def cancel(self, job_id: str) -> AnalysisJobSummary:
        with self._lock:
            entry = self._entry(job_id)
            status = entry.summary.status
            if status is AnalysisJobStatus.QUEUED and entry.future is not None and entry.future.cancel():
                entry.summary = replace(
                    entry.summary,
                    status=AnalysisJobStatus.CANCELLED,
                    finished_at=self._utc_now(),
                    progress_stage="CANCELLED",
                    cancellation_requested=True,
                )
                return entry.summary
            if status is AnalysisJobStatus.RUNNING:
                entry.summary = replace(entry.summary, cancellation_requested=True)
                raise AnalysisJobStateConflictError(
                    "Running job cancellation desteklenmiyor; pipeline timeout geçerlidir."
                )
            raise AnalysisJobStateConflictError("Terminal job iptal edilemez.")

    def wait(self, job_id: str, timeout: float | None = None) -> ExternalSourceAnalysisResult:
        with self._lock:
            future = self._entry(job_id).future
        if future is None:
            raise AnalysisJobStateConflictError("Job future mevcut değil.")
        return future.result(timeout=timeout)

    def purge_expired(self) -> int:
        threshold = self._utc_now() - timedelta(seconds=float(self.settings.retention_seconds))
        removed: list[str] = []
        with self._lock:
            for job_id, entry in tuple(self._jobs.items()):
                if (
                    entry.summary.status.terminal
                    and entry.summary.finished_at is not None
                    and entry.summary.finished_at < threshold
                ):
                    self._jobs.pop(job_id)
                    removed.append(job_id)
        for job_id in removed:
            self._remove_job_output(job_id)
        return len(removed)

    def capacity(self) -> tuple[int, int, int]:
        with self._lock:
            running = sum(entry.summary.status is AnalysisJobStatus.RUNNING for entry in self._jobs.values())
            queued = sum(entry.summary.status is AnalysisJobStatus.QUEUED for entry in self._jobs.values())
        return running, queued, self.settings.maximum_running_jobs + self.settings.maximum_queued_jobs

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run_job(self, job_id: str, request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        with self._lock:
            entry = self._entry(job_id)
            if entry.summary.status is AnalysisJobStatus.CANCELLED:
                raise AnalysisJobStateConflictError("Cancelled job çalıştırılamaz.")
            entry.summary = replace(
                entry.summary,
                status=AnalysisJobStatus.RUNNING,
                started_at=self._utc_now(),
                progress_stage="EXTERNAL_ANALYSIS",
            )
        try:
            result = self._runner_factory().run(request)
            with self._lock:
                entry = self._entry(job_id)
                terminal = self._job_status(result.status)
                artifacts = self._collect_artifacts(job_id, result)
                entry.artifacts = artifacts
                entry.result = self._result_summary(job_id, result, terminal)
                entry.summary = replace(
                    entry.summary,
                    status=terminal,
                    finished_at=self._utc_now(),
                    progress_stage=terminal.value,
                    safe_error_category=(result.issues[0] if result.issues and terminal is AnalysisJobStatus.FAILED else None),
                    artifact_count=len(artifacts),
                )
            return result
        finally:
            self._capacity.release()

    def _on_done(self, job_id: str, future: Future[ExternalSourceAnalysisResult]) -> None:
        if future.cancelled():
            self._capacity.release()
            return
        error = future.exception()
        if error is None:
            return
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.summary.status is AnalysisJobStatus.CANCELLED:
                return
            entry.summary = replace(
                entry.summary,
                status=AnalysisJobStatus.FAILED,
                finished_at=self._utc_now(),
                progress_stage="FAILED",
                safe_error_category="INTERNAL_WORKER_ERROR",
            )

    def _collect_artifacts(
        self, job_id: str, result: ExternalSourceAnalysisResult
    ) -> dict[str, tuple[AnalysisArtifactSummary, Path]]:
        root = (self.settings.output_root.resolve() / job_id).resolve()
        candidates = [result.report_path]
        if result.project_coverage is not None:
            candidates.extend(result.project_coverage.artifact_paths)
        for module in result.module_results:
            candidates.extend(module.artifact_paths)
        artifacts: dict[str, tuple[AnalysisArtifactSummary, Path]] = {}
        for path in dict.fromkeys(Path(item) for item in candidates):
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            if resolved.suffix.lower() == ".json":
                content_type = "application/json"
            elif resolved.suffix.lower() == ".py" and resolved.name.startswith(("test_", "generated_test")):
                content_type = "text/x-python"
            else:
                continue
            artifact_id = uuid.uuid4().hex
            metadata = AnalysisArtifactSummary(
                artifact_id=artifact_id,
                filename=resolved.name,
                content_type=content_type,
                size_bytes=resolved.stat().st_size,
            )
            artifacts[artifact_id] = (metadata, resolved)
        return artifacts

    @staticmethod
    def _result_summary(
        job_id: str, result: ExternalSourceAnalysisResult, status: AnalysisJobStatus
    ) -> AnalysisJobResultSummary:
        modules: list[AnalysisModuleSummary] = []
        for module in result.module_results:
            functions: list[AnalysisFunctionSummary] = []
            project = module.project_result
            for function in getattr(project, "function_results", ()) if project is not None else ():
                diagnostic = function.diagnostic
                comparison = function.strategy_comparison
                scenario_pool = function.scenario_pool_coverage
                minimization = function.minimization_result
                best_rl_coverage = function.best_rl_coverage
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
                best_trace = next(
                    (
                        trace
                        for trace in getattr(comparison, "episode_traces", ())
                        if trace.episode_number
                        == getattr(comparison, "best_rl_episode_number", None)
                    ),
                    None,
                )
                functions.append(
                    AnalysisFunctionSummary(
                        qualified_name=function.target.qualified_name,
                        status=function.status.value,
                        skip_reason=function.skip_reason,
                        scenario_count=function.scenario_count,
                        concrete_accepted_count=function.concrete_accepted_count,
                        concrete_rejected_count=function.concrete_rejected_count,
                        rl_test_count=function.rl_test_count,
                        q_table_state_count=function.q_table_state_count,
                        line_coverage_percent=scenario_pool_line,
                        branch_coverage_percent=scenario_pool_branch,
                        scenario_pool_line_coverage_percent=scenario_pool_line,
                        scenario_pool_branch_coverage_percent=scenario_pool_branch,
                        greedy_line_coverage_percent=getattr(
                            minimization, "final_verified_line_percentage", None
                        ),
                        greedy_branch_coverage_percent=getattr(
                            minimization, "final_verified_branch_percentage", None
                        ),
                        greedy_coverage_preserved=getattr(
                            minimization, "coverage_preserved", None
                        ),
                        best_rl_line_coverage_percent=getattr(
                            best_rl_coverage,
                            "line_coverage_percent",
                            None,
                        ) if verified_rl_line is None else verified_rl_line,
                        best_rl_branch_coverage_percent=getattr(
                            best_rl_coverage,
                            "branch_coverage_percent",
                            None,
                        ) if verified_rl_branch is None else verified_rl_branch,
                        best_rl_coverage_preserved=(
                            getattr(
                                comparison,
                                "rl_coverage_preserved",
                                function.best_rl_coverage_preserved,
                            )
                        ),
                        duration_seconds=getattr(diagnostic, "total_duration_seconds", None),
                        stopped_stage=getattr(
                            getattr(diagnostic, "stopped_stage", None), "value", None
                        ),
                        error_category=getattr(diagnostic, "error_category", None),
                        greedy_selected_count=getattr(
                            minimization, "final_selected_count", None
                        ),
                        rl_selected_count=(
                            getattr(
                                comparison,
                                "best_rl_executed_test_count",
                                None,
                            )
                            or function.rl_test_count
                        ),
                        strategy_winner=getattr(getattr(comparison, "winner", None), "value", None),
                        comparison_status=getattr(
                            getattr(comparison, "status", None), "value", None
                        ),
                        comparison_scenario_pool_count=getattr(
                            comparison,
                            "scenario_pool_count",
                            getattr(minimization, "full_pool_scenario_count", None),
                        ),
                        greedy_reduction_percentage=getattr(
                            minimization, "reduction_percentage", None
                        ),
                        rl_reduction_percentage=getattr(
                            comparison, "rl_reduction_percentage", None
                        ),
                        coverage_equality_verified=getattr(
                            comparison, "coverage_equality_verified", None
                        ),
                        globally_minimal=getattr(
                            comparison,
                            "globally_minimal",
                            getattr(minimization, "globally_minimal", None),
                        ),
                        rl_done_reason=getattr(best_trace, "done_reason", None),
                    )
                )
            modules.append(
                AnalysisModuleSummary(
                    relative_path=module.relative_path,
                    module_name=module.module_name,
                    status=module.status.value,
                    discovered_function_count=module.discovered_function_count,
                    analyzed_function_count=module.analyzed_function_count,
                    limit_skipped_function_count=module.limit_skipped_function_count,
                    selection_skipped_function_count=(
                        module.selection_skipped_function_count
                    ),
                    discovered_function_names=module.discovered_function_names,
                    functions=tuple(functions),
                )
            )
        return AnalysisJobResultSummary(
            job_id=job_id,
            source_kind=result.source_kind,
            analysis_policy=result.execution_policy,
            status=status,
            acquisition_status=result.acquisition_status,
            discovered_module_count=result.discovered_module_count,
            selected_module_count=result.selected_module_count,
            discovered_function_count=result.discovered_function_count,
            analyzed_function_count=result.analyzed_function_count,
            limit_skipped_function_count=result.limit_skipped_function_count,
            selection_skipped_function_count=(
                result.selection_skipped_function_count
            ),
            project_line_coverage_percent=(
                result.project_coverage.full_line_coverage_percent
                if result.project_coverage is not None
                else None
            ),
            project_branch_coverage_percent=(
                result.project_coverage.full_branch_coverage_percent
                if result.project_coverage is not None
                else None
            ),
            duration_seconds=result.duration_seconds,
            cleanup_status=result.cleanup_status.value,
            modules=tuple(modules),
            issues=result.issues,
            project_coverage=(
                result.project_coverage.to_dict()
                if result.project_coverage is not None
                else None
            ),
        )

    @staticmethod
    def _job_status(status: ExternalAnalysisStatus) -> AnalysisJobStatus:
        return {
            ExternalAnalysisStatus.STATIC_COMPLETED: AnalysisJobStatus.COMPLETED,
            ExternalAnalysisStatus.COMPLETED: AnalysisJobStatus.COMPLETED,
            ExternalAnalysisStatus.PARTIAL: AnalysisJobStatus.PARTIAL,
            ExternalAnalysisStatus.FAILED: AnalysisJobStatus.FAILED,
            ExternalAnalysisStatus.TIMED_OUT: AnalysisJobStatus.TIMED_OUT,
        }[status]

    def _entry(self, job_id: str) -> _JobEntry:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise AnalysisJobNotFoundError("Analysis job bulunamadı.") from error

    def _remove_job_output(self, job_id: str) -> None:
        root = self.settings.output_root.resolve()
        target = (root / job_id).resolve()
        try:
            uuid.UUID(hex=job_id)
        except ValueError as error:
            raise RuntimeError("Job output ID güvenli değil.") from error
        if target.parent != root:
            raise RuntimeError("Job output cleanup containment ihlali.")
        if target.exists():
            shutil.rmtree(target)

    def _utc_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock timezone-aware datetime üretmelidir.")
        return value.astimezone(UTC)
