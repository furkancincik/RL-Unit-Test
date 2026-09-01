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
from models.coverage_progress import CoverageProgressSnapshot
from models.external_source_analysis_result import (
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalAnalysisStatus,
    ExternalAnalysisConfiguration,
    ExternalExecutionPolicy,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
    PinnedGitHubDynamicAuthorization,
    PublicGitHubRepository,
    SourceAcquisitionLimits,
    validate_external_source_execution_policy,
)
from models.project_analysis_result import (
    QualifiedTargetSelector,
    TargetSelectionMode,
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


class PinnedGitHubAuthorizationError(ValueError):
    """Static discovery snapshot'ı pinned dynamic çalışmayı yetkilendirmedi."""


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
    github_discovery_snapshot: _GitHubDiscoverySnapshot | None = None
    artifact_leases: int = 0
    authorization_leases: int = 0
    purge_in_progress: bool = False
    run_generation: int = 0


@dataclass(frozen=True, slots=True)
class _GitHubDiscoverySnapshot:
    repository_url: str
    resolved_commit_sha: str
    selectors: frozenset[QualifiedTargetSelector]


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
        validate_external_source_execution_policy(
            request.source,
            request.execution_policy,
            request.configuration,
            request.pinned_github_authorization,
        )
        authorization = request.pinned_github_authorization
        authorization_lease: _JobEntry | None = None
        if authorization is not None:
            self.purge_expired()
            authorization_lease = self._acquire_pinned_github_authorization(request)
        try:
            return self._submit_validated(request)
        finally:
            if authorization_lease is not None:
                self._release_pinned_github_authorization(
                    authorization.discovery_job_id,
                    authorization_lease,
                )

    def submit_pinned_github_dynamic(
        self,
        *,
        discovery_job_id: str,
        repository_url: str,
        configuration: ExternalAnalysisConfiguration,
        trusted_execution_acknowledged: bool,
        acquisition_limits: SourceAcquisitionLimits = SourceAcquisitionLimits(),
    ) -> AnalysisJobSummary:
        """Başarılı static snapshot'tan exact-SHA dynamic request üretir."""
        if not isinstance(discovery_job_id, str) or not discovery_job_id:
            raise PinnedGitHubAuthorizationError(
                "Başarılı GitHub static discovery job referansı gereklidir."
            )
        if not isinstance(repository_url, str) or not repository_url:
            raise PinnedGitHubAuthorizationError(
                "GitHub repository identity doğrulanamadı."
            )
        if not isinstance(configuration, ExternalAnalysisConfiguration):
            raise TypeError("configuration ExternalAnalysisConfiguration olmalıdır.")
        if trusted_execution_acknowledged is not True:
            raise PinnedGitHubAuthorizationError(
                "Pinned GitHub dynamic açık trust acknowledgement gerektirir."
            )
        if not isinstance(acquisition_limits, SourceAcquisitionLimits):
            raise TypeError("acquisition_limits SourceAcquisitionLimits olmalıdır.")
        if self._shutdown:
            raise AnalysisJobStateConflictError("Job service kapatıldı.")
        self.purge_expired()
        with self._lock:
            try:
                discovery = self._entry(discovery_job_id)
            except AnalysisJobNotFoundError as error:
                raise PinnedGitHubAuthorizationError(
                    "GitHub static discovery snapshot artık mevcut değil."
                ) from error
            snapshot = discovery.github_discovery_snapshot
        if snapshot is None:
            raise PinnedGitHubAuthorizationError(
                "Job başarılı GitHub static discovery snapshot'ı değildir."
            )
        if repository_url.casefold() != snapshot.repository_url.casefold():
            raise PinnedGitHubAuthorizationError(
                "GitHub repository identity discovery snapshot ile eşleşmiyor."
            )
        selection = configuration.target_selection
        if (
            selection.mode is not TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS
            or not selection.selectors
        ):
            raise PinnedGitHubAuthorizationError(
                "Pinned GitHub dynamic analiz explicit target seçimi gerektirir."
            )
        if any(item not in snapshot.selectors for item in selection.selectors):
            raise PinnedGitHubAuthorizationError(
                "Seçilen target GitHub static discovery snapshot'ında bulunmuyor."
            )
        selected_modules = tuple(
            dict.fromkeys(item.module_identity for item in selection.selectors)
        )
        pinned_configuration = replace(
            configuration,
            module_selection=ExternalModuleSelection(
                ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES,
                selected_modules,
            ),
        )
        authorization = PinnedGitHubDynamicAuthorization(
            discovery_job_id=discovery_job_id,
            repository_url=snapshot.repository_url,
            resolved_commit_sha=snapshot.resolved_commit_sha,
            selectors=selection.selectors,
        )
        request = ExternalSourceAnalysisRequest(
            source=PublicGitHubRepository(
                repository_url=snapshot.repository_url,
                ref=snapshot.resolved_commit_sha,
            ),
            execution_policy=ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            configuration=pinned_configuration,
            acquisition_limits=acquisition_limits,
            pinned_github_authorization=authorization,
        )
        return self.submit(request)

    def _submit_validated(
        self,
        request: ExternalSourceAnalysisRequest,
    ) -> AnalysisJobSummary:
        self.purge_expired()
        job_id = uuid.uuid4().hex
        now = self._utc_now()
        self.settings.output_root.mkdir(parents=True, exist_ok=True)
        run_owned_root = self.settings.output_root.resolve()
        job_output = run_owned_root / job_id
        if job_output.parent != run_owned_root:
            raise RuntimeError("Job output root containment ihlali.")
        safe_request = ExternalSourceAnalysisRequest(
            source=request.source,
            execution_policy=request.execution_policy,
            configuration=replace(request.configuration, output_root=job_output),
            acquisition_limits=request.acquisition_limits,
            pinned_github_authorization=request.pinned_github_authorization,
        )
        summary = AnalysisJobSummary(
            job_id=job_id,
            source_kind=request.source.source_kind,
            analysis_policy=request.execution_policy,
            status=AnalysisJobStatus.QUEUED,
            created_at=now,
            progress_stage="QUEUED",
            project_timeout_seconds=request.configuration.project_timeout_seconds,
        )
        entry = _JobEntry(summary=summary, artifacts={})
        with self._lock:
            if self._shutdown:
                raise AnalysisJobStateConflictError("Job service kapatıldı.")
            if not self._capacity.acquire(blocking=False):
                raise AnalysisJobQueueFullError("Analysis job kuyruğu dolu.")
            self._jobs[job_id] = entry
            try:
                future = self._executor.submit(self._run_job, job_id, safe_request)
            except RuntimeError as error:
                self._jobs.pop(job_id, None)
                self._capacity.release()
                if self._shutdown:
                    raise AnalysisJobStateConflictError(
                        "Job service kapatıldı."
                    ) from error
                raise
            entry.future = future
            future.add_done_callback(
                lambda completed, current=job_id: self._on_done(current, completed)
            )
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
        return self._validated_artifact(job_id, artifact_id)

    def acquire_artifact(
        self,
        job_id: str,
        artifact_id: str,
    ) -> tuple[AnalysisArtifactSummary, Path]:
        """Artifact purge'a karşı kısa ömürlü download lease'i edinir."""
        with self._lock:
            entry = self._entry(job_id)
            if entry.purge_in_progress:
                raise AnalysisArtifactNotFoundError("Artifact bulunamadı.")
            metadata, resolved = self._validated_artifact(
                job_id,
                artifact_id,
            )
            entry.artifact_leases += 1
            return metadata, resolved

    def release_artifact(self, job_id: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.artifact_leases <= 0:
                raise AnalysisJobStateConflictError(
                    "Aktif artifact download lease'i bulunamadı."
                )
            entry.artifact_leases -= 1

    def _validated_artifact(
        self,
        job_id: str,
        artifact_id: str,
    ) -> tuple[AnalysisArtifactSummary, Path]:
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
                self._publish_cancelled(entry)
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
        candidates: list[str] = []
        with self._lock:
            for job_id, entry in tuple(self._jobs.items()):
                if (
                    entry.summary.status.terminal
                    and entry.summary.finished_at is not None
                    and entry.summary.finished_at < threshold
                    and entry.artifact_leases == 0
                    and entry.authorization_leases == 0
                    and not entry.purge_in_progress
                ):
                    candidates.append(job_id)
        removed_count = 0
        for job_id in candidates:
            with self._lock:
                entry = self._jobs.get(job_id)
                if (
                    entry is None
                    or not entry.summary.status.terminal
                    or entry.summary.finished_at is None
                    or entry.summary.finished_at >= threshold
                    or entry.artifact_leases != 0
                    or entry.authorization_leases != 0
                    or entry.purge_in_progress
                ):
                    continue
                entry.purge_in_progress = True
            cleanup_succeeded = False
            try:
                self._remove_job_output(job_id)
                cleanup_succeeded = True
            finally:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current is entry:
                        if cleanup_succeeded:
                            self._jobs.pop(job_id, None)
                            removed_count += 1
                        else:
                            entry.purge_in_progress = False
        return removed_count

    def capacity(self) -> tuple[int, int, int]:
        with self._lock:
            running = sum(entry.summary.status is AnalysisJobStatus.RUNNING for entry in self._jobs.values())
            queued = sum(entry.summary.status is AnalysisJobStatus.QUEUED for entry in self._jobs.values())
        return running, queued, self.settings.maximum_running_jobs + self.settings.maximum_queued_jobs

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run_job(self, job_id: str, request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        try:
            with self._lock:
                entry = self._entry(job_id)
                if entry.summary.status is AnalysisJobStatus.CANCELLED:
                    raise AnalysisJobStateConflictError(
                        "Cancelled job çalıştırılamaz."
                    )
                entry.summary = replace(
                    entry.summary,
                    status=AnalysisJobStatus.RUNNING,
                    started_at=self._utc_now(),
                    progress_stage="EXTERNAL_ANALYSIS",
                )
                entry.run_generation += 1
                generation = entry.run_generation
            runner = self._runner_factory()
            if isinstance(runner, ExternalSourceAnalysisService):
                result = runner.run(
                    request,
                    coverage_progress_callback=lambda snapshot: self._publish_coverage_progress(
                        job_id, generation, snapshot
                    ),
                )
            else:
                result = runner.run(request)
            with self._lock:
                entry = self._entry(job_id)
                terminal = self._job_status(result.status)
                artifacts = self._collect_artifacts(job_id, result)
                entry.artifacts = artifacts
                entry.result = self._result_summary(
                    job_id,
                    result,
                    terminal,
                    coverage_progress=entry.summary.coverage_progress,
                )
                entry.github_discovery_snapshot = self._github_discovery_snapshot(
                    request,
                    result,
                    terminal,
                )
                entry.summary = replace(
                    entry.summary,
                    status=terminal,
                    finished_at=self._utc_now(),
                    progress_stage=(
                        result.deadline_stage
                        or terminal.value
                    ),
                    safe_error_category=(result.issues[0] if result.issues and terminal is AnalysisJobStatus.FAILED else None),
                    artifact_count=len(artifacts),
                    project_deadline_exceeded=result.project_deadline_exceeded,
                    last_completed_stage=result.last_completed_stage,
                    deadline_stage=result.deadline_stage,
                )
            return result
        finally:
            try:
                with self._lock:
                    entry = self._jobs.get(job_id)
                    if entry is not None and not entry.summary.status.terminal:
                        self._publish_worker_failure(entry)
            finally:
                self._capacity.release()

    def _publish_coverage_progress(
        self,
        job_id: str,
        generation: int,
        snapshot: CoverageProgressSnapshot,
    ) -> None:
        if not isinstance(snapshot, CoverageProgressSnapshot):
            raise TypeError("snapshot CoverageProgressSnapshot olmalıdır.")
        with self._lock:
            entry = self._jobs.get(job_id)
            if (
                entry is None
                or entry.run_generation != generation
                or entry.summary.status is not AnalysisJobStatus.RUNNING
                or entry.summary.cancellation_requested
                or entry.purge_in_progress
            ):
                return
            current = entry.summary.coverage_progress
            if current is not None and snapshot.revision <= current.revision:
                return
            entry.summary = replace(entry.summary, coverage_progress=snapshot)

    @staticmethod
    def _github_discovery_snapshot(
        request: ExternalSourceAnalysisRequest,
        result: ExternalSourceAnalysisResult,
        terminal: AnalysisJobStatus,
    ) -> _GitHubDiscoverySnapshot | None:
        source = request.source
        sha = result.resolved_commit_sha
        if (
            not isinstance(source, PublicGitHubRepository)
            or request.execution_policy
            is not ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY
            or result.status is not ExternalAnalysisStatus.STATIC_COMPLETED
            or terminal is not AnalysisJobStatus.COMPLETED
            or result.acquisition_status != "COMPLETED"
            or result.cleanup_status
            is not ExternalWorkspaceCleanupStatus.COMPLETED
            or not isinstance(sha, str)
            or len(sha) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in sha)
            or not result.github_owner
            or not result.github_repository
        ):
            return None
        resolved_repository_url = (
            f"https://github.com/{result.github_owner}/{result.github_repository}"
        )
        if source.repository_url.casefold() != resolved_repository_url.casefold():
            return None
        selectors = frozenset(
            QualifiedTargetSelector(module.module_name, qualified_name)
            for module in result.module_results
            if module.module_name is not None
            for qualified_name in module.discovered_function_names
        )
        if not selectors:
            return None
        return _GitHubDiscoverySnapshot(
            repository_url=source.repository_url,
            resolved_commit_sha=sha.lower(),
            selectors=selectors,
        )

    def _acquire_pinned_github_authorization(
        self,
        request: ExternalSourceAnalysisRequest,
    ) -> _JobEntry | None:
        authorization = request.pinned_github_authorization
        if authorization is None:
            return None
        with self._lock:
            try:
                entry = self._entry(authorization.discovery_job_id)
            except AnalysisJobNotFoundError as error:
                raise PinnedGitHubAuthorizationError(
                    "GitHub static discovery snapshot artık mevcut değil."
                ) from error
            snapshot = entry.github_discovery_snapshot
            if (
                entry.purge_in_progress
                or snapshot is None
                or snapshot.repository_url.casefold()
                != authorization.repository_url.casefold()
                or snapshot.resolved_commit_sha
                != authorization.resolved_commit_sha
                or any(
                    item not in snapshot.selectors
                    for item in authorization.selectors
                )
            ):
                raise PinnedGitHubAuthorizationError(
                    "Pinned GitHub authorization registry snapshot ile eşleşmiyor."
                )
            entry.authorization_leases += 1
            return entry

    def _release_pinned_github_authorization(
        self,
        discovery_job_id: str,
        leased_entry: _JobEntry,
    ) -> None:
        with self._lock:
            current = self._jobs.get(discovery_job_id)
            if current is not leased_entry or current.authorization_leases <= 0:
                raise AnalysisJobStateConflictError(
                    "Aktif pinned GitHub authorization lease'i bulunamadı."
                )
            current.authorization_leases -= 1

    def _on_done(self, job_id: str, future: Future[ExternalSourceAnalysisResult]) -> None:
        if future.cancelled():
            with self._lock:
                entry = self._jobs.get(job_id)
                if entry is not None:
                    self._publish_cancelled(entry)
            self._capacity.release()
            return
        error = future.exception()
        if error is None:
            return
        with self._lock:
            entry = self._jobs.get(job_id)
            if (
                entry is None
                or entry.summary.status
                in {AnalysisJobStatus.CANCELLED, AnalysisJobStatus.FAILED}
            ):
                return
            self._publish_worker_failure(entry)

    def _publish_cancelled(self, entry: _JobEntry) -> None:
        entry.summary = replace(
            entry.summary,
            status=AnalysisJobStatus.CANCELLED,
            finished_at=entry.summary.finished_at or self._utc_now(),
            progress_stage="CANCELLED",
            cancellation_requested=True,
        )
        if entry.result is None:
            entry.result = self._safe_empty_terminal_result(
                entry.summary,
                AnalysisJobStatus.CANCELLED,
                "CANCELLED",
                cleanup_status=ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
            )

    def _publish_worker_failure(self, entry: _JobEntry) -> None:
        entry.summary = replace(
            entry.summary,
            status=AnalysisJobStatus.FAILED,
            finished_at=self._utc_now(),
            progress_stage="FAILED",
            safe_error_category="INTERNAL_WORKER_ERROR",
        )
        entry.artifacts = {}
        entry.github_discovery_snapshot = None
        entry.result = self._safe_empty_terminal_result(
            entry.summary,
            AnalysisJobStatus.FAILED,
            "INTERNAL_WORKER_ERROR",
            cleanup_status=ExternalWorkspaceCleanupStatus.FAILED,
        )

    @staticmethod
    def _safe_empty_terminal_result(
        summary: AnalysisJobSummary,
        status: AnalysisJobStatus,
        issue: str,
        *,
        cleanup_status: ExternalWorkspaceCleanupStatus,
    ) -> AnalysisJobResultSummary:
        return AnalysisJobResultSummary(
            job_id=summary.job_id,
            source_kind=summary.source_kind,
            analysis_policy=summary.analysis_policy,
            status=status,
            acquisition_status=None,
            resolved_commit_sha=None,
            discovered_module_count=0,
            selected_module_count=0,
            discovered_function_count=0,
            analyzed_function_count=0,
            limit_skipped_function_count=0,
            selection_skipped_function_count=0,
            deadline_skipped_function_count=0,
            project_line_coverage_percent=None,
            project_branch_coverage_percent=None,
            project_coverage=None,
            duration_seconds=0.0,
            cleanup_status=cleanup_status.value,
            modules=(),
            issues=(issue,),
            project_timeout_seconds=summary.project_timeout_seconds,
            project_deadline_exceeded=summary.project_deadline_exceeded,
            last_completed_stage=summary.last_completed_stage,
            deadline_stage=summary.deadline_stage,
            completed_function_count=0,
            partial_function_count=0,
            timed_out_function_count=0,
            coverage_progress=summary.coverage_progress,
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
        job_id: str,
        result: ExternalSourceAnalysisResult,
        status: AnalysisJobStatus,
        *,
        coverage_progress: CoverageProgressSnapshot | None = None,
    ) -> AnalysisJobResultSummary:
        modules: list[AnalysisModuleSummary] = []
        for module in result.module_results:
            functions: list[AnalysisFunctionSummary] = []
            project = module.project_result
            for function in getattr(project, "function_results", ()) if project is not None else ():
                diagnostic = function.diagnostic
                funnel = getattr(diagnostic, "funnel", None)
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
                        bounded_path_count=AnalysisJobService._public_funnel_count(
                            funnel, "bounded_path_count"
                        ),
                        input_generation_accepted_count=(
                            AnalysisJobService._public_funnel_count(
                                funnel, "input_generation_accepted_count"
                            )
                        ),
                        input_generation_rejected_count=(
                            AnalysisJobService._public_funnel_count(
                                funnel, "input_generation_rejected_count"
                            )
                        ),
                        input_rejection_categories=(
                            getattr(diagnostic, "scenario_rejection_counts", ())
                            if diagnostic is not None
                            else ()
                        ),
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
                    deadline_skipped_function_count=(
                        module.deadline_skipped_function_count
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
            resolved_commit_sha=result.resolved_commit_sha,
            discovered_module_count=result.discovered_module_count,
            selected_module_count=result.selected_module_count,
            discovered_function_count=result.discovered_function_count,
            analyzed_function_count=result.analyzed_function_count,
            limit_skipped_function_count=result.limit_skipped_function_count,
            selection_skipped_function_count=(
                result.selection_skipped_function_count
            ),
            deadline_skipped_function_count=(
                result.deadline_skipped_function_count
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
            project_timeout_seconds=result.project_timeout_seconds,
            project_deadline_exceeded=result.project_deadline_exceeded,
            last_completed_stage=result.last_completed_stage,
            deadline_stage=result.deadline_stage,
            completed_function_count=result.completed_function_count,
            partial_function_count=result.partial_function_count,
            timed_out_function_count=result.timed_out_function_count,
            coverage_progress=coverage_progress,
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

    @staticmethod
    def _public_funnel_count(funnel: object, name: str) -> int:
        value = getattr(funnel, name, None) if funnel is not None else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

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
