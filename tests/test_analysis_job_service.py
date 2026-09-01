from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from models.analysis_job_result import AnalysisJobStatus
from models.coverage_progress import CoverageProgressSnapshot
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleStatus,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalSourceKind,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    PinnedGitHubDynamicAuthorization,
    PublicGitHubRepository,
)
from models.project_analysis_result import (
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)
from services.analysis_job_service import (
    AnalysisJobNotFoundError,
    AnalysisJobQueueFullError,
    AnalysisJobService,
    AnalysisJobSettings,
    AnalysisJobStateConflictError,
    PinnedGitHubAuthorizationError,
)
from services.external_source_analysis_service import ExternalSourceAnalysisService


def _request(output: Path) -> ExternalSourceAnalysisRequest:
    return ExternalSourceAnalysisRequest(
        InlinePythonSource("def target():\n    return 1\n"),
        ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        ExternalAnalysisConfiguration(output_root=output),
    )


def _result(output: Path, status: ExternalAnalysisStatus = ExternalAnalysisStatus.STATIC_COMPLETED) -> ExternalSourceAnalysisResult:
    output.mkdir(parents=True, exist_ok=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    return ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.INLINE_PYTHON_SOURCE,
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        status=status,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=0,
        selected_module_count=0,
        module_results=(),
        output_root=output,
        report_path=report,
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


_GITHUB_REPOSITORY_URL = "https://github.com/owner/repository"
_GITHUB_COMMIT_SHA = "a" * 40
_GITHUB_SELECTOR = QualifiedTargetSelector("package.module", "target")


def _github_request(output: Path) -> ExternalSourceAnalysisRequest:
    return ExternalSourceAnalysisRequest(
        PublicGitHubRepository(_GITHUB_REPOSITORY_URL),
        ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        ExternalAnalysisConfiguration(output_root=output),
    )


def _github_result(
    output: Path,
    *,
    status: ExternalAnalysisStatus = ExternalAnalysisStatus.STATIC_COMPLETED,
    acquisition_status: str = "COMPLETED",
    cleanup_status: ExternalWorkspaceCleanupStatus = ExternalWorkspaceCleanupStatus.COMPLETED,
) -> ExternalSourceAnalysisResult:
    output.mkdir(parents=True, exist_ok=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    module = ExternalModuleAnalysisResult(
        relative_path="package/module.py",
        module_name=_GITHUB_SELECTOR.module_identity,
        status=ExternalModuleStatus.STATIC_ONLY,
        discovered_function_count=1,
        project_result=None,
        issue_category=None,
        issue_message=None,
        artifact_paths=(),
        discovered_function_names=(_GITHUB_SELECTOR.qualified_name,),
    )
    return ExternalSourceAnalysisResult(
        source_kind=ExternalSourceKind.PUBLIC_GITHUB_REPOSITORY,
        execution_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        status=status,
        acquisition_status=acquisition_status,
        repository_name="repository",
        github_owner="owner",
        github_repository="repository",
        resolved_commit_sha=_GITHUB_COMMIT_SHA,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
        output_root=output,
        report_path=report,
        duration_seconds=0.1,
        cleanup_status=cleanup_status,
        issues=(),
    )


def _pinned_configuration(output: Path) -> ExternalAnalysisConfiguration:
    return ExternalAnalysisConfiguration(
        output_root=output,
        target_selection=TargetSelection(
            TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
            (_GITHUB_SELECTOR,),
        ),
    )


def test_job_runs_once_and_reaches_terminal_state(tmp_path: Path) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _result(request.configuration.output_root)
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path, maximum_running_jobs=1, maximum_queued_jobs=1),
        runner_factory=Mock(return_value=runner),
    )
    job = service.submit(_request(tmp_path / "ignored"))

    service.wait(job.job_id, timeout=5)
    completed = service.get(job.job_id)

    assert completed.status is AnalysisJobStatus.COMPLETED
    assert runner.run.call_count == 1
    assert completed.started_at is not None
    assert completed.finished_at is not None
    service.shutdown()


def test_submission_uses_one_canonical_run_owned_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "jobs"
    canonical_root = output_root.absolute()
    original_resolve = Path.resolve
    root_resolution_count = 0
    started = Event()
    release = Event()
    requests: list[ExternalSourceAnalysisRequest] = []

    def resolve_once(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal root_resolution_count
        if path == output_root:
            root_resolution_count += 1
            if root_resolution_count > 1:
                raise AssertionError(
                    "Run-owned output root submission sırasında yeniden resolve edildi."
                )
            return canonical_root
        return original_resolve(path, *args, **kwargs)

    runner = Mock()

    def run(
        request: ExternalSourceAnalysisRequest,
    ) -> ExternalSourceAnalysisResult:
        requests.append(request)
        started.set()
        release.wait(5)
        return _result(request.configuration.output_root)

    runner.run.side_effect = run
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=output_root),
        runner_factory=Mock(return_value=runner),
    )
    monkeypatch.setattr(Path, "resolve", resolve_once)
    job = None
    try:
        job = service.submit(_request(tmp_path / "ignored"))
        assert started.wait(2)
        assert root_resolution_count == 1
        assert requests[0].configuration.output_root.parent == canonical_root
    finally:
        monkeypatch.undo()
        release.set()
        if job is not None:
            service.wait(job.job_id, timeout=5)
        service.shutdown()


def _progress(revision: int, coverage_percent: float) -> CoverageProgressSnapshot:
    covered = int(coverage_percent // 25)
    return CoverageProgressSnapshot(
        revision=revision,
        stage="COVERAGE_OPTIMIZATION",
        metric="LINE",
        coverage_percent=coverage_percent,
        line_percent=coverage_percent,
        branch_percent=None,
        covered_lines=covered,
        total_lines=4,
        covered_branches=0,
        total_branches=0,
        candidate_count=4,
        validated_count=min(covered, 4),
        effective_test_count=min(covered, 4),
        last_gain_percent=25.0 if covered else 0.0,
        last_new_line_count=1 if covered else 0,
        last_new_branch_count=0,
        plateau_count=0,
        stop_reason=None,
    )


def test_real_runner_progress_is_published_monotonically_and_reaches_result(
    tmp_path: Path,
) -> None:
    started = Event()
    release = Event()

    class ProgressRunner(ExternalSourceAnalysisService):
        def __init__(self) -> None:
            pass

        def run(self, request, *, coverage_progress_callback):
            coverage_progress_callback(_progress(1, 25.0))
            coverage_progress_callback(_progress(2, 50.0))
            started.set()
            assert release.wait(5)
            return _result(request.configuration.output_root)

    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=ProgressRunner,
    )
    job = service.submit(_request(tmp_path / "ignored"))
    assert started.wait(5)
    snapshot = service.get(job.job_id)
    assert snapshot.coverage_progress == _progress(2, 50.0)
    release.set()
    service.wait(job.job_id, timeout=5)
    assert service.get_result(job.job_id).coverage_progress == _progress(2, 50.0)
    service.shutdown()


def test_progress_rejects_stale_revision_and_ignores_terminal_callback(
    tmp_path: Path,
) -> None:
    callbacks = []

    class ProgressRunner(ExternalSourceAnalysisService):
        def __init__(self) -> None:
            pass

        def run(self, request, *, coverage_progress_callback):
            callbacks.append(coverage_progress_callback)
            coverage_progress_callback(_progress(2, 50.0))
            coverage_progress_callback(_progress(1, 25.0))
            return _result(request.configuration.output_root)

    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=ProgressRunner,
    )
    job = service.submit(_request(tmp_path / "ignored"))
    service.wait(job.job_id, timeout=5)
    callbacks[0](_progress(3, 75.0))
    assert service.get(job.job_id).coverage_progress == _progress(2, 50.0)
    service.shutdown()


def test_worker_failure_preserves_last_valid_progress_snapshot(tmp_path: Path) -> None:
    class FailingProgressRunner(ExternalSourceAnalysisService):
        def __init__(self) -> None:
            pass

        def run(self, request, *, coverage_progress_callback):
            coverage_progress_callback(_progress(1, 25.0))
            raise RuntimeError("internal failure")

    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=FailingProgressRunner,
    )
    job = service.submit(_request(tmp_path / "ignored"))
    with pytest.raises(RuntimeError, match="internal failure"):
        service.wait(job.job_id, timeout=5)

    assert service.get(job.job_id).coverage_progress == _progress(1, 25.0)
    assert service.get_result(job.job_id).coverage_progress == _progress(1, 25.0)
    service.shutdown()


def test_invalid_progress_callback_payload_fails_safely_and_preserves_snapshot(
    tmp_path: Path,
) -> None:
    private_detail = "private callback object detail"

    class InvalidProgressRunner(ExternalSourceAnalysisService):
        def __init__(self) -> None:
            pass

        def run(self, request, *, coverage_progress_callback):
            coverage_progress_callback(_progress(1, 25.0))
            coverage_progress_callback(object())
            raise AssertionError(private_detail)

    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=InvalidProgressRunner,
    )
    job = service.submit(_request(tmp_path / "ignored"))
    with pytest.raises(TypeError, match="CoverageProgressSnapshot"):
        service.wait(job.job_id, timeout=5)

    summary = service.get(job.job_id)
    result = service.get_result(job.job_id)
    assert summary.status is AnalysisJobStatus.FAILED
    assert summary.safe_error_category == "INTERNAL_WORKER_ERROR"
    assert summary.coverage_progress == _progress(1, 25.0)
    assert result.coverage_progress == _progress(1, 25.0)
    serialized = repr((summary.to_dict(), result.to_dict()))
    assert "TypeError" not in serialized
    assert private_detail not in serialized
    assert "private callback object" not in serialized
    service.shutdown()


def test_running_cancellation_request_ignores_later_progress_and_preserves_snapshot(
    tmp_path: Path,
) -> None:
    callback_ready = Event()
    release = Event()
    callbacks = []

    class ProgressRunner(ExternalSourceAnalysisService):
        def __init__(self) -> None:
            pass

        def run(self, request, *, coverage_progress_callback):
            callbacks.append(coverage_progress_callback)
            coverage_progress_callback(_progress(1, 25.0))
            callback_ready.set()
            assert release.wait(5)
            return _result(request.configuration.output_root)

    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=ProgressRunner,
    )
    job = service.submit(_request(tmp_path / "ignored"))
    assert callback_ready.wait(5)
    with pytest.raises(AnalysisJobStateConflictError, match="desteklenmiyor"):
        service.cancel(job.job_id)

    callbacks[0](_progress(2, 50.0))
    assert service.get(job.job_id).coverage_progress == _progress(1, 25.0)
    release.set()
    service.wait(job.job_id, timeout=5)
    assert service.get_result(job.job_id).coverage_progress == _progress(1, 25.0)
    service.shutdown()


def test_legacy_fake_runner_does_not_require_progress_keyword(tmp_path: Path) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _result(request.configuration.output_root)
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    job = service.submit(_request(tmp_path / "ignored"))
    service.wait(job.job_id, timeout=5)
    runner.run.assert_called_once()
    assert runner.run.call_args.kwargs == {}
    service.shutdown()


def test_public_github_dynamic_policy_is_rejected_at_domain_and_service_boundaries(
    tmp_path: Path,
) -> None:
    configuration = ExternalAnalysisConfiguration(output_root=tmp_path)
    with pytest.raises(ValueError, match="yalnız statik"):
        ExternalSourceAnalysisRequest(
            PublicGitHubRepository("https://github.com/owner/repository"),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            configuration,
        )

    request = ExternalSourceAnalysisRequest(
        PublicGitHubRepository("https://github.com/owner/repository"),
        ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        configuration,
    )
    object.__setattr__(
        request,
        "execution_policy",
        ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
    )
    runner = Mock()
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    before = service.capacity()

    with pytest.raises(ValueError, match="yalnız statik"):
        service.submit(request)

    assert service.capacity() == before
    runner.run.assert_not_called()
    service.shutdown()


def test_pinned_github_dynamic_requires_explicit_trust_acknowledgement(
    tmp_path: Path,
) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _github_result(
        request.configuration.output_root
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    discovery = service.submit(_github_request(tmp_path / "discovery"))
    service.wait(discovery.job_id, timeout=5)
    before = service.capacity()

    with pytest.raises(PinnedGitHubAuthorizationError, match="acknowledgement"):
        service.submit_pinned_github_dynamic(
            discovery_job_id=discovery.job_id,
            repository_url=_GITHUB_REPOSITORY_URL,
            configuration=_pinned_configuration(tmp_path / "dynamic"),
            trusted_execution_acknowledged=False,
        )

    assert service.capacity() == before
    assert runner.run.call_count == 1
    service.shutdown()


@pytest.mark.parametrize(
    ("status", "cleanup_status"),
    (
        (ExternalAnalysisStatus.FAILED, ExternalWorkspaceCleanupStatus.COMPLETED),
        (ExternalAnalysisStatus.PARTIAL, ExternalWorkspaceCleanupStatus.COMPLETED),
        (ExternalAnalysisStatus.TIMED_OUT, ExternalWorkspaceCleanupStatus.COMPLETED),
        (ExternalAnalysisStatus.STATIC_COMPLETED, ExternalWorkspaceCleanupStatus.FAILED),
    ),
)
def test_unsuccessful_github_discovery_never_mints_dynamic_authorization(
    tmp_path: Path,
    status: ExternalAnalysisStatus,
    cleanup_status: ExternalWorkspaceCleanupStatus,
) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _github_result(
        request.configuration.output_root,
        status=status,
        cleanup_status=cleanup_status,
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    discovery = service.submit(_github_request(tmp_path / "discovery"))
    service.wait(discovery.job_id, timeout=5)
    before = service.capacity()

    with pytest.raises(
        PinnedGitHubAuthorizationError,
        match="başarılı GitHub static discovery",
    ):
        service.submit_pinned_github_dynamic(
            discovery_job_id=discovery.job_id,
            repository_url=_GITHUB_REPOSITORY_URL,
            configuration=_pinned_configuration(tmp_path / "dynamic"),
            trusted_execution_acknowledged=True,
        )

    assert service.capacity() == before
    assert runner.run.call_count == 1
    service.shutdown()


def test_expired_github_discovery_cannot_authorize_later_dynamic_job(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    runner = Mock()
    runner.run.side_effect = lambda request: _github_result(
        request.configuration.output_root
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path, retention_seconds=10),
        runner_factory=Mock(return_value=runner),
        clock=lambda: now[0],
    )
    discovery = service.submit(_github_request(tmp_path / "discovery"))
    service.wait(discovery.job_id, timeout=5)
    now[0] += timedelta(seconds=11)

    with pytest.raises(PinnedGitHubAuthorizationError, match="artık mevcut değil"):
        service.submit_pinned_github_dynamic(
            discovery_job_id=discovery.job_id,
            repository_url=_GITHUB_REPOSITORY_URL,
            configuration=_pinned_configuration(tmp_path / "dynamic"),
            trusted_execution_acknowledged=True,
        )

    assert runner.run.call_count == 1
    service.shutdown()


def test_direct_forged_pinned_authorization_must_match_registry_snapshot(
    tmp_path: Path,
) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _github_result(
        request.configuration.output_root
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    discovery = service.submit(_github_request(tmp_path / "discovery"))
    service.wait(discovery.job_id, timeout=5)
    forged_sha = "b" * 40
    authorization = PinnedGitHubDynamicAuthorization(
        discovery_job_id=discovery.job_id,
        repository_url=_GITHUB_REPOSITORY_URL,
        resolved_commit_sha=forged_sha,
        selectors=(_GITHUB_SELECTOR,),
    )
    request = ExternalSourceAnalysisRequest(
        PublicGitHubRepository(_GITHUB_REPOSITORY_URL, ref=forged_sha),
        ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
        _pinned_configuration(tmp_path / "dynamic"),
        pinned_github_authorization=authorization,
    )
    before = service.capacity()

    with pytest.raises(PinnedGitHubAuthorizationError, match="registry snapshot"):
        service.submit(request)

    assert service.capacity() == before
    assert runner.run.call_count == 1
    service.shutdown()


def test_pinned_github_authorization_is_omitted_from_public_request_payload(
    tmp_path: Path,
) -> None:
    authorization = PinnedGitHubDynamicAuthorization(
        discovery_job_id="f" * 32,
        repository_url=_GITHUB_REPOSITORY_URL,
        resolved_commit_sha=_GITHUB_COMMIT_SHA,
        selectors=(_GITHUB_SELECTOR,),
    )
    request = ExternalSourceAnalysisRequest(
        PublicGitHubRepository(
            _GITHUB_REPOSITORY_URL,
            ref=_GITHUB_COMMIT_SHA,
        ),
        ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
        _pinned_configuration(tmp_path),
        pinned_github_authorization=authorization,
    )

    public_payload = request.to_dict()
    serialized = repr(public_payload)

    assert "pinned_github_authorization" not in public_payload
    assert "authorization" not in serialized.casefold()
    assert authorization.discovery_job_id not in serialized


def test_queue_is_bounded_and_queued_job_can_be_cancelled(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    runner = Mock()

    def run(request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        started.set()
        release.wait(5)
        return _result(request.configuration.output_root)

    runner.run.side_effect = run
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path, maximum_running_jobs=1, maximum_queued_jobs=1),
        runner_factory=Mock(return_value=runner),
    )
    first = service.submit(_request(tmp_path / "one"))
    assert started.wait(2)
    second = service.submit(_request(tmp_path / "two"))
    with pytest.raises(AnalysisJobQueueFullError):
        service.submit(_request(tmp_path / "three"))

    cancelled = service.cancel(second.job_id)
    assert cancelled.status is AnalysisJobStatus.CANCELLED
    release.set()
    service.wait(first.job_id, timeout=5)
    assert runner.run.call_count == 1
    service.shutdown()


def test_running_and_completed_cancellation_are_not_faked(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    runner = Mock()
    runner.run.side_effect = lambda request: (started.set(), release.wait(5), _result(request.configuration.output_root))[-1]
    service = AnalysisJobService(settings=AnalysisJobSettings(output_root=tmp_path), runner_factory=Mock(return_value=runner))
    job = service.submit(_request(tmp_path / "job"))
    assert started.wait(2)
    with pytest.raises(AnalysisJobStateConflictError):
        service.cancel(job.job_id)
    assert service.get(job.job_id).cancellation_requested is True
    release.set()
    service.wait(job.job_id, timeout=5)
    with pytest.raises(AnalysisJobStateConflictError):
        service.cancel(job.job_id)
    service.shutdown()


def test_retention_removes_only_terminal_jobs_idempotently(tmp_path: Path) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    runner = Mock()
    runner.run.side_effect = lambda request: _result(request.configuration.output_root)
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path, retention_seconds=10),
        runner_factory=Mock(return_value=runner),
        clock=lambda: now[0],
    )
    job = service.submit(_request(tmp_path / "source"))
    service.wait(job.job_id, timeout=5)
    now[0] += timedelta(seconds=11)

    assert service.purge_expired() == 1
    assert service.purge_expired() == 0
    assert not (tmp_path / job.job_id).exists()
    service.shutdown()


@pytest.mark.parametrize(
    "error",
    (
        AssertionError("bug"), TypeError("bug"), RuntimeError("bug"),
        ValueError("bug"), SystemExit("bug"), KeyboardInterrupt("bug"),
    ),
)
def test_worker_programming_failure_is_sanitized_but_re_raisable(
    tmp_path: Path, error: BaseException
) -> None:
    runner = Mock()
    runner.run.side_effect = error
    service = AnalysisJobService(settings=AnalysisJobSettings(output_root=tmp_path), runner_factory=Mock(return_value=runner))
    job = service.submit(_request(tmp_path / "source"))
    with pytest.raises(type(error), match="bug"):
        service.wait(job.job_id, timeout=5)
    snapshot = service.get(job.job_id)
    assert snapshot.status is AnalysisJobStatus.FAILED
    assert snapshot.safe_error_category == "INTERNAL_WORKER_ERROR"
    assert "bug" not in repr(snapshot.to_dict())
    service.shutdown()


@pytest.mark.parametrize(
    ("external_status", "job_status"),
    (
        (ExternalAnalysisStatus.PARTIAL, AnalysisJobStatus.PARTIAL),
        (ExternalAnalysisStatus.TIMED_OUT, AnalysisJobStatus.TIMED_OUT),
        (ExternalAnalysisStatus.FAILED, AnalysisJobStatus.FAILED),
    ),
)
def test_external_terminal_status_is_preserved(
    tmp_path: Path,
    external_status: ExternalAnalysisStatus,
    job_status: AnalysisJobStatus,
) -> None:
    runner = Mock()
    runner.run.side_effect = lambda request: _result(
        request.configuration.output_root, external_status
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    job = service.submit(_request(tmp_path / "source"))
    service.wait(job.job_id, timeout=5)
    assert service.get(job.job_id).status is job_status
    assert service.get(job.job_id).progress_stage == job_status.value
    assert service.get_result(job.job_id).status is job_status
    service.shutdown()


def test_no_python_github_partial_job_keeps_safe_terminal_summary(tmp_path: Path) -> None:
    runner = Mock()

    def run(request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        result = _result(request.configuration.output_root, ExternalAnalysisStatus.PARTIAL)
        return ExternalSourceAnalysisResult(
            source_kind=ExternalSourceKind.PUBLIC_GITHUB_REPOSITORY,
            execution_policy=result.execution_policy,
            status=result.status,
            acquisition_status="PARTIAL",
            repository_name="repository",
            github_owner="owner",
            github_repository="repository",
            resolved_commit_sha="a" * 40,
            discovered_module_count=0,
            selected_module_count=0,
            module_results=(),
            output_root=result.output_root,
            report_path=result.report_path,
            duration_seconds=result.duration_seconds,
            cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
            issues=("NO_PYTHON_FILES",),
        )

    runner.run.side_effect = run
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    job = service.submit(
        ExternalSourceAnalysisRequest(
            PublicGitHubRepository("https://github.com/owner/repository"),
            configuration=ExternalAnalysisConfiguration(output_root=tmp_path / "ignored"),
        )
    )
    service.wait(job.job_id, timeout=5)

    snapshot = service.get(job.job_id)
    result = service.get_result(job.job_id)
    assert snapshot.status is AnalysisJobStatus.PARTIAL
    assert snapshot.progress_stage == "PARTIAL"
    assert snapshot.safe_error_category is None
    assert result.status is AnalysisJobStatus.PARTIAL
    assert result.project_line_coverage_percent is None
    assert result.project_branch_coverage_percent is None
    assert result.cleanup_status == "COMPLETED"
    assert result.issues == ("NO_PYTHON_FILES",)
    service.shutdown()


def test_unexpected_worker_failure_publishes_safe_result_and_isolates_next_run(
    tmp_path: Path,
) -> None:
    callback_completed = Event()
    private_detail = str(tmp_path / "private-worker-payload")
    runner = Mock()

    def run(request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        if runner.run.call_count == 1:
            raise RuntimeError(private_detail)
        return _result(request.configuration.output_root)

    runner.run.side_effect = run
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_running_jobs=1,
            maximum_queued_jobs=1,
        ),
        runner_factory=Mock(return_value=runner),
    )

    first = service.submit(_request(tmp_path / "ignored-first"))
    first_future = service._entry(first.job_id).future
    assert first_future is not None
    first_future.add_done_callback(lambda _: callback_completed.set())
    with pytest.raises(RuntimeError, match="private-worker-payload"):
        service.wait(first.job_id, timeout=5)
    assert callback_completed.wait(2)

    first_snapshot = service.get(first.job_id)
    first_result = service.get_result(first.job_id)
    frozen_payload = first_result.to_dict()
    assert first_snapshot.status is AnalysisJobStatus.FAILED
    assert first_snapshot.safe_error_category == "INTERNAL_WORKER_ERROR"
    assert first_result.status is AnalysisJobStatus.FAILED
    assert first_result.issues == ("INTERNAL_WORKER_ERROR",)
    assert first_result.modules == ()
    assert first_result.project_coverage is None
    assert private_detail not in repr(frozen_payload)
    assert service.capacity()[:2] == (0, 0)

    second = service.submit(_request(tmp_path / "ignored-second"))
    service.wait(second.job_id, timeout=5)
    assert second.job_id != first.job_id
    assert service.get(second.job_id).status is AnalysisJobStatus.COMPLETED
    assert service.get_result(first.job_id).to_dict() == frozen_payload
    assert service.capacity()[:2] == (0, 0)
    service.shutdown()


def test_worker_failure_publishes_terminal_result_before_releasing_capacity(
    tmp_path: Path,
) -> None:
    private_detail = str(tmp_path / "private-worker-payload")
    runner = Mock()
    runner.run.side_effect = RuntimeError(private_detail)
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_running_jobs=1,
            maximum_queued_jobs=1,
        ),
        runner_factory=Mock(return_value=runner),
    )
    release_observations: list[tuple[AnalysisJobStatus, bool, str | None]] = []
    real_release = service._capacity.release

    def observe_release() -> None:
        with service._lock:
            entry = next(iter(service._jobs.values()))
            release_observations.append(
                (
                    entry.summary.status,
                    entry.result is not None,
                    entry.summary.safe_error_category,
                )
            )
        real_release()

    service._capacity.release = observe_release
    job = service.submit(_request(tmp_path / "ignored"))

    with pytest.raises(RuntimeError, match="private-worker-payload"):
        service.wait(job.job_id, timeout=5)

    assert release_observations == [
        (AnalysisJobStatus.FAILED, True, "INTERNAL_WORKER_ERROR")
    ]
    assert service.get_result(job.job_id).issues == ("INTERNAL_WORKER_ERROR",)
    service.shutdown()


def test_pinned_authorization_lease_prevents_purge_during_job_admission(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    runner = Mock()
    runner.run.side_effect = lambda request: _github_result(
        request.configuration.output_root
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path, retention_seconds=10),
        runner_factory=Mock(return_value=runner),
        clock=lambda: now[0],
    )
    discovery = service.submit(_github_request(tmp_path / "discovery"))
    service.wait(discovery.job_id, timeout=5)
    admission_started = Event()
    continue_admission = Event()
    real_submit_validated = service._submit_validated
    outcome: dict[str, object] = {}

    def pause_after_authorization(
        request: ExternalSourceAnalysisRequest,
    ) -> object:
        admission_started.set()
        assert continue_admission.wait(5)
        return real_submit_validated(request)

    def submit_dynamic() -> None:
        try:
            outcome["job"] = service.submit_pinned_github_dynamic(
                discovery_job_id=discovery.job_id,
                repository_url=_GITHUB_REPOSITORY_URL,
                configuration=_pinned_configuration(tmp_path / "dynamic"),
                trusted_execution_acknowledged=True,
            )
        except BaseException as error:
            outcome["error"] = error

    service._submit_validated = pause_after_authorization
    submit_thread = Thread(target=submit_dynamic, daemon=True)
    submit_thread.start()
    assert admission_started.wait(5)

    try:
        now[0] += timedelta(seconds=11)
        assert service.purge_expired() == 0
        assert service.get(discovery.job_id).status is AnalysisJobStatus.COMPLETED
    finally:
        continue_admission.set()
        submit_thread.join(5)

    assert not submit_thread.is_alive()
    assert "error" not in outcome
    dynamic = outcome["job"]
    assert hasattr(dynamic, "job_id")
    service.wait(dynamic.job_id, timeout=5)
    assert service.purge_expired() == 1
    with pytest.raises(AnalysisJobNotFoundError):
        service.get(discovery.job_id)
    service.shutdown()


def test_shutdown_marks_cancelled_queued_job_terminal_and_releases_capacity(
    tmp_path: Path,
) -> None:
    started = Event()
    release = Event()
    runner = Mock()

    def run(request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        started.set()
        assert release.wait(5)
        return _result(request.configuration.output_root)

    runner.run.side_effect = run
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_running_jobs=1,
            maximum_queued_jobs=1,
        ),
        runner_factory=Mock(return_value=runner),
    )
    first = service.submit(_request(tmp_path / "ignored-first"))
    assert started.wait(2)
    queued = service.submit(_request(tmp_path / "ignored-queued"))

    try:
        service.shutdown(wait=False)
        queued_snapshot = service.get(queued.job_id)
        assert queued_snapshot.status is AnalysisJobStatus.CANCELLED
        assert queued_snapshot.finished_at is not None
        assert queued_snapshot.progress_stage == "CANCELLED"
        assert queued_snapshot.cancellation_requested is True
        queued_result = service.get_result(queued.job_id)
        assert queued_result.status is AnalysisJobStatus.CANCELLED
        assert queued_result.issues == ("CANCELLED",)
        assert service.capacity()[:2] == (1, 0)
    finally:
        release.set()

    service.wait(first.job_id, timeout=5)
    assert service.get(first.job_id).status is AnalysisJobStatus.COMPLETED
    assert service.capacity()[:2] == (0, 0)
    assert runner.run.call_count == 1


def test_retention_cleanup_failure_keeps_registry_for_safe_retry(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    runner = Mock()
    runner.run.side_effect = lambda request: _result(
        request.configuration.output_root
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            retention_seconds=10,
        ),
        runner_factory=Mock(return_value=runner),
        clock=lambda: now[0],
    )
    job = service.submit(_request(tmp_path / "ignored"))
    service.wait(job.job_id, timeout=5)
    job_output = tmp_path / job.job_id
    assert job_output.is_dir()
    now[0] += timedelta(seconds=11)
    real_remove = service._remove_job_output
    cleanup_attempts = 0

    def fail_once(job_id: str) -> None:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise OSError("private retention cleanup detail")
        real_remove(job_id)

    service._remove_job_output = fail_once

    with pytest.raises(OSError, match="private retention cleanup detail"):
        service.purge_expired()

    assert service.get(job.job_id).status is AnalysisJobStatus.COMPLETED
    assert job_output.is_dir()
    assert service.purge_expired() == 1
    with pytest.raises(AnalysisJobNotFoundError):
        service.get(job.job_id)
    assert not job_output.exists()
    assert cleanup_attempts == 2
    service.shutdown()
