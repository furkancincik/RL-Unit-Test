from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from models.analysis_job_result import AnalysisJobStatus
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalSourceKind,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
)
from services.analysis_job_service import (
    AnalysisJobQueueFullError,
    AnalysisJobService,
    AnalysisJobSettings,
    AnalysisJobStateConflictError,
)


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
    assert service.get_result(job.job_id).status is job_status
    service.shutdown()
