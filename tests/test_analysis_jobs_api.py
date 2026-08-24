import json
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
)
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def _runner_result(request) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root
    output.mkdir(parents=True, exist_ok=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    return ExternalSourceAnalysisResult(
        source_kind=request.source.source_kind,
        execution_policy=request.execution_policy,
        status=ExternalAnalysisStatus.STATIC_COMPLETED,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(),
        output_root=output,
        report_path=report,
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


def _client(tmp_path: Path) -> tuple[TestClient, AnalysisJobService, Mock]:
    runner = Mock()
    runner.run.side_effect = _runner_result
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    return TestClient(create_app(job_service=service)), service, runner


def test_create_app_is_import_safe_health_and_openapi(tmp_path: Path) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        health = client.get("/api/v1/health")
        schema = client.get("/openapi.json")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert schema.status_code == 200
    assert "/api/v1/jobs/inline" in schema.json()["paths"]
    runner.run.assert_not_called()
    service.shutdown()


def test_inline_submission_defaults_to_static_and_result_is_polled(tmp_path: Path) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        response = client.post("/api/v1/jobs/inline", json={"source_code": "def target():\n    return 1\n"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        service.wait(job_id, timeout=5)
        result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["analysis_policy"] == "STATIC_DISCOVERY_ONLY"
    assert result.json()["project_line_coverage_percent"] is None
    request = runner.run.call_args.args[0]
    assert request.execution_policy.value == "STATIC_DISCOVERY_ONLY"
    service.shutdown()


@pytest.mark.parametrize(
    "source_code, analysis",
    [
        ("", {}),
        (" \t\r\n", {}),
        (
            " \n\t ",
            {
                "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                "trusted_execution_acknowledged": True,
                "function_pipeline_timeout_seconds": 30,
            },
        ),
    ],
)
def test_blank_inline_source_is_rejected_before_job_creation(
    tmp_path: Path,
    source_code: str,
    analysis: dict[str, object],
) -> None:
    client, service, runner = _client(tmp_path)
    capacity_before = service.capacity()

    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": source_code, "analysis": analysis},
        )

    assert response.status_code == 422
    assert service.capacity() == capacity_before
    runner.run.assert_not_called()
    for detail in response.json()["detail"]:
        if "source_code" in detail.get("loc", []):
            assert "input" not in detail
    service.shutdown()


def test_inline_validation_preserves_nonblank_source_content(tmp_path: Path) -> None:
    client, service, runner = _client(tmp_path)
    source = "\n# leading content\ndef target():\n    return 1\n\n"

    with client:
        response = client.post("/api/v1/jobs/inline", json={"source_code": source})
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    request = runner.run.call_args.args[0]
    assert request.source.source_text == source
    service.shutdown()


def test_trusted_dynamic_requires_acknowledgement_and_timeout(tmp_path: Path) -> None:
    client, service, _ = _client(tmp_path)
    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": "def target():\n    return 1\n", "analysis": {"policy": "TRUSTED_DYNAMIC_ANALYSIS"}},
        )
    assert response.status_code == 422
    assert "sandbox" in json.dumps(client.app.openapi()).lower()
    service.shutdown()


def test_upload_and_github_use_separate_payloads(tmp_path: Path) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        upload = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("target.py", b"def target():\n    return 1\n", "text/x-python")},
            data={"analysis": "{}"},
        )
        github = client.post(
            "/api/v1/jobs/github",
            json={"repository_url": "https://github.com/owner/repository"},
        )
    assert upload.status_code == 202
    assert github.status_code == 202
    assert "/api/v1/jobs/local" not in client.app.openapi()["paths"]
    assert client.post("/api/v1/jobs/local", json={"path": "C:/server"}).status_code in {404, 405}
    assert upload.json()["source_kind"] == "UPLOADED_PYTHON_FILE"
    assert github.json()["source_kind"] == "PUBLIC_GITHUB_REPOSITORY"
    service.wait(upload.json()["job_id"], timeout=5)
    service.wait(github.json()["job_id"], timeout=5)
    service.shutdown()


def test_upload_validation_and_pending_unknown_contracts(tmp_path: Path) -> None:
    client, service, _ = _client(tmp_path)
    with client:
        wrong = client.post("/api/v1/jobs/upload", files={"file": ("target.txt", b"x=1", "text/plain")}, data={"analysis": "{}"})
        unknown = client.get("/api/v1/jobs/not-a-job")
    assert wrong.status_code == 422
    assert unknown.status_code == 404
    service.shutdown()


def test_inline_and_upload_byte_limits(tmp_path: Path) -> None:
    runner = Mock()
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_inline_source_bytes=4,
            maximum_upload_bytes=4,
        ),
        runner_factory=Mock(return_value=runner),
    )
    with TestClient(create_app(job_service=service)) as client:
        inline = client.post("/api/v1/jobs/inline", json={"source_code": "value = 1"})
        upload = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("target.py", b"value = 1", "text/x-python")},
            data={"analysis": "{}"},
        )
    assert inline.status_code == 413
    assert upload.status_code == 413
    runner.run.assert_not_called()
    service.shutdown()


def test_api_queue_pending_and_cancellation_contract(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    runner = Mock()

    def blocking(request):
        started.set()
        release.wait(5)
        return _runner_result(request)

    runner.run.side_effect = blocking
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_running_jobs=1,
            maximum_queued_jobs=1,
        ),
        runner_factory=Mock(return_value=runner),
    )
    with TestClient(create_app(job_service=service)) as client:
        first = client.post("/api/v1/jobs/inline", json={"source_code": "x=1"})
        assert started.wait(2)
        second = client.post("/api/v1/jobs/inline", json={"source_code": "x=2"})
        full = client.post("/api/v1/jobs/inline", json={"source_code": "x=3"})
        pending = client.get(f"/api/v1/jobs/{first.json()['job_id']}/result")
        running_cancel = client.post(f"/api/v1/jobs/{first.json()['job_id']}/cancel")
        queued_cancel = client.post(f"/api/v1/jobs/{second.json()['job_id']}/cancel")
        assert full.status_code == 429
        assert pending.status_code == 409
        assert running_cancel.status_code == 409
        assert queued_cancel.status_code == 200
        assert queued_cancel.json()["status"] == "CANCELLED"
        release.set()
        service.wait(first.json()["job_id"], timeout=5)
        completed_cancel = client.post(f"/api/v1/jobs/{first.json()['job_id']}/cancel")
        assert completed_cancel.status_code == 409
    service.shutdown()


def test_unexpected_submission_errors_are_not_sanitized_as_domain_errors(tmp_path: Path) -> None:
    service = Mock(spec=AnalysisJobService)
    service.settings = AnalysisJobSettings(output_root=tmp_path)
    service.submit.side_effect = ValueError("internal bug")
    with TestClient(create_app(job_service=service), raise_server_exceptions=True) as client:
        with pytest.raises(ValueError, match="internal bug"):
            client.post("/api/v1/jobs/inline", json={"source_code": "x=1"})
