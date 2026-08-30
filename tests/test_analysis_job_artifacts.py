from pathlib import Path
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleStatus,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
)
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def _request(root: Path) -> ExternalSourceAnalysisRequest:
    return ExternalSourceAnalysisRequest(
        InlinePythonSource("def target():\n    return 1\n"),
        ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        ExternalAnalysisConfiguration(output_root=root),
    )


def _artifact_result(request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root
    output.mkdir(parents=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    generated = output / "test_generated_target.py"
    generated.write_text("def test_target():\n    assert True\n", encoding="utf-8")
    secret = output / "secret.txt"
    secret.write_text("never expose", encoding="utf-8")
    module = ExternalModuleAnalysisResult(
        relative_path="inline_source.py",
        module_name="inline_source",
        status=ExternalModuleStatus.STATIC_ONLY,
        discovered_function_count=1,
        project_result=None,
        issue_category=None,
        issue_message=None,
        artifact_paths=(generated, secret),
    )
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
        module_results=(module,),
        output_root=output,
        report_path=report,
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


def test_artifact_listing_and_download_use_opaque_ids(tmp_path: Path) -> None:
    runner = Mock()
    runner.run.side_effect = _artifact_result
    service = AnalysisJobService(settings=AnalysisJobSettings(output_root=tmp_path), runner_factory=Mock(return_value=runner))
    job = service.submit(_request(tmp_path))
    service.wait(job.job_id, timeout=5)

    with TestClient(create_app(job_service=service)) as client:
        listing = client.get(f"/api/v1/jobs/{job.job_id}/artifacts")
        assert listing.status_code == 200
        artifacts = listing.json()["artifacts"]
        assert {item["filename"] for item in artifacts} == {
            "external_source_analysis_report.json", "test_generated_target.py"
        }
        for item in artifacts:
            response = client.get(
                f"/api/v1/jobs/{job.job_id}/artifacts/{item['artifact_id']}"
            )
            assert response.status_code == 200
            assert "attachment" in response.headers["content-disposition"]
        assert "secret.txt" not in repr(artifacts)
    service.shutdown()


@pytest.mark.parametrize("artifact_id", ("../secret", "%2e%2e%2fsecret", "C:%5Csecret", "unknown"))
def test_artifact_traversal_and_unknown_ids_are_rejected(tmp_path: Path, artifact_id: str) -> None:
    runner = Mock()
    runner.run.side_effect = _artifact_result
    service = AnalysisJobService(settings=AnalysisJobSettings(output_root=tmp_path), runner_factory=Mock(return_value=runner))
    job = service.submit(_request(tmp_path))
    service.wait(job.job_id, timeout=5)
    with TestClient(create_app(job_service=service)) as client:
        response = client.get(f"/api/v1/jobs/{job.job_id}/artifacts/{artifact_id}")
    assert response.status_code in {404, 405}
    service.shutdown()


def test_artifact_id_cannot_cross_job_boundary(tmp_path: Path) -> None:
    runner = Mock()
    runner.run.side_effect = _artifact_result
    service = AnalysisJobService(settings=AnalysisJobSettings(output_root=tmp_path), runner_factory=Mock(return_value=runner))
    first = service.submit(_request(tmp_path))
    second = service.submit(_request(tmp_path))
    service.wait(first.job_id, timeout=5)
    service.wait(second.job_id, timeout=5)
    artifact_id = service.list_artifacts(first.job_id)[0].artifact_id
    with TestClient(create_app(job_service=service)) as client:
        response = client.get(f"/api/v1/jobs/{second.job_id}/artifacts/{artifact_id}")
    assert response.status_code == 404
    service.shutdown()


def test_active_artifact_download_lease_prevents_retention_purge(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    runner = Mock()
    runner.run.side_effect = _artifact_result
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            retention_seconds=10,
        ),
        runner_factory=Mock(return_value=runner),
        clock=lambda: now[0],
    )
    job = service.submit(_request(tmp_path))
    service.wait(job.job_id, timeout=5)
    artifact = service.list_artifacts(job.job_id)[0]
    metadata, path = service.acquire_artifact(job.job_id, artifact.artifact_id)
    now[0] += timedelta(seconds=11)

    assert metadata.artifact_id == artifact.artifact_id
    assert path.is_file()
    assert service.purge_expired() == 0
    assert path.is_file()

    service.release_artifact(job.job_id)
    assert service.purge_expired() == 1
    assert not path.exists()
    service.shutdown()


def test_artifact_response_releases_download_lease_after_send(
    tmp_path: Path,
) -> None:
    runner = Mock()
    runner.run.side_effect = _artifact_result
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    job = service.submit(_request(tmp_path))
    service.wait(job.job_id, timeout=5)
    artifact = service.list_artifacts(job.job_id)[0]

    with TestClient(create_app(job_service=service)) as client:
        response = client.get(
            f"/api/v1/jobs/{job.job_id}/artifacts/{artifact.artifact_id}"
        )

    assert response.status_code == 200
    assert service._entry(job.job_id).artifact_leases == 0
    service.shutdown()
