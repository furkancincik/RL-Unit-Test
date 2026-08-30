import asyncio
import codecs
import io
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile

from api.app import create_app
from api.routes.analysis_jobs import submit_upload
from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleStatus,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
)
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)
from models.project_analysis_result import (
    FunctionAnalysisResult,
    FunctionRunStatus,
    FunctionSelectionMode,
    FunctionTarget,
    ProjectAnalysisResult,
    ProjectRunStatus,
    TargetSelectionMode,
)
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings
from services.external_source_analysis_service import (
    portable_upload_module_identity,
)


def _runner_result(request) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root
    output.mkdir(parents=True, exist_ok=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    return ExternalSourceAnalysisResult(
        source_kind=request.source.source_kind,
        execution_policy=request.execution_policy,
        status=ExternalAnalysisStatus.STATIC_COMPLETED,
        acquisition_status="COMPLETED",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=(
            "a" * 40
            if request.source.source_kind.value == "PUBLIC_GITHUB_REPOSITORY"
            else None
        ),
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(),
        output_root=output,
        report_path=report,
        duration_seconds=0.1,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


def _github_static_inventory_result(
    request,
    *,
    status: ExternalAnalysisStatus = ExternalAnalysisStatus.STATIC_COMPLETED,
    resolved_sha: str = "b" * 40,
) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root
    output.mkdir(parents=True, exist_ok=True)
    report = output / "external_source_analysis_report.json"
    report.write_text("{}", encoding="utf-8")
    module = ExternalModuleAnalysisResult(
        relative_path="package/worker.py",
        module_name="package.worker",
        status=ExternalModuleStatus.STATIC_ONLY,
        discovered_function_count=2,
        project_result=None,
        issue_category=None,
        issue_message=None,
        artifact_paths=(),
        discovered_function_names=("target", "Helper.run"),
    )
    return ExternalSourceAnalysisResult(
        source_kind=request.source.source_kind,
        execution_policy=request.execution_policy,
        status=status,
        acquisition_status="COMPLETED",
        repository_name="repository",
        github_owner="owner",
        github_repository="repository",
        resolved_commit_sha=resolved_sha,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
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


def _completed_rejection_result(request) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root
    module_output = output / "module"
    function_output = module_output / "target"
    function_output.mkdir(parents=True, exist_ok=True)
    source_file = module_output / "target.py"
    source_file.write_text("def target():\n    return 1\n", encoding="utf-8")
    project_report = module_output / "project_analysis_report.json"
    project_report.write_text("{}", encoding="utf-8")
    external_report = output / "external_source_analysis_report.json"
    external_report.write_text("{}", encoding="utf-8")
    target = FunctionTarget(
        name="target",
        qualified_name="target",
        start_line=1,
        end_line=2,
        parameters=(),
        return_annotation=None,
        is_async=False,
        is_nested=False,
        is_method=False,
        is_supported=True,
    )
    diagnostic = PipelineDiagnosticResult(
        status=PipelineRunStatus.COMPLETED,
        source_file=source_file,
        function_name="target",
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=1.0,
        funnel=PipelineFunnelSnapshot(
            bounded_path_count=19,
            input_generation_accepted_count=1,
            input_generation_rejected_count=18,
            pre_concrete_scenario_count=1,
            concrete_validation_accepted_count=1,
            concrete_validation_rejected_count=0,
            final_scenario_count=1,
            rl_executed_test_count=1,
            q_table_state_count=1,
        ),
        scenario_rejection_counts=(
            ("UNSUPPORTED_INPUT_SYNTHESIS", 18),
        ),
        line_coverage_percent=18.75,
        branch_coverage_percent=12.5,
    )
    function_result = FunctionAnalysisResult(
        target=target,
        status=FunctionRunStatus.COMPLETED,
        diagnostic=diagnostic,
        output_directory=function_output,
    )
    project = ProjectAnalysisResult(
        source_file=source_file,
        module_path="target",
        selection_mode=FunctionSelectionMode.SINGLE,
        discovered_targets=(target,),
        function_results=(function_result,),
        total_duration_seconds=1.0,
        status=ProjectRunStatus.COMPLETED,
        output_root=module_output,
        report_path=project_report,
    )
    module = ExternalModuleAnalysisResult(
        relative_path="target.py",
        module_name="target",
        status=ExternalModuleStatus.COMPLETED,
        discovered_function_count=1,
        project_result=project,
        issue_category=None,
        issue_message=None,
        artifact_paths=(project_report,),
        discovered_function_names=("target",),
    )
    return ExternalSourceAnalysisResult(
        source_kind=request.source.source_kind,
        execution_policy=ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
        status=ExternalAnalysisStatus.COMPLETED,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(module,),
        output_root=output,
        report_path=external_report,
        duration_seconds=1.0,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
    )


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


def test_completed_job_result_exposes_input_rejections_without_status_promotion(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    runner.run.side_effect = _completed_rejection_result

    with client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def target():\n    return 1\n",
                "analysis": {
                    "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                    "trusted_execution_acknowledged": True,
                    "function_pipeline_timeout_seconds": 30,
                },
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=5)
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    function = payload["modules"][0]["functions"][0]
    assert function["status"] == "COMPLETED"
    assert function["bounded_path_count"] == 19
    assert function["input_generation_accepted_count"] == 1
    assert function["input_generation_rejected_count"] == 18
    assert function["input_rejection_categories"] == [
        {"category": "UNSUPPORTED_INPUT_SYNTHESIS", "count": 18}
    ]
    assert function["scenario_count"] == 1
    assert function["concrete_accepted_count"] == 1
    assert function["concrete_rejected_count"] == 0
    assert function["line_coverage_percent"] == 18.75
    assert function["branch_coverage_percent"] == 12.5
    serialized = json.dumps(payload)
    for forbidden in (
        "rejection reason",
        "constructor_arguments",
        "keyword_arguments",
        "expected_result",
        "actual_result",
    ):
        assert forbidden not in serialized
    service.shutdown()


@pytest.mark.parametrize(
    "value",
    (True, False, 0, -1, "NaN", "Infinity", 14_401),
)
def test_invalid_project_timeout_is_422_without_consuming_job_capacity(
    tmp_path: Path,
    value: object,
) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def target() -> int:\n    return 1\n",
                "analysis": {"project_timeout_seconds": value},
            },
        )

    assert response.status_code == 422
    assert service.capacity()[:2] == (0, 0)
    runner.run.assert_not_called()
    service.shutdown()


def test_api_forwards_project_timeout_as_distinct_configuration_value(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def target() -> int:\n    return 1\n",
                "analysis": {
                    "project_timeout_seconds": 45.5,
                    "function_pipeline_timeout_seconds": 12.0,
                },
            },
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    configuration = runner.run.call_args.args[0].configuration
    assert configuration.project_timeout_seconds == pytest.approx(45.5)
    assert configuration.per_function_pipeline_timeout_seconds == pytest.approx(12.0)
    service.shutdown()


def test_project_deadline_terminal_result_is_safe_http_200_not_internal_error(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)

    def deadline_result(request) -> ExternalSourceAnalysisResult:
        output = request.configuration.output_root
        output.mkdir(parents=True, exist_ok=True)
        report = output / "external_source_analysis_report.json"
        report.write_text("{}", encoding="utf-8")
        return ExternalSourceAnalysisResult(
            source_kind=request.source.source_kind,
            execution_policy=request.execution_policy,
            status=ExternalAnalysisStatus.TIMED_OUT,
            acquisition_status="NOT_STARTED",
            repository_name=None,
            github_owner=None,
            github_repository=None,
            resolved_commit_sha=None,
            discovered_module_count=0,
            selected_module_count=0,
            module_results=(),
            output_root=output,
            report_path=report,
            duration_seconds=0.25,
            cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
            issues=("PROJECT_DEADLINE_EXCEEDED",),
            project_timeout_seconds=0.25,
            project_deadline_exceeded=True,
            last_completed_stage="REQUEST_VALIDATION",
            deadline_stage="SOURCE_ACQUISITION",
        )

    runner.run.side_effect = deadline_result
    with client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def target() -> int:\n    return 1\n",
                "analysis": {"project_timeout_seconds": 0.25},
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=5)
        snapshot = client.get(f"/api/v1/jobs/{job_id}")
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "TIMED_OUT"
    assert snapshot.json()["progress_stage"] == "SOURCE_ACQUISITION"
    assert snapshot.json()["safe_error_category"] is None
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "TIMED_OUT"
    assert payload["project_deadline_exceeded"] is True
    assert payload["deadline_stage"] == "SOURCE_ACQUISITION"
    assert payload["project_line_coverage_percent"] is None
    assert "INTERNAL_WORKER_ERROR" not in json.dumps(payload)
    service.shutdown()


def test_inline_explicit_qualified_targets_are_bound_to_public_module_identity(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "class Vessel:\n    def inspect(self):\n        return 1\n",
                "analysis": {
                    "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                    "explicit_target_names": ["Vessel.inspect", "Vessel.inspect"],
                },
            },
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    selection = runner.run.call_args.args[0].configuration.target_selection
    assert selection.mode is TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS
    assert [item.to_dict() for item in selection.selectors] == [
        {
            "module_identity": "inline_source",
            "qualified_name": "Vessel.inspect",
        }
    ]
    service.shutdown()


@pytest.mark.parametrize(
    "qualified_name",
    ("", ".run", "Owner.", "Owner..run", "Owner.run.more", "Owner/run", "Owner.run()"),
)
def test_malformed_explicit_target_is_422_without_consuming_queue(
    tmp_path: Path,
    qualified_name: str,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()
    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def run():\n    return 1\n",
                "analysis": {
                    "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                    "explicit_target_names": [qualified_name],
                },
            },
        )
    assert response.status_code == 422
    assert service.capacity() == before
    runner.run.assert_not_called()
    service.shutdown()


def test_upload_and_github_explicit_target_schemas_are_unambiguous(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    upload_options = json.dumps(
        {
            "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
            "explicit_target_names": ["Handler.run"],
        }
    )
    with client:
        upload = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("worker.py", b"class Handler:\n    def run(self):\n        return 1\n", "text/x-python")},
            data={"analysis": upload_options},
        )
        github = client.post(
            "/api/v1/jobs/github",
            json={
                "repository_url": "https://github.com/owner/repository",
                "analysis": {
                    "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                    "explicit_module_targets": [
                        {
                            "module_identity": "package.worker",
                            "qualified_name": "Handler.run",
                        }
                    ],
                },
            },
        )
        assert upload.status_code == 202
        assert github.status_code == 202
        service.wait(upload.json()["job_id"], timeout=5)
        service.wait(github.json()["job_id"], timeout=5)

    requests = [call.args[0] for call in runner.run.call_args_list]
    assert requests[0].configuration.target_selection.selectors[0].module_identity == (
        "upload_worker"
    )
    assert requests[1].configuration.target_selection.selectors[0].to_dict() == {
        "module_identity": "package.worker",
        "qualified_name": "Handler.run",
    }
    service.shutdown()


def test_github_ref_is_bound_before_queue_and_resolved_sha_is_public(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    commit = "b" * 40

    with client:
        response = client.post(
            "/api/v1/jobs/github",
            json={
                "repository_url": "https://github.com/owner/repository.git",
                "ref": commit,
            },
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)
        result = client.get(
            f"/api/v1/jobs/{response.json()['job_id']}/result"
        )

    request = runner.run.call_args.args[0]
    assert request.source.repository_url == "https://github.com/owner/repository"
    assert request.source.ref == commit
    assert result.status_code == 200
    assert result.json()["resolved_commit_sha"] == "a" * 40
    service.shutdown()


def test_trusted_dynamic_github_is_rejected_before_job_creation(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()

    with client:
        response = client.post(
            "/api/v1/jobs/github",
            json={
                "repository_url": "https://github.com/owner/repository",
                "analysis": {
                    "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                    "trusted_execution_acknowledged": True,
                    "pytest_coverage_timeout_seconds": 30,
                    "function_pipeline_timeout_seconds": 120,
                    "project_timeout_seconds": 300,
                },
            },
        )

    assert response.status_code == 422
    assert "static discovery" in response.text.lower()
    assert "owner/repository" not in response.text
    assert "120" not in response.text
    assert service.capacity() == before
    runner.run.assert_not_called()
    service.shutdown()


def test_pinned_github_dynamic_requires_successful_discovery_job(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()

    with client:
        response = client.post(
            "/api/v1/jobs/github",
            json={
                "repository_url": "https://github.com/owner/repository",
                "discovery_job_id": "0" * 32,
                "analysis": {
                    "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                    "trusted_execution_acknowledged": True,
                    "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                    "explicit_module_targets": [
                        {
                            "module_identity": "package.worker",
                            "qualified_name": "target",
                        }
                    ],
                },
            },
        )

    assert response.status_code == 422
    assert service.capacity() == before
    runner.run.assert_not_called()
    service.shutdown()


def test_successful_github_discovery_authorizes_exact_pinned_dynamic_request(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    runner.run.side_effect = lambda request: (
        _github_static_inventory_result(request)
        if request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY
        else _runner_result(request)
    )

    with client:
        discovery = client.post(
            "/api/v1/jobs/github",
            json={
                "repository_url": "https://github.com/owner/repository",
                "ref": "main",
            },
        )
        assert discovery.status_code == 202
        service.wait(discovery.json()["job_id"], timeout=5)
        dynamic_payload = {
            "repository_url": "https://github.com/owner/repository",
            "discovery_job_id": discovery.json()["job_id"],
            "analysis": {
                "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                "trusted_execution_acknowledged": True,
                "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                "explicit_module_targets": [
                    {
                        "module_identity": "package.worker",
                        "qualified_name": "target",
                    }
                ],
                "greedy_minimization": True,
            },
        }
        dynamic = client.post(
            "/api/v1/jobs/github",
            json=dynamic_payload,
        )
        assert dynamic.status_code == 202
        service.wait(dynamic.json()["job_id"], timeout=5)
        second_dynamic = client.post(
            "/api/v1/jobs/github",
            json=dynamic_payload,
        )
        assert second_dynamic.status_code == 202
        service.wait(second_dynamic.json()["job_id"], timeout=5)

    first_dynamic_request = runner.run.call_args_list[-2].args[0]
    dynamic_request = runner.run.call_args_list[-1].args[0]
    assert dynamic_request.execution_policy is ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS
    assert dynamic_request.source.repository_url == "https://github.com/owner/repository"
    assert dynamic_request.source.ref == "b" * 40
    assert dynamic_request.configuration.target_selection.selectors[0].to_dict() == {
        "module_identity": "package.worker",
        "qualified_name": "target",
    }
    assert first_dynamic_request.source == dynamic_request.source
    assert (
        first_dynamic_request.configuration.target_selection
        == dynamic_request.configuration.target_selection
    )
    assert (
        first_dynamic_request.configuration.output_root
        != dynamic_request.configuration.output_root
    )
    service.shutdown()


@pytest.mark.parametrize(
    ("repository_url", "ref", "qualified_name"),
    (
        ("https://github.com/other/repository", None, "target"),
        ("https://github.com/owner/repository", "c" * 40, "target"),
        ("https://github.com/owner/repository", None, "unknown"),
    ),
)
def test_pinned_github_dynamic_rejects_repository_sha_and_target_tampering(
    tmp_path: Path,
    repository_url: str,
    ref: str | None,
    qualified_name: str,
) -> None:
    client, service, runner = _client(tmp_path)
    runner.run.side_effect = _github_static_inventory_result

    with client:
        discovery = client.post(
            "/api/v1/jobs/github",
            json={"repository_url": "https://github.com/owner/repository"},
        )
        assert discovery.status_code == 202
        service.wait(discovery.json()["job_id"], timeout=5)
        before = service.capacity()
        payload: dict[str, object] = {
            "repository_url": repository_url,
            "discovery_job_id": discovery.json()["job_id"],
            "analysis": {
                "policy": "TRUSTED_DYNAMIC_ANALYSIS",
                "trusted_execution_acknowledged": True,
                "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                "explicit_module_targets": [
                    {
                        "module_identity": "package.worker",
                        "qualified_name": qualified_name,
                    }
                ],
            },
        }
        if ref is not None:
            payload["ref"] = ref
        response = client.post("/api/v1/jobs/github", json=payload)

    assert response.status_code == 422
    assert service.capacity() == before
    assert runner.run.call_count == 1
    service.shutdown()


@pytest.mark.parametrize(
    ("repository_url", "ref"),
    (
        ("https://user@github.com/owner/repository", None),
        ("https://github.com/owner/repository?token=secret", None),
        ("https://github.com/owner/repository", "../main"),
        ("https://github.com/owner/repository", "main@{1}"),
    ),
)
def test_github_url_and_ref_are_rejected_before_job_creation(
    tmp_path: Path,
    repository_url: str,
    ref: str | None,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()
    payload: dict[str, object] = {"repository_url": repository_url}
    if ref is not None:
        payload["ref"] = ref

    with client:
        response = client.post("/api/v1/jobs/github", json=payload)

    assert response.status_code == 422
    assert service.capacity() == before
    runner.run.assert_not_called()
    assert "secret" not in response.text
    service.shutdown()


@pytest.mark.parametrize(
    ("filename", "expected_identity"),
    (
        ("my-file.py", "upload_my_file"),
        ("my file.py", "upload_my_file"),
        ("my.module.py", "upload_my_module"),
        ("module.py", "upload_module"),
    ),
)
def test_upload_explicit_target_uses_external_service_canonical_module_identity(
    tmp_path: Path,
    filename: str,
    expected_identity: str,
) -> None:
    client, service, runner = _client(tmp_path)
    analysis = json.dumps(
        {
            "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
            "explicit_target_names": ["target"],
        }
    )

    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files={
                "file": (
                    filename,
                    b"def target():\n    return 1\n",
                    "text/x-python",
                )
            },
            data={"analysis": analysis},
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    request = runner.run.call_args.args[0]
    selector = request.configuration.target_selection.selectors[0]
    assert selector.module_identity == expected_identity
    assert selector.module_identity == portable_upload_module_identity(request.source)
    assert selector.qualified_name == "target"
    service.shutdown()


def test_hyphenated_upload_default_selection_remains_backward_compatible(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files={
                "file": (
                    "my-file.py",
                    b"def target():\n    return 1\n",
                    "text/x-python",
                )
            },
            data={"analysis": "{}"},
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    selection = runner.run.call_args.args[0].configuration.target_selection
    assert selection.mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    assert selection.selectors == ()
    service.shutdown()


def test_malformed_upload_target_is_422_without_consuming_queue(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()
    analysis = json.dumps(
        {
            "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
            "explicit_target_names": ["Owner.run()"],
        }
    )

    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files={
                "file": (
                    "my-file.py",
                    b"def target():\n    return 1\n",
                    "text/x-python",
                )
            },
            data={"analysis": analysis},
        )

    assert response.status_code == 422
    assert service.capacity() == before
    runner.run.assert_not_called()
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


def test_inline_leading_bom_is_normalized_before_job_creation(
    tmp_path: Path,
) -> None:
    client, service, runner = _client(tmp_path)
    source = "\ufeffdef target():\r\n    return 1\r\n"

    with client:
        response = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": source},
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    request = runner.run.call_args.args[0]
    assert request.source.source_text == source.removeprefix("\ufeff")
    service.shutdown()


@pytest.mark.parametrize(
    "content",
    (
        b"",
        b" \t\r\n\f",
        codecs.BOM_UTF8,
        codecs.BOM_UTF8 + b" \t\r\n",
    ),
)
def test_blank_upload_is_rejected_before_job_creation(
    tmp_path: Path,
    content: bytes,
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()

    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("empty.py", content, "text/x-python")},
            data={"analysis": "{}"},
        )

    assert response.status_code == 422
    assert service.capacity() == before
    runner.run.assert_not_called()
    service.shutdown()


@pytest.mark.parametrize(
    "content",
    (
        b"# comment-only module\n",
        b'"""module documentation only"""\n',
        b"answer = 42\n",
    ),
)
def test_nonblank_zero_function_uploads_remain_accepted(
    tmp_path: Path,
    content: bytes,
) -> None:
    client, service, runner = _client(tmp_path)

    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("module.py", content, "text/x-python")},
            data={"analysis": "{}"},
        )
        assert response.status_code == 202
        service.wait(response.json()["job_id"], timeout=5)

    assert runner.run.call_args.args[0].source.file_bytes == content
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


class _TrackedUpload(StarletteUploadFile):
    def __init__(
        self,
        filename: str,
        content: bytes,
        *,
        read_error: BaseException | None = None,
    ) -> None:
        super().__init__(file=io.BytesIO(content), filename=filename)
        self._content = content
        self._read_error = read_error
        self.closed = False

    async def read(self, _: int) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._content

    async def close(self) -> None:
        self.closed = True
        await super().close()


class _TrackedUploadRequest:
    def __init__(
        self,
        *,
        service: AnalysisJobService,
        upload: _TrackedUpload,
        analysis: str,
    ) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(analysis_job_service=service),
        )
        self._form = FormData(
            (("file", upload), ("analysis", analysis))
        )

    async def form(self) -> FormData:
        return self._form


@pytest.mark.parametrize(
    ("filename", "content", "analysis"),
    (
        ("module.py", b"value = 1\n", "{"),
        ("module.txt", b"value = 1\n", "{}"),
        ("module.py", b"", "{}"),
        ("module.py", b" \t\r\n", "{}"),
    ),
)
def test_upload_handle_is_closed_on_every_pre_admission_rejection(
    tmp_path: Path,
    filename: str,
    content: bytes,
    analysis: str,
) -> None:
    runner = Mock()
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    upload = _TrackedUpload(filename, content)
    request = _TrackedUploadRequest(
        service=service,
        upload=upload,
        analysis=analysis,
    )

    with pytest.raises(HTTPException):
        asyncio.run(submit_upload(request, upload, analysis))

    assert upload.closed is True
    assert service.capacity()[:2] == (0, 0)
    assert service._jobs == {}
    runner.run.assert_not_called()
    service.shutdown()


def test_upload_handle_is_closed_when_read_raises_unexpected_error(
    tmp_path: Path,
) -> None:
    private_detail = str(tmp_path / "private-upload-read")
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(),
    )
    upload = _TrackedUpload(
        "module.py",
        b"",
        read_error=RuntimeError(private_detail),
    )
    request = _TrackedUploadRequest(
        service=service,
        upload=upload,
        analysis="{}",
    )

    with pytest.raises(RuntimeError, match="private-upload-read"):
        asyncio.run(submit_upload(request, upload, "{}"))

    assert upload.closed is True
    assert service.capacity()[:2] == (0, 0)
    assert service._jobs == {}
    service.shutdown()


def test_upload_handle_is_closed_after_size_limit_rejection(
    tmp_path: Path,
) -> None:
    runner = Mock()
    service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path,
            maximum_upload_bytes=4,
        ),
        runner_factory=Mock(return_value=runner),
    )
    upload = _TrackedUpload("module.py", b"value")
    request = _TrackedUploadRequest(
        service=service,
        upload=upload,
        analysis="{}",
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(submit_upload(request, upload, "{}"))

    assert captured.value.status_code == 413
    assert upload.closed is True
    assert service.capacity()[:2] == (0, 0)
    assert service._jobs == {}
    runner.run.assert_not_called()
    service.shutdown()


def test_invalid_multipart_shape_closes_every_upload_before_admission(
    tmp_path: Path,
) -> None:
    runner = Mock()
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    selected = _TrackedUpload("selected.py", b"value = 1\n")
    extra = _TrackedUpload("extra.py", b"value = 2\n")
    request = _TrackedUploadRequest(
        service=service,
        upload=selected,
        analysis="{}",
    )
    request._form = FormData(
        (
            ("file", selected),
            ("attachment", extra),
            ("analysis", "{}"),
        )
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(submit_upload(request, selected, "{}"))

    assert captured.value.status_code == 422
    assert selected.closed is True
    assert extra.closed is True
    assert service.capacity()[:2] == (0, 0)
    assert service._jobs == {}
    runner.run.assert_not_called()
    service.shutdown()


@pytest.mark.parametrize(
    "files",
    (
        [
            ("file", ("first.py", b"value = 1\n", "text/x-python")),
            ("file", ("second.py", b"value = 2\n", "text/x-python")),
        ],
        [
            ("file", ("module.py", b"value = 1\n", "text/x-python")),
            ("attachment", ("extra.py", b"value = 2\n", "text/x-python")),
        ],
    ),
)
def test_upload_rejects_duplicate_or_unexpected_file_parts_before_admission(
    tmp_path: Path,
    files: list[tuple[str, tuple[str, bytes, str]]],
) -> None:
    client, service, runner = _client(tmp_path)
    before = service.capacity()

    with client:
        response = client.post(
            "/api/v1/jobs/upload",
            files=files,
            data={"analysis": "{}"},
        )

    assert response.status_code == 422
    assert service.capacity() == before
    assert service._jobs == {}
    runner.run.assert_not_called()
    service.shutdown()


def test_unexpected_worker_failure_has_safe_http_200_terminal_result(
    tmp_path: Path,
) -> None:
    callback_completed = Event()
    private_detail = str(tmp_path / "private-worker-trace")
    runner = Mock()
    runner.run.side_effect = RuntimeError(private_detail)
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )

    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": "def target():\n    return 1\n"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        future = service._entry(job_id).future
        assert future is not None
        future.add_done_callback(lambda _: callback_completed.set())
        with pytest.raises(RuntimeError, match="private-worker-trace"):
            service.wait(job_id, timeout=5)
        assert callback_completed.wait(2)
        snapshot = client.get(f"/api/v1/jobs/{job_id}")
        result = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "FAILED"
    assert snapshot.json()["safe_error_category"] == "INTERNAL_WORKER_ERROR"
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "FAILED"
    assert payload["issues"] == ["INTERNAL_WORKER_ERROR"]
    assert private_detail not in json.dumps(payload)
    assert service.capacity()[:2] == (0, 0)
    service.shutdown()
