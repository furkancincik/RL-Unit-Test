from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.app import create_app
from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
)
from models.project_coverage_result import (
    ProjectCoverageResult,
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
)
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def _result(request) -> ExternalSourceAnalysisResult:
    output = request.configuration.output_root.resolve()
    combined = output / "project_combined"
    combined.mkdir(parents=True)
    full = combined / "test_project_full_pool.py"
    minimized = combined / "test_project_greedy_minimized.py"
    report = combined / "project_coverage_report.json"
    full.write_text("def test_full():\n    assert True\n", encoding="utf-8")
    minimized.write_text("def test_minimized():\n    assert True\n", encoding="utf-8")
    project_coverage = ProjectCoverageResult(
        status=ProjectCoverageStatus.PARTIAL,
        scope=ProjectCoverageScopeSummary(
            discovered_module_count=1,
            selected_module_count=1,
            completed_module_count=1,
            discovered_function_count=1,
            eligible_function_count=1,
            analyzed_function_count=1,
            completed_function_count=1,
            partial_function_count=0,
            failed_function_count=0,
            timed_out_function_count=0,
            unsupported_function_count=0,
            skipped_limit_function_count=0,
            scope_complete=False,
        ),
        executable_line_identities=(),
        executable_branch_identities=(),
        target_line_identities=(),
        target_branch_identities=(),
        minimized_covered_line_identities=(),
        minimized_covered_branch_identities=(),
        full_line_coverage_percent=100.0,
        full_branch_coverage_percent=100.0,
        minimized_line_coverage_percent=100.0,
        minimized_branch_coverage_percent=100.0,
        full_pytest_exit_code=0,
        minimized_pytest_exit_code=0,
        full_scenario_count=1,
        greedy_initially_selected_count=1,
        redundancy_elimination_removed_count=0,
        selected_project_test_ids=("module.py::target::scenario-1",),
        removed_project_test_ids=(),
        coverage_preserved=True,
        duration_seconds=0.1,
        output_root=output,
        full_test_file=full,
        minimized_test_file=minimized,
        report_path=report,
        artifact_paths=(full, minimized, report),
    )
    report.write_text("{}", encoding="utf-8")
    external_report = output / "external_source_analysis_report.json"
    external_report.write_text("{}", encoding="utf-8")
    return ExternalSourceAnalysisResult(
        source_kind=request.source.source_kind,
        execution_policy=request.execution_policy,
        status=ExternalAnalysisStatus.PARTIAL,
        acquisition_status="READY",
        repository_name=None,
        github_owner=None,
        github_repository=None,
        resolved_commit_sha=None,
        discovered_module_count=1,
        selected_module_count=1,
        module_results=(),
        output_root=output,
        report_path=external_report,
        duration_seconds=0.2,
        cleanup_status=ExternalWorkspaceCleanupStatus.COMPLETED,
        issues=(),
        project_coverage=project_coverage,
    )


def test_job_result_and_artifacts_expose_optional_safe_project_coverage(
    tmp_path: Path,
) -> None:
    runner = Mock()
    runner.run.side_effect = _result
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path),
        runner_factory=Mock(return_value=runner),
    )
    payload = {
        "source_code": "def target():\n    return 1\n",
        "analysis": {
            "policy": "TRUSTED_DYNAMIC_ANALYSIS",
            "trusted_execution_acknowledged": True,
            "function_pipeline_timeout_seconds": 30,
        },
    }
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post("/api/v1/jobs/inline", json=payload)
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=5)
        response = client.get(f"/api/v1/jobs/{job_id}/result")
        artifacts = client.get(f"/api/v1/jobs/{job_id}/artifacts")
        artifact_items = artifacts.json()["artifacts"]
        downloads = {
            item["filename"]: client.get(
                f"/api/v1/jobs/{job_id}/artifacts/{item['artifact_id']}"
            )
            for item in artifact_items
            if item["filename"] in {
                "test_project_full_pool.py",
                "test_project_greedy_minimized.py",
                "project_coverage_report.json",
            }
        }

    assert response.status_code == 200
    body = response.json()
    assert body["project_line_coverage_percent"] == 100.0
    assert body["project_coverage"]["coverage_scope"] == "ANALYZED_PROJECT_SCOPE_COVERAGE"
    assert body["project_coverage"]["whole_repository_line_coverage_percent"] is None
    filenames = {item["filename"] for item in artifacts.json()["artifacts"]}
    assert {
        "test_project_full_pool.py",
        "test_project_greedy_minimized.py",
        "project_coverage_report.json",
    } <= filenames
    assert set(downloads) == {
        "test_project_full_pool.py",
        "test_project_greedy_minimized.py",
        "project_coverage_report.json",
    }
    assert all(response.status_code == 200 for response in downloads.values())
    service.shutdown()
