from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def _service(tmp_path: Path) -> AnalysisJobService:
    return AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=tmp_path / "api-output",
            maximum_running_jobs=1,
            maximum_queued_jobs=2,
            maximum_upload_bytes=100_000,
            maximum_inline_source_bytes=100_000,
        )
    )


def test_real_inline_static_job_is_safe_and_pollable(tmp_path: Path) -> None:
    marker = "never_return_raw_source"
    service = _service(tmp_path)
    before_path = tuple(sys.path)
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": f"def target():\n    return {marker!r}\n"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        status = client.get(f"/api/v1/jobs/{job_id}")
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        artifacts = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert status.json()["status"] == "COMPLETED"
    assert result.json()["analysis_policy"] == "STATIC_DISCOVERY_ONLY"
    assert result.json()["project_line_coverage_percent"] is None
    assert marker not in json.dumps(status.json())
    assert marker not in json.dumps(result.json())
    assert tuple(sys.path) == before_path
    assert artifacts.status_code == 200
    service.shutdown()


def test_real_upload_trusted_dynamic_runs_coverage_greedy_and_comparison(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""def classify(value: int) -> str:\n    if value > 0:\n        return 'positive'\n    return 'other'\n"""
    analysis = {
        "policy": "TRUSTED_DYNAMIC_ANALYSIS",
        "trusted_execution_acknowledged": True,
        "maximum_module_count": 1,
        "maximum_function_count": 1,
        "episode_count": 1,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30,
        "function_pipeline_timeout_seconds": 90,
        "greedy_minimization": True,
        "strategy_comparison": True,
    }
    before_path = tuple(sys.path)
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("branch.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        artifacts = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    payload = result.json()
    assert result.status_code == 200
    assert payload["status"] in {"COMPLETED", "PARTIAL"}
    function = payload["modules"][0]["functions"][0]
    assert function["line_coverage_percent"] is not None
    assert function["branch_coverage_percent"] is not None
    assert function["greedy_selected_count"] is not None
    assert function["rl_selected_count"] is not None
    assert function["strategy_winner"] is not None
    assert payload["project_line_coverage_percent"] is None
    assert artifacts.json()["artifacts"]
    assert tuple(sys.path) == before_path
    service.shutdown()


def test_risky_github_url_is_rejected_by_real_acquisition_without_network(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/github",
            json={"repository_url": "http://github.com/owner/repository"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["status"] == "FAILED"
    assert result.json()["issues"]
    service.shutdown()
