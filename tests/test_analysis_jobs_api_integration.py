from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.app import create_app
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings
from services.external_source_analysis_service import ExternalSourceAnalysisService
from services.source_acquisition_service import SourceAcquisitionService


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
    marker = "price * quantity"
    source = (
        "def calculate_total(price: float, quantity: int) -> float:\n"
        "    if price < 0:\n        return 0.0\n"
        "    if quantity <= 0:\n        return 0.0\n"
        "    return price * quantity\n\n"
        "def classify_stock(stock: int) -> str:\n"
        "    if stock < 0:\n        return 'invalid'\n"
        "    if stock == 0:\n        return 'out_of_stock'\n"
        "    return 'available'\n"
    )
    service = _service(tmp_path)
    before_path = tuple(sys.path)
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={"source_code": source},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        status = client.get(f"/api/v1/jobs/{job_id}")
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        artifacts = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert status.json()["status"] == "COMPLETED"
    assert result.json()["analysis_policy"] == "STATIC_DISCOVERY_ONLY"
    assert result.json()["discovered_module_count"] == 1
    assert result.json()["selected_module_count"] == 1
    assert result.json()["discovered_function_count"] == 2
    assert result.json()["analyzed_function_count"] == 0
    assert result.json()["modules"][0]["discovered_function_names"] == [
        "calculate_total",
        "classify_stock",
    ]
    assert result.json()["modules"][0]["functions"] == []
    assert result.json()["project_line_coverage_percent"] is None
    assert result.json()["project_branch_coverage_percent"] is None
    assert result.json()["duration_seconds"] >= 0.0
    assert result.json()["cleanup_status"] == "COMPLETED"
    assert marker not in json.dumps(status.json())
    assert marker not in json.dumps(result.json())
    assert tuple(sys.path) == before_path
    assert artifacts.status_code == 200
    assert all(item["content_type"] != "text/x-python" for item in artifacts.json()["artifacts"])
    service.shutdown()


def test_real_inline_static_accepts_nonblank_sources_without_functions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    sources = ("# comment-only Python source\n", "answer = 42\n")

    with TestClient(create_app(job_service=service)) as client:
        for source in sources:
            submitted = client.post(
                "/api/v1/jobs/inline",
                json={"source_code": source},
            )
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            service.wait(job_id, timeout=30)
            result = client.get(f"/api/v1/jobs/{job_id}/result")
            assert result.status_code == 200
            assert result.json()["status"] == "COMPLETED"
            assert result.json()["discovered_function_count"] == 0
            assert source not in json.dumps(result.json())

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
    assert function["scenario_pool_line_coverage_percent"] is not None
    assert function["scenario_pool_branch_coverage_percent"] is not None
    assert function["greedy_line_coverage_percent"] is not None
    assert function["greedy_branch_coverage_percent"] is not None
    assert function["greedy_coverage_preserved"] is True
    assert function["best_rl_line_coverage_percent"] is not None
    assert function["best_rl_branch_coverage_percent"] is not None
    assert function["best_rl_coverage_preserved"] is True
    assert function["line_coverage_percent"] == function[
        "scenario_pool_line_coverage_percent"
    ]
    assert function["branch_coverage_percent"] == function[
        "scenario_pool_branch_coverage_percent"
    ]
    assert function["greedy_selected_count"] is not None
    assert function["rl_selected_count"] is not None
    assert function["strategy_winner"] is not None
    assert function["concrete_accepted_count"] is not None
    assert function["concrete_rejected_count"] is not None
    assert function["q_table_state_count"] is not None
    assert function["duration_seconds"] is not None
    assert function["comparison_status"] is not None
    assert function["greedy_reduction_percentage"] is not None
    assert function["rl_reduction_percentage"] is not None
    assert function["coverage_equality_verified"] is True
    assert function["globally_minimal"] is False
    assert payload["project_line_coverage_percent"] is not None
    assert payload["project_branch_coverage_percent"] is not None
    assert payload["project_coverage"]["coverage_scope"] == (
        "ANALYZED_PROJECT_SCOPE_COVERAGE"
    )
    assert payload["project_coverage"]["coverage_preserved"] is True
    assert payload["project_coverage"]["globally_minimal"] is False
    assert payload["project_coverage"]["whole_repository_line_coverage_percent"] is None
    assert payload["cleanup_status"] == "COMPLETED"
    assert artifacts.json()["artifacts"]
    assert tuple(sys.path) == before_path
    service.shutdown()


def test_tuple_handler_rejection_does_not_become_internal_worker_error(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""\
def convert(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return -1
"""
    analysis = {
        "policy": "TRUSTED_DYNAMIC_ANALYSIS",
        "trusted_execution_acknowledged": True,
        "maximum_module_count": 1,
        "maximum_function_count": 1,
        "episode_count": 1,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30,
        "function_pipeline_timeout_seconds": 90,
        "greedy_minimization": False,
        "strategy_comparison": False,
    }
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("tuple_handler.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        result = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert result.status_code == 200
    service.shutdown()


def test_attribute_truthiness_rejection_returns_safe_partial_result(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""\
def inspect(custom_object):
    if not custom_object.attribute:
        return "missing"
    return "present"
"""
    analysis = {
        "policy": "TRUSTED_DYNAMIC_ANALYSIS",
        "trusted_execution_acknowledged": True,
        "maximum_module_count": 1,
        "maximum_function_count": 1,
        "episode_count": 1,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30,
        "function_pipeline_timeout_seconds": 90,
        "greedy_minimization": False,
        "strategy_comparison": False,
    }
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("attribute_truthiness.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        result = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot["status"] == "PARTIAL"
    assert snapshot["safe_error_category"] is None
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "PARTIAL"
    assert payload["modules"][0]["functions"][0]["status"] == "PARTIAL"
    assert payload["modules"][0]["functions"][0]["error_category"] == (
        "CONTROLLED_FAILURE"
    )
    service.shutdown()


def test_same_dynamic_configuration_is_coverage_deterministic(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""def classify(value: int) -> str:\n    if value < 0:\n        return 'negative'\n    if value == 0:\n        return 'zero'\n    return 'positive'\n"""
    analysis = {
        "policy": "TRUSTED_DYNAMIC_ANALYSIS",
        "trusted_execution_acknowledged": True,
        "maximum_module_count": 1,
        "maximum_function_count": 1,
        "episode_count": 2,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30,
        "function_pipeline_timeout_seconds": 90,
        "greedy_minimization": True,
        "strategy_comparison": True,
    }
    signatures = []
    with TestClient(create_app(job_service=service)) as client:
        for _ in range(2):
            submitted = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("deterministic.py", source, "text/x-python")},
                data={"analysis": json.dumps(analysis)},
            )
            job_id = submitted.json()["job_id"]
            external = service.wait(job_id, timeout=120)
            public_function = client.get(
                f"/api/v1/jobs/{job_id}/result"
            ).json()["modules"][0]["functions"][0]
            function = external.module_results[0].project_result.function_results[0]
            pool = function.scenario_pool_coverage
            greedy = function.minimization_result
            comparison = function.strategy_comparison
            assert pool is not None
            assert greedy is not None
            assert comparison is not None
            signatures.append(
                {
                    "scenario_count": function.scenario_count,
                    "pool_lines": pool.covered_lines,
                    "pool_branches": pool.covered_branches,
                    "pool_percentages": (
                        pool.line_coverage_percent,
                        pool.branch_coverage_percent,
                    ),
                    "greedy_ids": greedy.final_selected_scenario_ids,
                    "greedy_lines": greedy.final_verified_line_identities,
                    "greedy_branches": greedy.final_verified_branch_identities,
                    "rl_traces": tuple(
                        trace.ordered_scenario_ids
                        for trace in comparison.episode_traces
                    ),
                    "best_rl_suite": comparison.best_rl_ordered_scenario_ids,
                    "random_seed": comparison.random_seed,
                    "function_coverage_fields": {
                        key: public_function[key]
                        for key in (
                            "line_coverage_percent",
                            "branch_coverage_percent",
                            "scenario_pool_line_coverage_percent",
                            "scenario_pool_branch_coverage_percent",
                            "greedy_line_coverage_percent",
                            "greedy_branch_coverage_percent",
                            "greedy_coverage_preserved",
                            "best_rl_line_coverage_percent",
                            "best_rl_branch_coverage_percent",
                            "best_rl_coverage_preserved",
                        )
                    },
                }
            )

    assert signatures[0] == signatures[1]
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


def test_successful_github_without_python_is_partial_through_public_api(
    tmp_path: Path,
) -> None:
    def git_runner(
        arguments: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "README.md").write_text("public fixture\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "b" * 40, "")

    analysis_runner = ExternalSourceAnalysisService(
        acquisition_service=SourceAcquisitionService(
            subprocess_runner=git_runner,
            git_executable="git",
        ),
        orchestrator_factory=Mock(),
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path / "api-output"),
        runner_factory=Mock(return_value=analysis_runner),
    )
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/github",
            json={"repository_url": "https://github.com/owner/repository"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()

    assert snapshot["status"] == "PARTIAL"
    assert snapshot["progress_stage"] == "PARTIAL"
    assert snapshot["safe_error_category"] is None
    assert result["status"] == "PARTIAL"
    assert result["source_kind"] == "PUBLIC_GITHUB_REPOSITORY"
    assert result["analysis_policy"] == "STATIC_DISCOVERY_ONLY"
    assert result["acquisition_status"] == "PARTIAL"
    assert result["discovered_module_count"] == 0
    assert result["discovered_function_count"] == 0
    assert result["analyzed_function_count"] == 0
    assert result["project_line_coverage_percent"] is None
    assert result["project_branch_coverage_percent"] is None
    assert result["cleanup_status"] == "COMPLETED"
    assert result["issues"] == ["NO_PYTHON_FILES"]
    assert result["modules"] == []
    service.shutdown()
