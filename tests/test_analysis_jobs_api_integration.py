from __future__ import annotations

import json
import codecs
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.app import create_app
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings
from services.external_source_analysis_service import ExternalSourceAnalysisService
from services.pipeline_timeout_service import GlobalPipelineTimeoutRunner
from services.real_rl_training_service import RealRLTrainingService
from services.source_acquisition_service import SourceAcquisitionService
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator


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


def test_real_upload_static_accepts_utf8_bom_without_public_source_leak(
    tmp_path: Path,
) -> None:
    source = (
        codecs.BOM_UTF8
        + b"def api_bom_target(value: int) -> int:\n    return value + 4\n"
    )
    service = _service(tmp_path)

    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("api_bom.py", source, "text/x-python")},
            data={"analysis": "{}"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        result = client.get(f"/api/v1/jobs/{job_id}/result")

    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "COMPLETED"
    assert payload["modules"][0]["discovered_function_names"] == [
        "api_bom_target"
    ]
    serialized = json.dumps(payload)
    assert "def api_bom_target" not in serialized
    assert source.hex() not in serialized
    service.shutdown()


def test_real_hyphenated_upload_explicit_target_matches_canonical_module(
    tmp_path: Path,
) -> None:
    source = b"def chosen():\n    return 7\n\ndef other():\n    return 9\n"
    analysis = {
        "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
        "explicit_target_names": ["chosen"],
    }
    service = _service(tmp_path)

    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("my-file.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    assert payload["modules"][0]["discovered_function_names"] == [
        "chosen",
        "other",
    ]
    serialized = json.dumps(payload)
    assert source.decode("utf-8") not in serialized
    assert source.hex() not in serialized
    assert str(tmp_path) not in serialized
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


def test_unknown_explicit_target_is_controlled_without_internal_worker_error(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/inline",
            json={
                "source_code": "def available() -> int:\n    return 1\n",
                "analysis": {
                    "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
                    "explicit_target_names": ["Missing.run"],
                },
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=30)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "FAILED"
    assert payload["issues"] == ["UNKNOWN_TARGET_SELECTION"]
    assert "available" not in json.dumps(payload)
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


def test_pipeline_cleanup_failure_is_a_safe_api_result(
    tmp_path: Path,
) -> None:
    private_detail = str(tmp_path / "private-cleanup-path")
    retained_roots: list[Path] = []

    def fail_cleanup(path: Path, **kwargs: object) -> None:
        del kwargs
        retained_roots.append(Path(path))
        raise PermissionError(private_detail)

    timeout_runner = GlobalPipelineTimeoutRunner(
        cleanup_attempts=1,
        cleanup_backoff_seconds=0.0,
        cleanup_sleeper=lambda _: None,
        cleanup_rmtree=fail_cleanup,
    )
    external_service = ExternalSourceAnalysisService(
        orchestrator_factory=lambda: SourceAnalysisOrchestrator(
            training_service_factory=lambda: RealRLTrainingService(
                global_timeout_runner=timeout_runner
            )
        )
    )
    service = AnalysisJobService(
        settings=AnalysisJobSettings(output_root=tmp_path / "api-output"),
        runner_factory=lambda: external_service,
    )
    source = "def classify(value: int) -> str:\n    return 'yes' if value else 'no'\n"
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
    try:
        with TestClient(create_app(job_service=service)) as client:
            submitted = client.post(
                "/api/v1/jobs/inline",
                json={"source_code": source, "analysis": analysis},
            )
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            service.wait(job_id, timeout=120)
            snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
            response = client.get(f"/api/v1/jobs/{job_id}/result")
    finally:
        service.shutdown()
        system_temp = Path(tempfile.gettempdir()).resolve()
        for root in set(retained_roots):
            assert root.parent.resolve() == system_temp
            assert root.name.startswith("rl-unit-test-pipeline-")
            shutil.rmtree(root)
            assert not root.exists()

    assert snapshot["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert response.status_code == 200
    payload = response.json()
    function = payload["modules"][0]["functions"][0]
    assert function["error_category"] == "PIPELINE_CLEANUP_FAILED"
    serialized = json.dumps(payload)
    assert private_detail not in serialized
    assert source not in serialized


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


def test_instance_method_result_uses_qualified_name_without_runtime_state(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""\
class Flag:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def label(self) -> str:
        if self.enabled:
            return "enabled"
        return "disabled"

def helper(value: int) -> int:
    return value + 1
"""
    analysis = {
        "policy": "TRUSTED_DYNAMIC_ANALYSIS",
        "trusted_execution_acknowledged": True,
        "maximum_module_count": 1,
        "maximum_function_count": 10,
        "target_selection_mode": "EXPLICIT_QUALIFIED_TARGETS",
        "explicit_target_names": ["Flag.label"],
        "episode_count": 1,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30,
        "function_pipeline_timeout_seconds": 90,
        "greedy_minimization": True,
        "strategy_comparison": False,
    }
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("my-file.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert response.status_code == 200
    payload = response.json()
    assert client.get(f"/api/v1/jobs/{job_id}").json()[
        "safe_error_category"
    ] != "INTERNAL_WORKER_ERROR"
    method = next(
        item
        for item in payload["modules"][0]["functions"]
        if item["qualified_name"] == "Flag.label"
    )
    assert method["status"] == "COMPLETED"
    helper = next(
        item
        for item in payload["modules"][0]["functions"]
        if item["qualified_name"] == "helper"
    )
    assert helper["status"] == "SKIPPED_SELECTION"
    assert helper["skip_reason"] == "TARGET_NOT_SELECTED"
    selection_skipped = [
        item
        for item in payload["modules"][0]["functions"]
        if item["status"] == "SKIPPED_SELECTION"
    ]
    assert payload["selection_skipped_function_count"] == len(selection_skipped)
    assert helper in selection_skipped
    coverage = payload["project_coverage"]
    assert coverage["scope"]["skipped_selection_function_count"] == len(
        selection_skipped
    )
    assert coverage["scope"]["scope_complete"] is False
    assert all(
        "Flag.label" in project_test_id
        for project_test_id in coverage["selected_project_test_ids"]
    )
    serialized = json.dumps(payload)
    assert '"self"' not in serialized
    assert "module_identity" not in serialized
    assert source.decode("utf-8") not in serialized
    assert "constructor_arguments" not in serialized
    assert "keyword_arguments" not in serialized
    assert str(tmp_path) not in serialized
    service.shutdown()


def test_empty_collection_instance_state_is_not_exposed_by_public_api(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""\
class Shelf:
    def __init__(self):
        self.private_records = {}

    def missing(self, key: str) -> bool:
        return key not in self.private_records
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
            files={"file": ("shelf.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        snapshot = client.get(f"/api/v1/jobs/{job_id}")
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot.json()["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert response.status_code == 200
    payload = response.json()
    method = next(
        item
        for item in payload["modules"][0]["functions"]
        if item["qualified_name"] == "Shelf.missing"
    )
    assert method["status"] == "COMPLETED"
    serialized = json.dumps(payload)
    assert "private_records" not in serialized
    assert "__self_" not in serialized
    assert "constructor_arguments" not in serialized
    assert "keyword_arguments" not in serialized
    assert source.decode("utf-8") not in serialized
    service.shutdown()


def test_annotated_custom_object_job_is_safe_and_hides_construction_data(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = b"""\
class Parcel:
    def __init__(self, private_measure: int = 6):
        self.private_measure = private_measure

def classify(parcel: Parcel, boundary: int) -> str:
    if parcel.private_measure >= boundary:
        return "accepted"
    return "rejected"
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
        "greedy_minimization": True,
        "strategy_comparison": False,
    }
    with TestClient(create_app(job_service=service)) as client:
        submitted = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("custom_input.py", source, "text/x-python")},
            data={"analysis": json.dumps(analysis)},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        service.wait(job_id, timeout=120)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert snapshot["safe_error_category"] != "INTERNAL_WORKER_ERROR"
    assert response.status_code == 200
    payload = response.json()
    target = next(
        item
        for item in payload["modules"][0]["functions"]
        if item["qualified_name"] == "classify"
    )
    assert target["status"] == "COMPLETED"
    serialized = json.dumps(payload)
    assert "private_measure" not in serialized
    assert "constructor_arguments" not in serialized
    assert "keyword_arguments" not in serialized
    assert "module_identity" not in serialized
    assert "SafeObjectConstructionBlueprint" not in serialized
    assert "object at 0x" not in serialized
    assert source.decode("utf-8") not in serialized
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
