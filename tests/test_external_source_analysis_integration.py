from __future__ import annotations

import json
import codecs
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from models.analysis_job_result import AnalysisJobStatus
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalSourceAnalysisRequest,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    LocalProjectDirectory,
    UploadedPythonFile,
)
from models.pipeline_diagnostic_result import PipelineDiagnosticResult
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings
from services.external_source_analysis_service import ExternalSourceAnalysisService
from services.real_rl_training_service import RealRLTrainingResult, RealRLTrainingService
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator


BRANCH_SOURCE = """\
def classify(value: int) -> str:
    if value > 0:
        return "positive"
    return "non_positive"
"""

THREE_FUNCTION_SOURCE = """\
def first(value: int) -> int:
    return value + 1

def second(value: int) -> int:
    return value + 2

def third(value: int) -> int:
    return value + 3
"""

UNSUPPORTED_ATTRIBUTE_SOURCE = """\
def inspect(custom_object):
    if not custom_object.attribute:
        return "missing"
    return "present"
"""

INSTANCE_METHOD_SOURCE = """\
class Flag:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def label(self) -> str:
        if self.enabled:
            return "enabled"
        return "disabled"
"""


def _dynamic_configuration(output_root: Path, **values: object) -> ExternalAnalysisConfiguration:
    return ExternalAnalysisConfiguration(
        output_root=output_root,
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        pytest_coverage_timeout_seconds=30.0,
        **values,
    )


def test_real_inline_dynamic_analysis_persists_artifacts_and_cleans(tmp_path: Path) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(BRANCH_SOURCE),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "inline_output",
                run_greedy_baseline=True,
                run_strategy_comparison=True,
            ),
        )
    )

    assert result.status in {ExternalAnalysisStatus.COMPLETED, ExternalAnalysisStatus.PARTIAL}
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert result.module_results[0].project_result is not None
    assert result.report_path.is_file()
    assert all(path.is_file() for path in result.module_results[0].artifact_paths)
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["execution_policy"] == "TRUSTED_DYNAMIC_ANALYSIS"
    assert "rl-unit-test-inline-" not in json.dumps(payload)
    assert payload["modules"][0]["relative_path"] == "inline_source.py"
    for json_artifact in result.output_root.rglob("*.json"):
        serialized = json_artifact.read_text(encoding="utf-8")
        assert "rl-unit-test-inline-" not in serialized
        assert re.search(r"source_[0-9a-f]{32}", serialized) is None


def test_real_uploaded_dynamic_analysis_uses_distinct_tool_workspace(tmp_path: Path) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile("branch.py", BRANCH_SOURCE.encode("utf-8")),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(tmp_path / "upload_output"),
        )
    )

    assert result.status in {ExternalAnalysisStatus.COMPLETED, ExternalAnalysisStatus.PARTIAL}
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.COMPLETED
    assert result.repository_name == "branch"
    assert "def classify" not in result.report_path.read_text(encoding="utf-8")
    assert result.module_results[0].relative_path == "branch.py"
    for json_artifact in result.output_root.rglob("*.json"):
        assert "rl-unit-test-upload-" not in json_artifact.read_text(encoding="utf-8")


def test_real_uploaded_bom_dynamic_analysis_completes_without_internal_error(
    tmp_path: Path,
) -> None:
    source = codecs.BOM_UTF8 + BRANCH_SOURCE.encode("utf-8")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile("bom_branch.py", source),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(tmp_path / "bom_output"),
        )
    )

    assert result.status in {
        ExternalAnalysisStatus.COMPLETED,
        ExternalAnalysisStatus.PARTIAL,
    }
    assert "INTERNAL_WORKER_ERROR" not in result.issues
    assert result.module_results[0].project_result is not None
    assert result.module_results[0].project_result.function_results


def test_real_inline_dynamic_analysis_exposes_qualified_instance_method(
    tmp_path: Path,
) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(INSTANCE_METHOD_SOURCE),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "method_output",
                run_greedy_baseline=True,
            ),
        )
    )

    module = result.module_results[0]
    assert module.project_result is not None
    method = next(
        item
        for item in module.project_result.function_results
        if item.target.qualified_name == "Flag.label"
    )
    assert method.status.value == "COMPLETED"
    assert method.scenario_pool_coverage is not None
    assert method.scenario_pool_coverage.branch_coverage_percent == 100.0
    assert all(
        "self" not in target.parameters
        for target in module.project_result.discovered_targets
        if target.is_method
    )
    project_payload = json.loads(
        module.project_result.report_path.read_text(encoding="utf-8")
    )
    assert all(
        "self" not in target["parameters"]
        for target in project_payload["discovered_functions"]
        if target["is_method"]
    )
    assert result.project_coverage is not None
    project_test_ids = (
        *result.project_coverage.selected_project_test_ids,
        *result.project_coverage.removed_project_test_ids,
    )
    assert project_test_ids
    assert all("::Flag.label::" in value for value in project_test_ids)
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert '"qualified_name": "Flag.label"' in serialized
    assert "keyword_arguments" not in serialized
    assert "constructor_arguments" not in serialized


def test_uploaded_generated_pytest_is_portable_with_public_import_name(
    tmp_path: Path,
) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile("branch-file.py", BRANCH_SOURCE.encode("utf-8")),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "upload_output",
                run_greedy_baseline=True,
            ),
        )
    )
    project_coverage = result.project_coverage
    assert project_coverage is not None
    assert result.module_results[0].module_name == "upload_branch_file"

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    source_file = downloaded / "upload_branch_file.py"
    source_file.write_text(BRANCH_SOURCE, encoding="utf-8")
    test_file = downloaded / project_coverage.full_test_file.name
    shutil.copy2(project_coverage.full_test_file, test_file)
    generated = test_file.read_text(encoding="utf-8")

    assert "from upload_branch_file import classify" in generated
    assert re.search(r"source_[0-9a-f]{32}", generated) is None
    completed = subprocess.run(
        (sys.executable, "-m", "pytest", test_file.name, "-q"),
        cwd=downloaded,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_local_package_import_uses_subprocess_root_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text("MARKER = 'loaded'\n", encoding="utf-8")
    (package / "service.py").write_text(
        "from .helpers import MARKER\n\n" + BRANCH_SOURCE,
        encoding="utf-8",
    )
    selection = ExternalModuleSelection(
        ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES,
        ("package.service",),
    )
    before_path = tuple(sys.path)
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "local_output",
                module_selection=selection,
            ),
        )
    )

    assert result.status in {ExternalAnalysisStatus.COMPLETED, ExternalAnalysisStatus.PARTIAL}
    assert result.cleanup_status is ExternalWorkspaceCleanupStatus.NOT_REQUIRED
    assert tuple(sys.path) == before_path
    assert project.is_dir()


def test_real_three_function_limit_is_visible_end_to_end(tmp_path: Path) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(THREE_FUNCTION_SOURCE),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "function_limit_output",
                maximum_functions_per_module=2,
            ),
        )
    )

    assert result.status is ExternalAnalysisStatus.PARTIAL
    module = result.module_results[0]
    assert module.discovered_function_count == 3
    assert module.analyzed_function_count == 2
    assert module.limit_skipped_function_count == 1
    project = module.project_result
    assert project is not None
    assert [item.status.value for item in project.function_results] == [
        "COMPLETED",
        "COMPLETED",
        "SKIPPED_LIMIT",
    ]
    assert project.function_results[-1].diagnostic is None
    assert result.project_coverage is not None
    assert result.project_coverage.scope.scope_complete is False
    assert result.project_coverage.scope.skipped_limit_function_count == 1
    assert result.project_coverage.status.value == "PARTIAL"
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["analyzed_function_count"] == 2
    assert payload["limit_skipped_function_count"] == 1
    assert payload["modules"][0]["functions"][-1]["skip_reason"] == "FUNCTION_LIMIT_EXCEEDED"


def test_all_attribute_truthiness_paths_produce_controlled_partial_result(
    tmp_path: Path,
) -> None:
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            InlinePythonSource(UNSUPPORTED_ATTRIBUTE_SOURCE),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(tmp_path / "unsupported_attribute_output"),
        )
    )

    assert result.status is ExternalAnalysisStatus.PARTIAL
    module = result.module_results[0]
    assert module.status.value == "PARTIAL"
    function = module.project_result.function_results[0]
    assert function.status.value == "PARTIAL"
    assert function.diagnostic.scenario_rejection_counts == (
        ("UNSUPPORTED_INPUT_SYNTHESIS", 2),
    )
    assert function.scenario_count is None


def test_real_multi_module_project_produces_exact_combined_minimized_suite(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    package = project / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "first.py").write_text(
        "def first(values: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total\n\n"
        "def second(value: int) -> str:\n"
        "    if value == 0:\n"
        "        return 'zero'\n"
        "    return 'nonzero'\n",
        encoding="utf-8",
    )
    (package / "second.py").write_text(
        "from .first import first\n\n"
        "def third(value: int) -> str:\n"
        "    if value >= 5:\n"
        "        return 'large'\n"
        "    return 'small'\n",
        encoding="utf-8",
    )
    selection = ExternalModuleSelection(
        ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES,
        ("package.first", "package.second"),
    )

    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            _dynamic_configuration(
                tmp_path / "combined_output",
                module_selection=selection,
                run_greedy_baseline=True,
            ),
        )
    )

    combined = result.project_coverage
    assert combined is not None
    assert combined.full_pytest_exit_code == 0
    assert combined.minimized_pytest_exit_code == 0
    assert combined.coverage_preserved is True
    assert combined.final_selected_count < combined.full_scenario_count
    assert combined.target_line_identities == combined.minimized_covered_line_identities
    assert combined.target_branch_identities == combined.minimized_covered_branch_identities
    assert combined.scope.discovered_function_count == 3
    assert combined.scope.completed_function_count == 3
    assert combined.scope.scope_complete is True
    assert all(path.is_file() for path in combined.artifact_paths)
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["project_coverage"]["coverage_scope"] == "ANALYZED_PROJECT_SCOPE_COVERAGE"
    assert payload["project_coverage"]["whole_repository_line_coverage_percent"] is None


class _MutableProjectClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _DynamicDeadlineRunController:
    def __init__(self, clock: _MutableProjectClock) -> None:
        self.clock = clock
        self.mode = ""
        self.deadlines: dict[str, list[object]] = {"short": [], "wide": []}
        self.calls: dict[str, list[dict[str, object]]] = {
            "short": [],
            "wide": [],
        }

    def begin(self, mode: str) -> None:
        self.mode = mode
        self.calls[mode].clear()

    def record_deadline(self, deadline: object) -> None:
        self.deadlines[self.mode].append(deadline)

    def record_result(
        self,
        *,
        arguments: dict[str, object],
        result: RealRLTrainingResult | PipelineDiagnosticResult,
    ) -> None:
        entries = self.calls[self.mode]
        entries.append(
            {
                "function_name": arguments["function_name"],
                "pipeline_timeout_seconds": arguments[
                    "pipeline_timeout_seconds"
                ],
                "result_type": type(result).__name__,
                "status": result.diagnostic.status.value
                if isinstance(result, RealRLTrainingResult)
                else result.status.value,
            }
        )
        deadline = self.deadlines[self.mode][-1]
        if self.mode == "short":
            if len(entries) == 1:
                self.clock.value = (
                    deadline.started_at + deadline.timeout_seconds - 0.05
                )
            else:
                self.clock.value = deadline.started_at + deadline.timeout_seconds
        else:
            self.clock.value += 0.1


class _RecordingTrainingService:
    def __init__(self, controller: _DynamicDeadlineRunController) -> None:
        self._controller = controller

    def run_with_diagnostics(
        self, **arguments: object
    ) -> RealRLTrainingResult | PipelineDiagnosticResult:
        result = RealRLTrainingService().run_with_diagnostics(**arguments)
        self._controller.record_result(arguments=arguments, result=result)
        return result


class _RecordingOrchestrator:
    def __init__(
        self,
        delegate: SourceAnalysisOrchestrator,
        controller: _DynamicDeadlineRunController,
    ) -> None:
        self._delegate = delegate
        self._controller = controller

    def run(self, **arguments: object):
        self._controller.record_deadline(arguments["project_deadline"])
        return self._delegate.run(**arguments)


@pytest.mark.parametrize(
    ("module_name", "function_names", "literals"),
    (
        ("harbor_tasks", ("signal", "anchor", "depart"), (11, 23, 37)),
        ("garden_jobs", ("seed", "irrigate", "harvest"), (41, 53, 67)),
    ),
)
def test_trusted_dynamic_second_run_resets_project_deadline_and_job_state(
    tmp_path: Path,
    module_name: str,
    function_names: tuple[str, str, str],
    literals: tuple[int, int, int],
) -> None:
    first_name, active_name, last_name = function_names
    first_literal, threshold, last_literal = literals
    source_marker = f"private_source_marker_{module_name}"
    project = tmp_path / f"{module_name}_project"
    project.mkdir()
    (project / f"{module_name}.py").write_text(
        f"# {source_marker}\n"
        f"def {first_name}() -> int:\n"
        f"    return {first_literal}\n\n"
        f"def {active_name}(value: int) -> str:\n"
        f"    if value < {threshold}:\n"
        "        return 'below'\n"
        "    return 'at_or_above'\n\n"
        f"def {last_name}(enabled: bool) -> int:\n"
        f"    return {last_literal} if enabled else -{last_literal}\n",
        encoding="utf-8",
    )

    clock = _MutableProjectClock()
    controller = _DynamicDeadlineRunController(clock)

    def orchestrator_factory() -> _RecordingOrchestrator:
        delegate = SourceAnalysisOrchestrator(
            training_service_factory=lambda: _RecordingTrainingService(controller),
            clock=clock,
        )
        return _RecordingOrchestrator(delegate, controller)

    external_service = ExternalSourceAnalysisService(
        orchestrator_factory=orchestrator_factory,
        clock=clock,
    )
    job_root = tmp_path / f"{module_name}_jobs"
    job_service = AnalysisJobService(
        settings=AnalysisJobSettings(
            output_root=job_root,
            maximum_running_jobs=1,
            maximum_queued_jobs=1,
        ),
        runner_factory=lambda: external_service,
    )
    common_configuration = {
        "output_root": tmp_path / "request_output_is_replaced_per_job",
        "max_visits_per_node": 2,
        "episode_count": 1,
        "epsilon": 0.0,
        "learning_rate": 0.5,
        "discount_factor": 0.9,
        "random_seed": 42,
        "pytest_coverage_timeout_seconds": 30.0,
        "per_function_pipeline_timeout_seconds": 30.0,
        "run_greedy_baseline": False,
        "run_strategy_comparison": False,
    }

    def request(project_timeout_seconds: float) -> ExternalSourceAnalysisRequest:
        return ExternalSourceAnalysisRequest(
            LocalProjectDirectory(project),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            ExternalAnalysisConfiguration(
                **common_configuration,
                project_timeout_seconds=project_timeout_seconds,
            ),
        )

    try:
        controller.begin("short")
        first_job = job_service.submit(request(10.0))
        first = job_service.wait(first_job.job_id, timeout=90.0)
        first_public_before = json.dumps(first.to_dict(), sort_keys=True)
        first_summary = job_service.get(first_job.job_id)
        first_result_summary = job_service.get_result(first_job.job_id)
        first_artifacts = job_service.list_artifacts(first_job.job_id)

        assert first.status is ExternalAnalysisStatus.PARTIAL
        assert first.project_coverage is None
        assert first.project_deadline_exceeded is True
        assert first.completed_function_count == 1
        assert first.timed_out_function_count == 1
        assert first.deadline_skipped_function_count == 1
        assert first_summary.status is AnalysisJobStatus.PARTIAL
        assert first_summary.cancellation_requested is False
        assert first_result_summary.status is AnalysisJobStatus.PARTIAL
        first_function_results = first.module_results[0].project_result.function_results
        assert [item.target.qualified_name for item in first_function_results] == list(
            function_names
        )
        assert [item.status.value for item in first_function_results] == [
            "COMPLETED",
            "TIMED_OUT",
            "SKIPPED_DEADLINE",
        ]
        assert [item["function_name"] for item in controller.calls["short"]] == [
            first_name,
            active_name,
        ]
        assert controller.calls["short"][1]["status"] == "TIMED_OUT"
        assert controller.calls["short"][1][
            "pipeline_timeout_seconds"
        ] == pytest.approx(0.05)

        controller.begin("wide")
        second_job = job_service.submit(request(120.0))
        second = job_service.wait(second_job.job_id, timeout=180.0)
        second_summary = job_service.get(second_job.job_id)
        second_result_summary = job_service.get_result(second_job.job_id)
        second_artifacts = job_service.list_artifacts(second_job.job_id)

        assert second.status is ExternalAnalysisStatus.COMPLETED
        assert second.project_deadline_exceeded is False
        assert second.deadline_stage is None
        assert second.last_completed_stage == "PROJECT_COVERAGE"
        assert second.completed_function_count == 3
        assert second.timed_out_function_count == 0
        assert second.deadline_skipped_function_count == 0
        assert second_summary.status is AnalysisJobStatus.COMPLETED
        assert second_summary.cancellation_requested is False
        assert second_summary.deadline_stage is None
        assert second_result_summary.status is AnalysisJobStatus.COMPLETED
        second_function_results = second.module_results[0].project_result.function_results
        assert [item.target.qualified_name for item in second_function_results] == list(
            function_names
        )
        assert [item.status.value for item in second_function_results] == [
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
        ]
        assert [item["function_name"] for item in controller.calls["wide"]] == list(
            function_names
        )
        assert all(
            item["result_type"] == "RealRLTrainingResult"
            and item["status"] == "COMPLETED"
            and item["pipeline_timeout_seconds"] == pytest.approx(30.0)
            for item in controller.calls["wide"]
        )

        assert len(controller.deadlines["short"]) == 1
        assert len(controller.deadlines["wide"]) == 1
        short_deadline = controller.deadlines["short"][0]
        wide_deadline = controller.deadlines["wide"][0]
        assert wide_deadline is not short_deadline
        assert wide_deadline.started_at > short_deadline.started_at
        assert short_deadline.remaining_seconds() == 0.0
        assert wide_deadline.remaining_seconds() == pytest.approx(119.7)

        combined = second.project_coverage
        assert combined is not None
        assert combined.status.value == "COMPLETED"
        assert combined.full_pytest_exit_code == 0
        assert combined.minimized_pytest_exit_code == 0
        assert combined.full_scenario_count > 0
        assert combined.full_line_coverage_percent is not None
        assert combined.full_branch_coverage_percent is not None

        assert json.dumps(first.to_dict(), sort_keys=True) == first_public_before
        assert first.report_path != second.report_path
        assert first.report_path.is_relative_to(job_root / first_job.job_id)
        assert second.report_path.is_relative_to(job_root / second_job.job_id)
        assert {item.artifact_id for item in first_artifacts}.isdisjoint(
            item.artifact_id for item in second_artifacts
        )
        for job_id, artifacts in (
            (first_job.job_id, first_artifacts),
            (second_job.job_id, second_artifacts),
        ):
            for artifact in artifacts:
                _, artifact_path = job_service.artifact_path(
                    job_id, artifact.artifact_id
                )
                assert artifact_path.is_relative_to(job_root / job_id)

        public_payload = json.dumps(second_result_summary.to_dict(), sort_keys=True)
        assert source_marker not in public_payload
        assert source_marker not in second.report_path.read_text(encoding="utf-8")
        for forbidden in (
            "keyword_arguments",
            "expected_result",
            "actual_result",
            "object_state",
        ):
            assert forbidden not in public_payload
    finally:
        job_service.shutdown()
