from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

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
from services.external_source_analysis_service import ExternalSourceAnalysisService


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
