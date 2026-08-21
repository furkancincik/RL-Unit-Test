from __future__ import annotations

import json
import re
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
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["analyzed_function_count"] == 2
    assert payload["limit_skipped_function_count"] == 1
    assert payload["modules"][0]["functions"][-1]["skip_reason"] == "FUNCTION_LIMIT_EXCEEDED"
