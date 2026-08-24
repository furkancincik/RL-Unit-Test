from __future__ import annotations

from pathlib import Path

import pytest

from models.project_coverage_result import (
    ProjectBranchIdentity,
    ProjectCoverageResult,
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
    ProjectLineIdentity,
)


def test_project_identities_are_module_qualified_and_preserve_exit_arcs() -> None:
    line = ProjectLineIdentity("package\\module.py", 12)
    branch = ProjectBranchIdentity("package\\module.py", 12, -12)

    assert line.to_tuple() == ("package/module.py", 12)
    assert branch.to_tuple() == ("package/module.py", 12, -12)


@pytest.mark.parametrize(
    "relative_path",
    ("/absolute.py", "../escape.py", "C:/absolute.py", ""),
)
def test_project_identities_reject_non_relative_paths(relative_path: str) -> None:
    with pytest.raises(ValueError):
        ProjectLineIdentity(relative_path, 1)


def test_project_coverage_result_serializes_exact_scope_without_absolute_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "project_combined"
    full_test = output / "test_project_full_pool.py"
    minimized_test = output / "test_project_greedy_minimized.py"
    report = output / "project_coverage_report.json"
    scope = ProjectCoverageScopeSummary(
        discovered_module_count=2,
        selected_module_count=2,
        completed_module_count=2,
        discovered_function_count=3,
        eligible_function_count=3,
        analyzed_function_count=3,
        completed_function_count=3,
        partial_function_count=0,
        failed_function_count=0,
        timed_out_function_count=0,
        unsupported_function_count=0,
        skipped_limit_function_count=0,
        scope_complete=True,
    )
    lines = (
        ProjectLineIdentity("package/a.py", 2),
        ProjectLineIdentity("package/b.py", 2),
    )
    branches = (ProjectBranchIdentity("package/a.py", 2, -2),)
    result = ProjectCoverageResult(
        status=ProjectCoverageStatus.COMPLETED,
        scope=scope,
        executable_line_identities=lines,
        executable_branch_identities=branches,
        target_line_identities=lines,
        target_branch_identities=branches,
        minimized_covered_line_identities=lines,
        minimized_covered_branch_identities=branches,
        full_line_coverage_percent=100.0,
        full_branch_coverage_percent=100.0,
        minimized_line_coverage_percent=100.0,
        minimized_branch_coverage_percent=100.0,
        full_pytest_exit_code=0,
        minimized_pytest_exit_code=0,
        full_scenario_count=3,
        greedy_initially_selected_count=2,
        redundancy_elimination_removed_count=1,
        selected_project_test_ids=("package/a.py::first::scenario-1",),
        removed_project_test_ids=("package/a.py::first::scenario-2",),
        coverage_preserved=True,
        duration_seconds=0.2,
        output_root=output,
        full_test_file=full_test,
        minimized_test_file=minimized_test,
        report_path=report,
        artifact_paths=(full_test, minimized_test, report),
    )

    payload = result.to_dict()

    assert payload["coverage_scope"] == "ANALYZED_PROJECT_SCOPE_COVERAGE"
    assert payload["whole_repository_line_coverage_percent"] is None
    assert payload["whole_repository_branch_coverage_percent"] is None
    assert payload["target_line_identities"] == [
        ["package/a.py", 2],
        ["package/b.py", 2],
    ]
    assert payload["target_branch_identities"] == [["package/a.py", 2, -2]]
    assert payload["scope"]["scope_complete"] is True
    assert payload["scope"]["incomplete_function_count"] == 0
    assert payload["globally_minimal"] is False
    assert str(tmp_path.resolve()) not in str(payload)


def test_scope_serializes_selection_skips_as_incomplete_analysis_scope() -> None:
    scope = ProjectCoverageScopeSummary(
        discovered_module_count=2,
        selected_module_count=2,
        completed_module_count=2,
        discovered_function_count=4,
        eligible_function_count=4,
        analyzed_function_count=1,
        completed_function_count=1,
        partial_function_count=0,
        failed_function_count=0,
        timed_out_function_count=0,
        unsupported_function_count=0,
        skipped_limit_function_count=0,
        scope_complete=False,
        skipped_selection_function_count=3,
    )

    payload = scope.to_dict()

    assert payload["skipped_selection_function_count"] == 3
    assert payload["scope_complete"] is False
