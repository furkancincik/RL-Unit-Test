from __future__ import annotations

import ast
from pathlib import Path

import pytest

from generator.scenario_generator import Scenario
from models.project_coverage_result import (
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
    ProjectTestCandidate,
)
from services.project_coverage_service import (
    ProjectCoverageService,
    ProjectExactCoverageGreedySelector,
    ProjectScenarioCoverageSignature,
    _ProjectMeasurement,
)
from services.coverage_service import CoverageExecutionTimeoutError


def _scenario(
    identifier: str,
    *,
    value: int,
    expected: str,
    path_index: int,
) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        name=identifier,
        path_index=path_index,
        priority_rank=path_index,
        priority_level="High",
        dqm_score=1.0,
        node_ids=(1,),
        edge_labels=(None,),
        contains_loop=False,
        contains_exception=False,
        description=identifier,
        keyword_arguments=(("value", value),),
        expected_result=expected,
    )


def _function_range(path: Path, name: str) -> tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    assert node.end_lineno is not None
    return node.lineno, node.end_lineno


def _scope(*, complete: bool = True) -> ProjectCoverageScopeSummary:
    return ProjectCoverageScopeSummary(
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
        scope_complete=complete,
    )


def test_project_selector_uses_required_tie_break_and_backward_elimination() -> None:
    signatures = (
        ProjectScenarioCoverageSignature(
            project_test_id="z",
            original_order=1,
            covered_line_identities=(("a.py", 1),),
            covered_branch_identities=(("a.py", 1, 2),),
            execution_success=True,
        ),
        ProjectScenarioCoverageSignature(
            project_test_id="a",
            original_order=2,
            covered_line_identities=(("a.py", 1), ("a.py", 2)),
            covered_branch_identities=(),
            execution_success=True,
        ),
        ProjectScenarioCoverageSignature(
            project_test_id="redundant",
            original_order=3,
            covered_line_identities=(("a.py", 2),),
            covered_branch_identities=(),
            execution_success=True,
        ),
    )

    selection = ProjectExactCoverageGreedySelector.select(
        signatures=signatures,
        target_lines=(("a.py", 1), ("a.py", 2)),
        target_branches=(("a.py", 1, 2),),
    )

    assert selection.selected_project_test_ids == ("z", "a")
    assert selection.initial_selected_count == 2
    assert selection.redundancy_removed_count == 0


def test_project_selector_applies_backward_redundancy_elimination() -> None:
    line_sets = ((1,), (1, 2), (1, 3, 4), (3, 4, 5))
    signatures = tuple(
        ProjectScenarioCoverageSignature(
            project_test_id=f"scenario-{index}",
            original_order=index,
            covered_line_identities=tuple(
                ("module.py", line) for line in lines
            ),
            covered_branch_identities=(),
            execution_success=True,
        )
        for index, lines in enumerate(line_sets, start=1)
    )

    selection = ProjectExactCoverageGreedySelector.select(
        signatures=signatures,
        target_lines=tuple(("module.py", line) for line in range(1, 6)),
        target_branches=(),
    )

    assert selection.initial_selected_count == 3
    assert selection.redundancy_removed_count == 1
    assert selection.selected_project_test_ids == ("scenario-2", "scenario-4")


def test_real_two_module_project_combined_coverage_and_minimization(
    tmp_path: Path,
) -> None:
    package = tmp_path / "sample_project"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    first = package / "first.py"
    first.write_text(
        "def classify(value):\n"
        "    if value > 0:\n"
        "        return 'positive'\n"
        "    return 'other'\n\n"
        "def parity(value):\n"
        "    if value % 2:\n"
        "        return 'odd'\n"
        "    return 'even'\n",
        encoding="utf-8",
    )
    second = package / "second.py"
    second.write_text(
        "from .first import classify\n\n"
        "def render(value):\n"
        "    if classify(value) == 'positive':\n"
        "        return 'P'\n"
        "    return 'N'\n",
        encoding="utf-8",
    )
    definitions = (
        ("sample_project/first.py", "sample_project.first", first, "classify", 1, "positive"),
        ("sample_project/first.py", "sample_project.first", first, "classify", -1, "other"),
        ("sample_project/first.py", "sample_project.first", first, "classify", 1, "positive"),
        ("sample_project/first.py", "sample_project.first", first, "parity", 1, "odd"),
        ("sample_project/first.py", "sample_project.first", first, "parity", 2, "even"),
        ("sample_project/second.py", "sample_project.second", second, "render", 1, "P"),
        ("sample_project/second.py", "sample_project.second", second, "render", -1, "N"),
    )
    candidates = []
    for index, (relative, module, source, function, value, expected) in enumerate(
        definitions,
        start=1,
    ):
        start, end = _function_range(source, function)
        scenario = _scenario(
            f"scenario-{index}",
            value=value,
            expected=expected,
            path_index=index,
        )
        candidates.append(
            ProjectTestCandidate(
                project_test_id=f"{relative}::{function}::{scenario.scenario_id}",
                relative_module_path=relative,
                module_path=module,
                function_name=function,
                function_start_line=start,
                function_end_line=end,
                source_file=source,
                import_root=tmp_path,
                scenario=scenario,
                original_order=index,
            )
        )

    result = ProjectCoverageService().measure_and_minimize(
        candidates=tuple(candidates),
        scope=_scope(),
        output_root=tmp_path / "output",
        timeout_seconds=30.0,
    )

    assert result.status is ProjectCoverageStatus.COMPLETED
    assert result.scope.scope_complete is True
    assert result.full_pytest_exit_code == 0
    assert result.minimized_pytest_exit_code == 0
    assert result.full_scenario_count == len(candidates)
    assert result.final_selected_count < result.full_scenario_count
    assert result.coverage_preserved is True
    assert result.target_line_identities == result.minimized_covered_line_identities
    assert result.target_branch_identities == result.minimized_covered_branch_identities
    assert result.full_test_file.is_file()
    assert result.minimized_test_file.is_file()
    assert result.report_path.is_file()
    assert all("\\" not in item.relative_module_path for item in result.target_line_identities)
    first_lines = {
        item.line_number
        for item in result.target_line_identities
        if item.relative_module_path == "sample_project/first.py"
    }
    second_lines = {
        item.line_number
        for item in result.target_line_identities
        if item.relative_module_path == "sample_project/second.py"
    }
    assert first_lines & second_lines


def test_partial_scope_is_reported_even_when_measured_subset_is_exact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    scenario = Scenario(
        scenario_id="scenario-1",
        name="scenario-1",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=1.0,
        node_ids=(1,),
        edge_labels=(None,),
        contains_loop=False,
        contains_exception=False,
        description="identity",
        keyword_arguments=(("value", 1),),
        expected_result=1,
    )
    candidate = ProjectTestCandidate(
        project_test_id="module.py::target::scenario-1",
        relative_module_path="module.py",
        module_path="module",
        function_name="target",
        function_start_line=1,
        function_end_line=2,
        source_file=source,
        import_root=tmp_path,
        scenario=scenario,
        original_order=1,
    )
    partial_scope = ProjectCoverageScopeSummary(
        discovered_module_count=1,
        selected_module_count=1,
        completed_module_count=1,
        discovered_function_count=2,
        eligible_function_count=2,
        analyzed_function_count=1,
        completed_function_count=1,
        partial_function_count=0,
        failed_function_count=0,
        timed_out_function_count=0,
        unsupported_function_count=0,
        skipped_limit_function_count=1,
        scope_complete=False,
    )

    result = ProjectCoverageService().measure_and_minimize(
        candidates=(candidate,),
        scope=partial_scope,
        output_root=tmp_path / "output",
        timeout_seconds=30.0,
    )

    assert result.status is ProjectCoverageStatus.PARTIAL
    assert result.coverage_preserved is True
    assert result.scope.skipped_limit_function_count == 1
    assert result.full_branch_coverage_percent == 100.0
    assert result.minimized_branch_coverage_percent == 100.0


def test_full_target_below_one_hundred_is_preserved_without_inflation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "def target(value):\n"
        "    if value > 0:\n"
        "        return 'positive'\n"
        "    return 'other'\n",
        encoding="utf-8",
    )
    candidate = ProjectTestCandidate(
        project_test_id="module.py::target::positive",
        relative_module_path="module.py",
        module_path="module",
        function_name="target",
        function_start_line=1,
        function_end_line=4,
        source_file=source,
        import_root=tmp_path,
        scenario=_scenario("positive", value=1, expected="positive", path_index=1),
        original_order=1,
    )
    scope = ProjectCoverageScopeSummary(
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
        scope_complete=True,
    )

    result = ProjectCoverageService().measure_and_minimize(
        candidates=(candidate,),
        scope=scope,
        output_root=tmp_path / "output",
    )

    assert result.full_line_coverage_percent is not None
    assert result.full_line_coverage_percent < 100.0
    assert result.coverage_preserved is True
    assert result.target_line_identities == result.minimized_covered_line_identities


def test_non_additive_contributions_do_not_claim_minimum_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    candidate = ProjectTestCandidate(
        project_test_id="module.py::target::scenario-1",
        relative_module_path="module.py",
        module_path="module",
        function_name="target",
        function_start_line=1,
        function_end_line=2,
        source_file=source,
        import_root=tmp_path,
        scenario=_scenario("scenario-1", value=1, expected=1, path_index=1),
        original_order=1,
        precomputed_line_identities=(),
        precomputed_branch_identities=(),
        precomputed_execution_success=True,
    )
    scope = ProjectCoverageScopeSummary(
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
        scope_complete=True,
    )

    result = ProjectCoverageService().measure_and_minimize(
        candidates=(candidate,),
        scope=scope,
        output_root=tmp_path / "output",
    )

    assert result.status is ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
    assert result.coverage_preserved is True
    assert result.final_selected_count == result.full_scenario_count
    assert result.failure_category == "NON_ADDITIVE_COVERAGE"


def test_empty_pool_and_full_suite_timeout_persist_controlled_project_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = ProjectCoverageScopeSummary(
        discovered_module_count=1,
        selected_module_count=1,
        completed_module_count=0,
        discovered_function_count=1,
        eligible_function_count=1,
        analyzed_function_count=0,
        completed_function_count=0,
        partial_function_count=0,
        failed_function_count=1,
        timed_out_function_count=0,
        unsupported_function_count=0,
        skipped_limit_function_count=0,
        scope_complete=False,
    )
    empty = ProjectCoverageService().measure_and_minimize(
        candidates=(),
        scope=scope,
        output_root=tmp_path / "empty",
    )
    assert empty.status is ProjectCoverageStatus.FAILED
    assert empty.unmeasured_reason == "NO_USABLE_PROJECT_TESTS"
    assert empty.report_path.is_file()

    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    candidate = ProjectTestCandidate(
        project_test_id="module.py::target::scenario-1",
        relative_module_path="module.py",
        module_path="module",
        function_name="target",
        function_start_line=1,
        function_end_line=2,
        source_file=source,
        import_root=tmp_path,
        scenario=_scenario("scenario-1", value=1, expected=1, path_index=1),
        original_order=1,
    )
    service = ProjectCoverageService()

    def timeout(*args, **kwargs):
        raise CoverageExecutionTimeoutError("deadline")

    monkeypatch.setattr(service, "_measure", timeout)
    timed_out = service.measure_and_minimize(
        candidates=(candidate,),
        scope=scope,
        output_root=tmp_path / "timeout",
    )
    assert timed_out.status is ProjectCoverageStatus.TIMED_OUT
    assert timed_out.failure_category == "FULL_PROJECT_SUITE_TIMEOUT"
    assert timed_out.full_test_file.is_file()
    assert timed_out.report_path.is_file()


def test_equal_percentages_with_different_exact_identities_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    candidate = ProjectTestCandidate(
        project_test_id="module.py::target::scenario-1",
        relative_module_path="module.py",
        module_path="module",
        function_name="target",
        function_start_line=1,
        function_end_line=2,
        source_file=source,
        import_root=tmp_path,
        scenario=_scenario("scenario-1", value=1, expected=1, path_index=1),
        original_order=1,
        precomputed_line_identities=(1,),
        precomputed_branch_identities=(),
        precomputed_execution_success=True,
    )
    scope = ProjectCoverageScopeSummary(
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
        scope_complete=True,
    )
    measurements = iter(
        (
            _ProjectMeasurement(0, 0.1, (("module.py", 1),), (), (("module.py", 1),), ()),
            _ProjectMeasurement(0, 0.1, (("module.py", 2),), (), (("module.py", 2),), ()),
        )
    )
    service = ProjectCoverageService()
    monkeypatch.setattr(service, "_measure", lambda *args, **kwargs: next(measurements))

    result = service.measure_and_minimize(
        candidates=(candidate,),
        scope=scope,
        output_root=tmp_path / "output",
    )

    assert result.full_line_coverage_percent == result.minimized_line_coverage_percent
    assert result.coverage_preserved is False
    assert result.status is ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
    assert result.failure_category == "FINAL_EXACT_COVERAGE_MISMATCH"
