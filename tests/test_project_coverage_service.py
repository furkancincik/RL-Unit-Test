from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from analyzer.safe_custom_object import SafeObjectConstructionBlueprint
from analyzer.safe_method_summary import analyze_safe_method_summaries
from generator.safe_method_setup_plan import (
    SafeObjectSetupPlan,
    SafeObjectSlot,
    SafeSetupCall,
    SafeTargetBinding,
    SafeTypedArgumentBinding,
)
from generator.scenario_generator import Scenario
from models.coverage_progress import (
    CoverageProgressSnapshot,
    CoverageStopReason,
)
from models.project_coverage_result import (
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
    ProjectTestCandidate,
)
from services.project_coverage_service import (
    ProjectCoverageService,
    ProjectExactCoverageGreedySelector,
    ProjectScenarioCoverageSignature,
    _CoverageOptimizationPolicy,
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
    assert selection.greedy_selection_order_ids == ("z", "a")
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
    assert selection.greedy_selection_order_ids == (
        "scenario-3",
        "scenario-2",
        "scenario-4",
    )


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


def test_setup_plan_candidate_does_not_reuse_target_only_precomputed_signature(
    tmp_path: Path,
) -> None:
    class SetupPlanStub:
        execution_fingerprint = "setup-state"

    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    scenario = replace(
        _scenario("scenario-1", value=1, expected=1, path_index=1),
        setup_plan=SetupPlanStub(),
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
        precomputed_line_identities=(1,),
        precomputed_branch_identities=(),
        precomputed_execution_success=True,
    )

    assert ProjectCoverageService._precomputed_signature(candidate) is None


def test_setup_plan_candidate_uses_real_project_contribution_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SetupPlanStub:
        execution_fingerprint = "setup-state"

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
        scenario=replace(
            _scenario("scenario-1", value=1, expected=1, path_index=1),
            setup_plan=SetupPlanStub(),
        ),
        original_order=1,
        # Bu target-only imza setup satırlarını temsil etmez ve kullanılmamalıdır.
        precomputed_line_identities=(),
        precomputed_branch_identities=(),
        precomputed_execution_success=True,
    )
    measurement = _ProjectMeasurement(
        test_exit_code=0,
        duration_seconds=0.01,
        executable_lines=(("module.py", 1),),
        executable_branches=(),
        covered_lines=(("module.py", 1),),
        covered_branches=(),
    )
    measured: list[tuple[ProjectTestCandidate, ...]] = []
    service = ProjectCoverageService()

    def write_suite(candidates, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_generated():\n    assert True\n", encoding="utf-8")

    def measure(candidates, test_file, timeout, *, scope_candidates=None):
        measured.append(candidates)
        return measurement

    monkeypatch.setattr(service, "_write_suite", write_suite)
    monkeypatch.setattr(service, "_measure", measure)

    result = service.measure_and_minimize(
        candidates=(candidate,),
        scope=_scope(),
        output_root=tmp_path / "output",
    )

    # full pool + DQM-order cumulative + isolated contribution + final verify
    assert len(measured) == 4
    assert measured[1] == (candidate,)
    assert result.coverage_preserved is True


def test_real_setup_aware_project_coverage_preserves_constructor_and_helper_identities(
    tmp_path: Path,
) -> None:
    module = tmp_path / "setup_subject.py"
    module.write_text(
        "class Vessel:\n"
        "    def __init__(self):\n"
        "        self.total = 0\n\n"
        "    def load(self, amount: int) -> None:\n"
        "        self.total += amount\n\n"
        "class Parcel:\n"
        "    def __init__(self, weight: int):\n"
        "        self.weight = weight\n\n"
        "    def normalize(self, delta: int) -> None:\n"
        "        self.weight += delta\n\n"
        "def inspect(subject, item):\n"
        "    if subject.total > item.weight:\n"
        "        return 'larger'\n"
        "    return 'other'\n",
        encoding="utf-8",
    )
    summaries = analyze_safe_method_summaries(
        ast.parse(module.read_text(encoding="utf-8")),
        module_identity="setup_subject",
    )

    def plan(*, amount: int, weight: int, delta: int) -> SafeObjectSetupPlan:
        return SafeObjectSetupPlan(
            module_identity="setup_subject",
            target_identity="inspect",
            object_slots=(
                SafeObjectSlot(
                    slot_id="root",
                    blueprint=SafeObjectConstructionBlueprint(
                        module_identity="setup_subject",
                        class_name="Vessel",
                        class_fingerprint=(
                            summaries.summary_for("Vessel.load")
                            .receiver.class_fingerprint
                        ),
                    ),
                ),
                SafeObjectSlot(
                    slot_id="nested",
                    blueprint=SafeObjectConstructionBlueprint(
                        module_identity="setup_subject",
                        class_name="Parcel",
                        constructor_arguments=(("weight", weight),),
                        class_fingerprint=(
                            summaries.summary_for("Parcel.normalize")
                            .receiver.class_fingerprint
                        ),
                    ),
                    relation_depth=1,
                    owner_slot_id="root",
                ),
            ),
            setup_calls=(
                SafeSetupCall(
                    receiver_slot_id="root",
                    method_summary=summaries.summary_for("Vessel.load"),
                    arguments=(
                        SafeTypedArgumentBinding(
                            parameter_name="amount",
                            type_name="int",
                            value=amount,
                        ),
                    ),
                ),
                SafeSetupCall(
                    receiver_slot_id="nested",
                    method_summary=summaries.summary_for("Parcel.normalize"),
                    arguments=(
                        SafeTypedArgumentBinding(
                            parameter_name="delta",
                            type_name="int",
                            value=delta,
                        ),
                    ),
                ),
            ),
            target_bindings=(
                SafeTargetBinding(
                    parameter_name="subject",
                    object_slot_id="root",
                ),
                SafeTargetBinding(
                    parameter_name="item",
                    object_slot_id="nested",
                ),
            ),
        )

    start, end = _function_range(module, "inspect")
    scenarios = (
        replace(
            _scenario("setup-positive", value=0, expected="larger", path_index=1),
            keyword_arguments=(),
            setup_plan=plan(amount=3, weight=1, delta=1),
        ),
        replace(
            _scenario("setup-negative", value=0, expected="other", path_index=2),
            keyword_arguments=(),
            setup_plan=plan(amount=-2, weight=5, delta=-1),
        ),
    )
    candidates = tuple(
        ProjectTestCandidate(
            project_test_id=f"setup_subject.py::inspect::{scenario.scenario_id}",
            relative_module_path="setup_subject.py",
            module_path="setup_subject",
            function_name="inspect",
            function_start_line=start,
            function_end_line=end,
            source_file=module,
            import_root=tmp_path,
            scenario=scenario,
            original_order=index,
        )
        for index, scenario in enumerate(scenarios, start=1)
    )

    result = ProjectCoverageService().measure_and_minimize(
        candidates=candidates,
        scope=_scope(),
        output_root=tmp_path / "output",
        timeout_seconds=30.0,
    )

    covered_lines = {
        identity.line_number for identity in result.target_line_identities
    }
    assert {2, 3, 6, 9, 10, 13}.issubset(covered_lines)
    assert result.target_branch_identities
    assert result.status is ProjectCoverageStatus.COMPLETED
    assert result.coverage_preserved is True
    assert result.target_line_identities == result.minimized_covered_line_identities
    assert result.target_branch_identities == result.minimized_covered_branch_identities


def test_project_cache_key_includes_internal_setup_execution_identity(
    tmp_path: Path,
) -> None:
    class SetupPlanStub:
        def __init__(self, fingerprint: str) -> None:
            self.execution_fingerprint = fingerprint

    source = tmp_path / "module.py"
    source.write_text("def target(value):\n    return value\n", encoding="utf-8")
    baseline = _scenario("same-public-id", value=1, expected=1, path_index=1)

    def candidate(fingerprint: str) -> ProjectTestCandidate:
        return ProjectTestCandidate(
            project_test_id="module.py::target::same-public-id",
            relative_module_path="module.py",
            module_path="module",
            function_name="target",
            function_start_line=1,
            function_end_line=2,
            source_file=source,
            import_root=tmp_path,
            scenario=replace(
                baseline,
                setup_plan=SetupPlanStub(fingerprint),
            ),
            original_order=1,
        )

    assert ProjectCoverageService._cache_key(
        candidate("first"), 10.0
    ) != ProjectCoverageService._cache_key(candidate("second"), 10.0)


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


def test_final_exact_mismatch_preserves_last_verified_progress_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "def target(value):\n    return value\n",
        encoding="utf-8",
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
        scenario=_scenario(
            "scenario-1",
            value=1,
            expected=1,
            path_index=1,
        ),
        original_order=1,
        precomputed_line_identities=(1, 2),
        precomputed_branch_identities=(),
        precomputed_execution_success=True,
    )
    executable = (("module.py", 1), ("module.py", 2))
    measurements = iter(
        (
            _ProjectMeasurement(0, 0.1, executable, (), executable, ()),
            _ProjectMeasurement(0, 0.1, executable, (), executable, ()),
            _ProjectMeasurement(
                0,
                0.1,
                executable,
                (),
                (("module.py", 1),),
                (),
            ),
        )
    )
    service = ProjectCoverageService()
    monkeypatch.setattr(
        service,
        "_measure",
        lambda *args, **kwargs: next(measurements),
    )
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=(candidate,),
        scope=_scope(),
        output_root=tmp_path / "mismatch-progress",
        coverage_progress_callback=snapshots.append,
    )

    assert result.status is ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
    assert result.failure_category == "FINAL_EXACT_COVERAGE_MISMATCH"
    assert [item.covered_lines for item in snapshots] == [2, 2]
    assert snapshots[-1].stop_reason is CoverageStopReason.FAILED


def test_early_target_non_additive_fallback_publishes_honest_terminal_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _progress_candidates(tmp_path)
    executable = (("module.py", 1),)
    full = _ProjectMeasurement(0, 0.1, executable, (), executable, ())
    empty = _ProjectMeasurement(0, 0.1, executable, (), (), ())
    measurements = iter((full, full, full, empty, full))
    service = ProjectCoverageService()
    monkeypatch.setattr(
        service,
        "_measure",
        lambda *args, **kwargs: next(measurements),
    )
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=candidates,
        scope=_scope(),
        output_root=tmp_path / "non-additive-progress",
        coverage_progress_callback=snapshots.append,
    )

    assert result.status is ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
    assert result.failure_category == "NON_ADDITIVE_COVERAGE"
    assert result.final_selected_count == len(candidates)
    assert snapshots[-1].validated_count == len(candidates)
    assert snapshots[-1].effective_test_count == len(candidates)
    assert snapshots[-1].stop_reason is CoverageStopReason.TARGET_REACHED


class _DeterministicProgressCoverageService(ProjectCoverageService):
    def __init__(
        self,
        contributions: dict[str, tuple[int, ...]],
        *,
        policy: _CoverageOptimizationPolicy | None = None,
    ) -> None:
        super().__init__(_coverage_optimization_policy=policy)
        self._contributions = contributions
        self.measurement_inputs: list[
            tuple[tuple[str, ...], tuple[str, ...]]
        ] = []

    def _measure(
        self,
        candidates,
        test_file,
        timeout,
        *,
        scope_candidates=None,
    ) -> _ProjectMeasurement:
        scope_values = scope_candidates or candidates
        identifiers = tuple(item.project_test_id for item in candidates)
        scope_identifiers = tuple(
            item.project_test_id for item in scope_values
        )
        self.measurement_inputs.append((identifiers, scope_identifiers))
        covered = tuple(
            sorted(
                {
                    ("module.py", line)
                    for identifier in identifiers
                    for line in self._contributions[identifier]
                }
            )
        )
        return _ProjectMeasurement(
            test_exit_code=0,
            duration_seconds=0.01,
            executable_lines=tuple(
                ("module.py", line) for line in (1, 2, 3)
            ),
            executable_branches=(),
            covered_lines=covered,
            covered_branches=(),
        )


def _progress_candidates(tmp_path: Path) -> tuple[ProjectTestCandidate, ...]:
    source = tmp_path / "module.py"
    source.write_text(
        "def target(value):\n    return value\n",
        encoding="utf-8",
    )
    values: list[ProjectTestCandidate] = []
    for order, identifier in enumerate(
        ("high", "zero", "later"),
        start=1,
    ):
        scenario = _scenario(
            identifier,
            value=order,
            expected=order,
            path_index=order,
        )
        values.append(
            ProjectTestCandidate(
                project_test_id=identifier,
                relative_module_path="module.py",
                module_path="module",
                function_name="target",
                function_start_line=1,
                function_end_line=2,
                source_file=source,
                import_root=tmp_path,
                scenario=scenario,
                original_order=order,
            )
        )
    return tuple(reversed(values))


def test_strategy_evaluation_pool_is_measured_once_in_common_dqm_order(
    tmp_path: Path,
) -> None:
    service = _DeterministicProgressCoverageService(
        {
            "high": (1,),
            "zero": (),
            "later": (2,),
        }
    )

    pool = service.measure_strategy_evaluation_pool(
        candidates=_progress_candidates(tmp_path),
        output_root=tmp_path / "strategy-output",
    )

    assert pool.candidate_ids == ("high", "zero", "later")
    assert pool.executable_line_identities == (
        ("module.py", 1),
        ("module.py", 2),
        ("module.py", 3),
    )
    assert pool.target_line_identities == (("module.py", 1), ("module.py", 2))
    assert service.measurement_inputs == [
        (("high", "zero", "later"), ("high", "zero", "later")),
        (("high",), ("high", "zero", "later")),
        (("zero",), ("high", "zero", "later")),
        (("later",), ("high", "zero", "later")),
    ]


def test_online_progress_uses_dqm_order_fixed_scope_and_exact_marginal_gain(
    tmp_path: Path,
) -> None:
    service = _DeterministicProgressCoverageService(
        {
            "high": (1,),
            "zero": (),
            "later": (2,),
        }
    )
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=_progress_candidates(tmp_path),
        scope=_scope(),
        output_root=tmp_path / "output",
        coverage_progress_callback=snapshots.append,
    )

    assert result.coverage_preserved is True
    assert [item.revision for item in snapshots] == [1, 2, 3, 4]
    assert [item.validated_count for item in snapshots[:-1]] == [1, 2, 3]
    assert [item.covered_lines for item in snapshots[:-1]] == [1, 1, 2]
    assert [item.last_new_line_count for item in snapshots[:-1]] == [1, 0, 1]
    assert [item.effective_test_count for item in snapshots[:-1]] == [1, 1, 2]
    assert snapshots[-1].effective_test_count == result.final_selected_count
    assert snapshots[-1].stop_reason is CoverageStopReason.TARGET_REACHED
    assert all(item.total_lines == 3 for item in snapshots)
    assert all(item.metric == "LINE" for item in snapshots)
    assert all(
        scope_ids == ("high", "zero", "later")
        for _, scope_ids in service.measurement_inputs
    )


@pytest.mark.parametrize(
    "unsafe_measurement",
    (
        _ProjectMeasurement(
            0,
            0.01,
            (("module.py", 1), ("module.py", 2), ("module.py", 3)),
            (),
            (("module.py", 1),),
            (),
        ),
        _ProjectMeasurement(
            0,
            0.01,
            (("module.py", 1), ("module.py", 2)),
            (),
            (("module.py", 1), ("module.py", 2)),
            (),
        ),
    ),
)
def test_online_progress_rejects_shrink_or_denominator_drift_before_publish(
    tmp_path: Path,
    unsafe_measurement: _ProjectMeasurement,
) -> None:
    full = _ProjectMeasurement(
        0,
        0.01,
        (("module.py", 1), ("module.py", 2), ("module.py", 3)),
        (),
        (("module.py", 1), ("module.py", 2), ("module.py", 3)),
        (),
    )
    first = _ProjectMeasurement(
        0,
        0.01,
        full.executable_lines,
        (),
        (("module.py", 1), ("module.py", 2)),
        (),
    )
    measurements = iter((full, first, unsafe_measurement))
    service = ProjectCoverageService()
    service._measure = lambda *args, **kwargs: next(measurements)  # type: ignore[method-assign]
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=_progress_candidates(tmp_path),
        scope=_scope(),
        output_root=tmp_path / "unsafe",
        coverage_progress_callback=snapshots.append,
    )

    assert result.status is ProjectCoverageStatus.FAILED
    assert [item.covered_lines for item in snapshots] == [2, 2]
    assert snapshots[0].stop_reason is None
    assert snapshots[-1].stop_reason is CoverageStopReason.FAILED


def test_online_progress_deadline_preserves_last_verified_prefix(
    tmp_path: Path,
) -> None:
    candidates = _progress_candidates(tmp_path)
    executable = tuple(("module.py", line) for line in (1, 2, 3))
    full = _ProjectMeasurement(
        0,
        0.01,
        executable,
        (),
        executable,
        (),
    )
    first_prefix = _ProjectMeasurement(
        0,
        0.01,
        executable,
        (),
        (("module.py", 1),),
        (),
    )
    measured: list[tuple[str, ...]] = []

    def measure(values, *args, **kwargs):
        measured.append(tuple(item.project_test_id for item in values))
        if len(measured) == 1:
            return full
        if len(measured) == 2:
            return first_prefix
        raise CoverageExecutionTimeoutError("bounded progress timeout")

    service = ProjectCoverageService()
    service._measure = measure  # type: ignore[method-assign]
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=candidates,
        scope=_scope(),
        output_root=tmp_path / "deadline-prefix",
        coverage_progress_callback=snapshots.append,
    )

    assert measured == [
        ("high", "zero", "later"),
        ("high",),
        ("high", "zero"),
    ]
    assert result.status is ProjectCoverageStatus.TIMED_OUT
    assert result.failure_category == "PROJECT_COVERAGE_OPTIMIZATION_TIMEOUT"
    assert result.selected_project_test_ids == ("high",)
    assert result.removed_project_test_ids == ()
    assert result.final_selected_count == 1
    assert result.minimized_pytest_exit_code == 0
    assert tuple(
        item.to_tuple() for item in result.minimized_covered_line_identities
    ) == (("module.py", 1),)
    assert result.coverage_preserved is False
    assert result.minimized_test_file.is_file()
    assert "coverage_optimization/0001" in result.minimized_test_file.as_posix()
    assert result.minimized_test_file in result.artifact_paths
    assert snapshots[-1].stop_reason is CoverageStopReason.DEADLINE_REACHED
    assert snapshots[-1].validated_count == 1
    assert snapshots[-1].effective_test_count == result.final_selected_count
    assert snapshots[-1].covered_lines == 1


def test_online_progress_test_limit_preserves_prefix_without_resuming_candidates(
    tmp_path: Path,
) -> None:
    service = _DeterministicProgressCoverageService(
        {
            "high": (1,),
            "zero": (),
            "later": (2,),
        },
        policy=_CoverageOptimizationPolicy(test_limit=1),
    )
    snapshots: list[CoverageProgressSnapshot] = []

    result = service.measure_and_minimize(
        candidates=_progress_candidates(tmp_path),
        scope=_scope(),
        output_root=tmp_path / "test-limit-prefix",
        coverage_progress_callback=snapshots.append,
    )

    assert service.measurement_inputs == [
        (
            ("high", "zero", "later"),
            ("high", "zero", "later"),
        ),
        (("high",), ("high", "zero", "later")),
    ]
    assert result.status is ProjectCoverageStatus.PARTIAL
    assert result.failure_category == "PROJECT_COVERAGE_TEST_LIMIT_REACHED"
    assert result.selected_project_test_ids == ("high",)
    assert result.removed_project_test_ids == ()
    assert result.final_selected_count == 1
    assert result.minimized_pytest_exit_code == 0
    assert result.coverage_preserved is False
    assert result.minimized_test_file.is_file()
    assert "coverage_optimization/0001" in result.minimized_test_file.as_posix()
    assert snapshots[-1].stop_reason is CoverageStopReason.TEST_LIMIT_REACHED
    assert snapshots[-1].validated_count == 1
    assert snapshots[-1].effective_test_count == result.final_selected_count


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_per_measurement_timeout_resets_for_each_coverage_execution(
    tmp_path: Path,
) -> None:
    clock = _AdvancingClock()
    service = _DeterministicProgressCoverageService(
        {
            "high": (1,),
            "zero": (),
            "later": (2,),
        }
    )
    service._clock = clock
    original_measure = service._measure
    observed_timeouts: list[float] = []

    def measure(*args, **kwargs):
        observed_timeouts.append(float(args[2]))
        result = original_measure(*args, **kwargs)
        clock.advance(2.0)
        return result

    service._measure = measure  # type: ignore[method-assign]

    result = service.measure_and_minimize(
        candidates=_progress_candidates(tmp_path),
        scope=_scope(),
        output_root=tmp_path / "per-measurement-timeout",
        timeout_seconds=7.0,
    )

    assert result.status is ProjectCoverageStatus.COMPLETED
    assert len(observed_timeouts) > 2
    assert observed_timeouts == [7.0] * len(observed_timeouts)


def test_explicit_overall_timeout_clamps_and_stops_sequential_measurements(
    tmp_path: Path,
) -> None:
    clock = _AdvancingClock()
    service = _DeterministicProgressCoverageService(
        {
            "high": (1,),
            "zero": (),
            "later": (2,),
        }
    )
    service._clock = clock
    original_measure = service._measure
    observed_timeouts: list[float] = []

    def measure(*args, **kwargs):
        observed_timeouts.append(float(args[2]))
        result = original_measure(*args, **kwargs)
        clock.advance(1.0)
        return result

    service._measure = measure  # type: ignore[method-assign]

    result = service.measure_and_minimize(
        candidates=_progress_candidates(tmp_path),
        scope=_scope(),
        output_root=tmp_path / "overall-timeout",
        timeout_seconds=10.0,
        overall_timeout_seconds=2.5,
    )

    assert observed_timeouts == [2.5, 1.5, 0.5]
    assert result.status is ProjectCoverageStatus.TIMED_OUT
    assert result.failure_category == "PROJECT_COVERAGE_OPTIMIZATION_TIMEOUT"
    assert result.selected_project_test_ids == ("high", "zero")
    assert result.coverage_preserved is False


def test_online_progress_stop_policy_requires_explicit_plateau_proof() -> None:
    default = _CoverageOptimizationPolicy()
    base = dict(
        covered_lines=frozenset({("module.py", 1)}),
        target_lines=frozenset(
            {("module.py", 1), ("module.py", 2)}
        ),
        covered_branches=frozenset(),
        target_branches=frozenset(),
        validated_count=1,
        candidate_count=3,
        remaining_candidates=(object(), object()),
    )

    assert default.stop_reason(**base) is None
    assert default.stop_reason(
        **{**base, "covered_lines": base["target_lines"]}
    ) is CoverageStopReason.TARGET_REACHED
    assert default.stop_reason(
        **{**base, "validated_count": 3, "remaining_candidates": ()}
    ) is CoverageStopReason.CANDIDATES_EXHAUSTED
    assert _CoverageOptimizationPolicy(
        plateau_patience=2,
        remaining_identity_proof=lambda *_: True
    ).stop_reason(
        **base,
        plateau_count=2,
    ) is CoverageStopReason.PROVEN_PLATEAU
    assert _CoverageOptimizationPolicy(
        plateau_patience=2,
        remaining_identity_proof=lambda *_: True,
    ).stop_reason(
        **base,
        plateau_count=1,
    ) is None
    assert _CoverageOptimizationPolicy(test_limit=1).stop_reason(
        **base
    ) is CoverageStopReason.TEST_LIMIT_REACHED
    assert default.stop_reason(
        **base,
        deadline_reached=True,
    ) is CoverageStopReason.DEADLINE_REACHED
    assert _CoverageOptimizationPolicy(
        cancellation_requested=lambda: True
    ).stop_reason(**base) is CoverageStopReason.CANCELLED
    assert default.stop_reason(
        **base,
        failed=True,
    ) is CoverageStopReason.FAILED


def test_online_progress_stop_policy_uses_documented_priority_order() -> None:
    policy = _CoverageOptimizationPolicy(
        test_limit=1,
        plateau_patience=1,
        remaining_identity_proof=lambda *_: True,
        cancellation_requested=lambda: True,
    )
    common = dict(
        covered_lines=frozenset({("module.py", 1)}),
        target_lines=frozenset({("module.py", 1)}),
        covered_branches=frozenset(),
        target_branches=frozenset(),
        validated_count=1,
        candidate_count=1,
        remaining_candidates=(),
        plateau_count=1,
        deadline_reached=True,
        failed=True,
    )

    assert policy.stop_reason(**common) is CoverageStopReason.TARGET_REACHED
    assert policy.stop_reason(
        **{
            **common,
            "target_lines": frozenset(
                {("module.py", 1), ("module.py", 2)}
            ),
        }
    ) is CoverageStopReason.CANDIDATES_EXHAUSTED
