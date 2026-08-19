from __future__ import annotations

import ast
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult, FunctionCoverageResult
from models.scenario_minimization_result import (
    CoverageMeasurementStatus,
    ScenarioCoverageSignature,
    ScenarioMinimizationStatus,
)
from services.scenario_coverage_minimization_service import (
    ExactCoverageGreedySelector,
    ScenarioCoverageMinimizationService,
)
from services.coverage_service import CoverageExecutionTimeoutError
from services.scenario_suite_coverage_service import ScenarioSuiteCoverageResult


def _scenario(identifier: str, index: int, value: int = 0) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        name=f"Scenario {identifier}",
        path_index=index,
        priority_rank=index,
        priority_level="Medium",
        dqm_score=50.0,
        node_ids=(1, 2),
        edge_labels=(None,),
        contains_loop=False,
        contains_exception=False,
        description="Coverage contribution fixture.",
        keyword_arguments=(("value", value),),
        expected_result="negative" if value < 0 else "nonnegative",
        expected_exception=None,
    )


def _coverage(
    tmp_path: Path,
    *,
    lines: tuple[int, ...],
    branches: tuple[tuple[int, int], ...],
    exit_code: int = 0,
) -> FunctionCoverageResult:
    all_lines = (1, 2, 3, 4, 5)
    all_branches = ((2, 3), (2, 5))
    missing_lines = tuple(sorted(set(all_lines) - set(lines)))
    missing_branches = tuple(sorted(set(all_branches) - set(branches)))
    file_result = CoverageResult(
        source_file=tmp_path / "target.py",
        test_file=tmp_path / "test_target.py",
        line_coverage_percent=len(lines) / len(all_lines) * 100.0,
        branch_coverage_percent=len(branches) / len(all_branches) * 100.0,
        covered_line_count=len(lines),
        missing_line_count=len(missing_lines),
        total_line_count=len(all_lines),
        covered_branch_count=len(branches),
        missing_branch_count=len(missing_branches),
        total_branch_count=len(all_branches),
        test_exit_code=exit_code,
        duration_seconds=0.01,
    )
    return FunctionCoverageResult(
        source_file=file_result.source_file,
        test_file=file_result.test_file,
        function_name="classify",
        start_line=1,
        end_line=5,
        line_coverage_percent=file_result.line_coverage_percent,
        branch_coverage_percent=file_result.branch_coverage_percent,
        covered_lines=lines,
        missing_lines=missing_lines,
        covered_branch_count=len(branches),
        missing_branch_count=len(missing_branches),
        covered_branches=branches,
        missing_branches=missing_branches,
        test_exit_code=exit_code,
        duration_seconds=0.01,
        file_coverage=file_result,
    )


def _signature(
    identifier: str,
    index: int,
    lines: tuple[int, ...],
    branches: tuple[tuple[int, int], ...],
) -> ScenarioCoverageSignature:
    return ScenarioCoverageSignature(
        scenario_id=identifier,
        original_scenario_index=index,
        function_name="classify",
        covered_line_identities=lines,
        covered_branch_identities=branches,
        line_contribution_count=len(lines),
        branch_contribution_count=len(branches),
        execution_success=True,
        test_exit_code=0,
        duration_seconds=0.01,
        measurement_status=CoverageMeasurementStatus.SUCCESS,
    )


def test_signature_is_immutable_and_json_safe() -> None:
    signature = _signature("scenario_001", 1, (1, 2), ((2, -1),))

    with pytest.raises(FrozenInstanceError):
        signature.scenario_id = "changed"  # type: ignore[misc]

    value = signature.to_dict()
    assert value["covered_line_identities"] == [1, 2]
    assert value["covered_branch_identities"] == [[2, -1]]
    assert "keyword_arguments" not in json.dumps(value)


def test_signature_rejects_inconsistent_contribution_counts() -> None:
    with pytest.raises(ValueError, match="line_contribution_count"):
        ScenarioCoverageSignature(
            scenario_id="scenario_001",
            original_scenario_index=1,
            function_name="classify",
            covered_line_identities=(1,),
            covered_branch_identities=(),
            line_contribution_count=2,
            branch_contribution_count=0,
            execution_success=True,
            test_exit_code=0,
            duration_seconds=0.1,
            measurement_status=CoverageMeasurementStatus.SUCCESS,
        )


def test_exact_selector_preserves_line_and_branch_identities() -> None:
    signatures = (
        _signature("first", 1, (1, 2), ((2, 3),)),
        _signature("second", 2, (1, 2), ((2, 5),)),
        _signature("duplicate", 3, (1, 2), ((2, 3),)),
    )

    selection = ExactCoverageGreedySelector.select(
        signatures=signatures,
        target_lines=(1, 2),
        target_branches=((2, 3), (2, 5)),
    )

    assert selection.selected_scenario_ids == ("first", "second")
    assert selection.initial_selected_count == 2
    assert selection.redundancy_removed_count == 0


def test_exact_selector_uses_deterministic_tie_break() -> None:
    signatures = (
        _signature("later-id", 2, (1,), ()),
        _signature("earlier-index", 1, (1,), ()),
    )

    selection = ExactCoverageGreedySelector.select(
        signatures=signatures,
        target_lines=(1,),
        target_branches=(),
    )

    assert selection.selected_scenario_ids == ("earlier-index",)


def test_backward_elimination_removes_redundant_selection() -> None:
    signatures = (
        _signature("covers-both", 3, (1, 2), ()),
        _signature("first", 1, (1,), ()),
        _signature("second", 2, (2,), ()),
    )

    selected, removed = ExactCoverageGreedySelector.eliminate_redundancy(
        selected=signatures,
        target_lines=(1, 2),
        target_branches=(),
    )

    assert tuple(item.scenario_id for item in selected) == ("covers-both",)
    assert tuple(item.scenario_id for item in removed) == ("second", "first")


class _FakeSuiteService:
    def __init__(
        self,
        tmp_path: Path,
        coverage_by_ids: dict[tuple[str, ...], FunctionCoverageResult],
    ) -> None:
        self.tmp_path = tmp_path
        self.coverage_by_ids = coverage_by_ids
        self.calls: list[tuple[str, ...]] = []

    def measure_scenarios(self, **kwargs: object) -> ScenarioSuiteCoverageResult:
        scenarios = kwargs["scenarios"]
        assert isinstance(scenarios, tuple)
        ids = tuple(item.scenario_id for item in scenarios)
        self.calls.append(ids)
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        test_name = str(kwargs.get("test_file_name") or "test_suite.py")
        test_file = output / test_name
        test_file.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
        return ScenarioSuiteCoverageResult(
            scenarios=scenarios,
            test_file=test_file,
            coverage=self.coverage_by_ids[ids],
        )


def test_service_measures_full_pool_once_and_reduces_duplicate_signature(
    tmp_path: Path,
) -> None:
    first = _scenario("first", 1, -1)
    second = _scenario("second", 2, 1)
    duplicate = _scenario("duplicate", 3, 2)
    full = _coverage(tmp_path, lines=(1, 2, 3, 4, 5), branches=((2, 3), (2, 5)))
    negative = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    positive = _coverage(tmp_path, lines=(1, 2, 4, 5), branches=((2, 5),))
    fake = _FakeSuiteService(
        tmp_path,
        {
            ("first", "second", "duplicate"): full,
            ("first",): negative,
            ("second",): positive,
            ("duplicate",): positive,
            ("first", "second"): full,
        },
    )
    service = ScenarioCoverageMinimizationService(suite_service=fake)

    result = service.minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(first, second, duplicate),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.COMPLETED
    assert result.coverage_preserved is True
    assert result.final_selected_scenario_ids == ("first", "second")
    assert result.reduction_count == 1
    assert result.reduction_percentage == pytest.approx(100.0 / 3.0)
    assert result.globally_minimal is False
    assert fake.calls.count(("first", "second", "duplicate")) == 1


def test_service_reports_non_additive_final_verification_failure(
    tmp_path: Path,
) -> None:
    first = _scenario("first", 1, -1)
    second = _scenario("second", 2, 1)
    duplicate = _scenario("duplicate", 3, 2)
    full = _coverage(tmp_path, lines=(1, 2, 3, 4, 5), branches=((2, 3), (2, 5)))
    negative = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    positive = _coverage(tmp_path, lines=(1, 2, 4, 5), branches=((2, 5),))
    non_additive = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    fake = _FakeSuiteService(
        tmp_path,
        {
            ("first", "second", "duplicate"): full,
            ("first",): negative,
            ("second",): positive,
            ("duplicate",): positive,
            ("first", "second"): non_additive,
        },
    )

    result = ScenarioCoverageMinimizationService(
        suite_service=fake
    ).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(first, second, duplicate),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.NON_ADDITIVE_COVERAGE
    assert result.coverage_preserved is False


def test_service_returns_controlled_full_pool_failure(tmp_path: Path) -> None:
    scenario = _scenario("first", 1)
    failed = _coverage(tmp_path, lines=(), branches=(), exit_code=1)
    fake = _FakeSuiteService(tmp_path, {("first",): failed})

    result = ScenarioCoverageMinimizationService(
        suite_service=fake
    ).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(scenario,),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.FULL_POOL_FAILED
    assert result.coverage_preserved is False
    assert result.failure_category == "FULL_POOL_EXECUTION_FAILED"


def test_real_minimized_suite_preserves_exact_coverage_and_is_valid_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "target.py"
    source_file.write_text(
        "def classify(value: int) -> str:\n"
        "    if value < 0:\n"
        "        return 'negative'\n"
        "    return 'nonnegative'\n",
        encoding="utf-8",
    )
    existing_python_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path)
        if not existing_python_path
        else f"{tmp_path}{os.pathsep}{existing_python_path}",
    )
    scenarios = (
        _scenario("negative", 1, -1),
        _scenario("positive", 2, 1),
        _scenario("duplicate", 3, 2),
    )

    result = ScenarioCoverageMinimizationService().minimize(
        source_file=source_file,
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=4,
        scenarios=scenarios,
        output_root=tmp_path / "output",
        timeout_seconds=20.0,
    )

    assert result.coverage_preserved is True
    assert result.final_selected_count == 2
    assert result.target_line_identities == result.final_verified_line_identities
    assert result.target_branch_identities == result.final_verified_branch_identities
    generated = result.artifact_paths[0]
    output_root = (tmp_path / "output").resolve()
    assert all(path.resolve().is_relative_to(output_root) for path in result.artifact_paths)
    ast.parse(generated.read_text(encoding="utf-8"))
    report = json.loads(result.artifact_paths[1].read_text(encoding="utf-8"))
    assert report["globally_minimal"] is False
    assert "keyword_arguments" not in json.dumps(report)


def test_service_reports_honest_zero_reduction(tmp_path: Path) -> None:
    first = _scenario("first", 1, -1)
    second = _scenario("second", 2, 1)
    full = _coverage(tmp_path, lines=(1, 2, 3, 4, 5), branches=((2, 3), (2, 5)))
    negative = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    positive = _coverage(tmp_path, lines=(1, 2, 4, 5), branches=((2, 5),))
    fake = _FakeSuiteService(
        tmp_path,
        {
            ("first", "second"): full,
            ("first",): negative,
            ("second",): positive,
        },
    )

    result = ScenarioCoverageMinimizationService(
        suite_service=fake
    ).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(first, second),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.NO_REDUCTION
    assert result.reduction_count == 0
    assert result.reduction_percentage == 0.0


def test_service_reports_empty_pool_without_running_coverage(tmp_path: Path) -> None:
    fake = _FakeSuiteService(tmp_path, {})

    result = ScenarioCoverageMinimizationService(suite_service=fake).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.EMPTY_POOL
    assert result.target_line_percentage is None
    assert fake.calls == []


def test_service_marks_isolated_failure_as_interaction_dependent(
    tmp_path: Path,
) -> None:
    first = _scenario("first", 1)
    second = _scenario("second", 2)
    full = _coverage(tmp_path, lines=(1, 2, 3, 4, 5), branches=((2, 3), (2, 5)))
    first_only = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    second_failed = _coverage(tmp_path, lines=(1, 2, 4, 5), branches=((2, 5),), exit_code=1)
    fake = _FakeSuiteService(
        tmp_path,
        {
            ("first", "second"): full,
            ("first",): first_only,
            ("second",): second_failed,
        },
    )

    result = ScenarioCoverageMinimizationService(suite_service=fake).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(first, second),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.TARGET_NOT_COVERED
    assert result.contributions[1].measurement_status is CoverageMeasurementStatus.INTERACTION_DEPENDENT
    assert result.contributions[1].failure_category == "NON_ISOLATED_SCENARIO"


class _TimeoutSuiteService(_FakeSuiteService):
    def measure_scenarios(self, **kwargs: object) -> ScenarioSuiteCoverageResult:
        scenarios = kwargs["scenarios"]
        assert isinstance(scenarios, tuple)
        ids = tuple(item.scenario_id for item in scenarios)
        if ids == ("second",):
            self.calls.append(ids)
            raise CoverageExecutionTimeoutError("controlled contribution timeout")
        return super().measure_scenarios(**kwargs)


def test_timeout_preserves_partial_contributions_and_atomic_report(
    tmp_path: Path,
) -> None:
    first = _scenario("first", 1)
    second = _scenario("second", 2)
    full = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    first_only = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    fake = _TimeoutSuiteService(
        tmp_path,
        {("first", "second"): full, ("first",): first_only},
    )

    result = ScenarioCoverageMinimizationService(suite_service=fake).minimize(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="classify",
        function_start_line=1,
        function_end_line=5,
        scenarios=(first, second),
        output_root=tmp_path / "output",
    )

    assert result.status is ScenarioMinimizationStatus.TIMED_OUT
    assert result.measured_scenario_count == 2
    assert result.contributions[0].measurement_status is CoverageMeasurementStatus.SUCCESS
    assert result.contributions[1].measurement_status is CoverageMeasurementStatus.TIMED_OUT
    report = tmp_path / "output" / "greedy_minimized" / "minimization_report.json"
    assert json.loads(report.read_text(encoding="utf-8"))["run_status"] == "TIMED_OUT"
    assert not tuple(report.parent.glob(".*.tmp"))


def test_two_runs_keep_measurement_state_isolated(tmp_path: Path) -> None:
    scenario = _scenario("only", 1)
    full = _coverage(tmp_path, lines=(1, 2, 3), branches=((2, 3),))
    fake = _FakeSuiteService(
        tmp_path,
        {("only",): full},
    )
    service = ScenarioCoverageMinimizationService(suite_service=fake)

    for output_name in ("first-output", "second-output"):
        result = service.minimize(
            source_file=tmp_path / "target.py",
            module_path="target",
            function_name="classify",
            function_start_line=1,
            function_end_line=5,
            scenarios=(scenario,),
            output_root=tmp_path / output_name,
        )
        assert result.status is ScenarioMinimizationStatus.NO_REDUCTION

    assert fake.calls == [("only",), ("only",), ("only",), ("only",), ("only",), ("only",)]


def test_same_percentages_with_different_identities_are_not_equivalent() -> None:
    selection = ExactCoverageGreedySelector.select(
        signatures=(
            _signature("left", 1, (1,), ((2, 3),)),
            _signature("right", 2, (2,), ((2, 5),)),
        ),
        target_lines=(1, 2),
        target_branches=((2, 3), (2, 5)),
    )

    assert selection.selected_scenario_ids == ("left", "right")


def test_zero_contribution_scenario_is_not_selected() -> None:
    selection = ExactCoverageGreedySelector.select(
        signatures=(
            _signature("useful", 1, (1,), ((2, 3),)),
            _signature("zero", 2, (), ()),
        ),
        target_lines=(1,),
        target_branches=((2, 3),),
    )

    assert selection.selected_scenario_ids == ("useful",)


def test_backward_eliminated_suite_is_one_minimal() -> None:
    signatures = (
        _signature("left", 1, (1,), ((2, 3),)),
        _signature("right", 2, (2,), ((2, 5),)),
    )

    kept, removed = ExactCoverageGreedySelector.eliminate_redundancy(
        selected=signatures,
        target_lines=(1, 2),
        target_branches=((2, 3), (2, 5)),
    )

    assert kept == signatures
    assert removed == ()


def test_run_local_cache_key_isolated_by_source_function_and_config(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_source.write_text("pass\n", encoding="utf-8")
    second_source.write_text("pass\n", encoding="utf-8")
    scenario = _scenario("stable", 1)

    common = {
        "start_line": 1,
        "end_line": 5,
        "scenario": scenario,
    }
    baseline = ScenarioCoverageMinimizationService._cache_key(
        source=first_source,
        function_name="classify",
        timeout_seconds=10.0,
        **common,
    )

    assert baseline != ScenarioCoverageMinimizationService._cache_key(
        source=second_source,
        function_name="classify",
        timeout_seconds=10.0,
        **common,
    )
    assert baseline != ScenarioCoverageMinimizationService._cache_key(
        source=first_source,
        function_name="other_function",
        timeout_seconds=10.0,
        **common,
    )
    assert baseline != ScenarioCoverageMinimizationService._cache_key(
        source=first_source,
        function_name="classify",
        timeout_seconds=20.0,
        **common,
    )


def test_service_rejects_duplicate_scenario_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        ScenarioCoverageMinimizationService().minimize(
            source_file=tmp_path / "target.py",
            module_path="target",
            function_name="classify",
            function_start_line=1,
            function_end_line=5,
            scenarios=(_scenario("same", 1), _scenario("same", 2)),
            output_root=tmp_path / "output",
        )


def test_real_stateful_fixture_is_not_reported_as_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "stateful_target.py"
    source_file.write_text(
        "counter = 0\n"
        "def advance() -> str:\n"
        "    global counter\n"
        "    counter += 1\n"
        "    if counter == 2:\n"
        "        return 'second'\n"
        "    return 'first'\n",
        encoding="utf-8",
    )
    existing_python_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path)
        if not existing_python_path
        else f"{tmp_path}{os.pathsep}{existing_python_path}",
    )
    first = Scenario(
        scenario_id="first",
        name="first",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1,),
        edge_labels=(),
        contains_loop=False,
        contains_exception=False,
        description="stateful first",
        keyword_arguments=(),
        expected_result="first",
        expected_exception=None,
    )
    second = Scenario(
        scenario_id="second",
        name="second",
        path_index=2,
        priority_rank=2,
        priority_level="High",
        dqm_score=90.0,
        node_ids=(1,),
        edge_labels=(),
        contains_loop=False,
        contains_exception=False,
        description="stateful second",
        keyword_arguments=(),
        expected_result="second",
        expected_exception=None,
    )

    result = ScenarioCoverageMinimizationService().minimize(
        source_file=source_file,
        module_path="stateful_target",
        function_name="advance",
        function_start_line=2,
        function_end_line=7,
        scenarios=(first, second),
        output_root=tmp_path / "stateful-output",
        timeout_seconds=20.0,
    )

    assert result.coverage_preserved is False
    assert result.status is ScenarioMinimizationStatus.TARGET_NOT_COVERED
    assert result.contributions[1].measurement_status is CoverageMeasurementStatus.INTERACTION_DEPENDENT
