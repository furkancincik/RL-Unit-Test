from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)
from models.project_analysis_result import (
    FunctionRunStatus,
    ProjectRunStatus,
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)
from services.source_analysis_orchestrator import (
    SourceAnalysisOrchestrator,
    SourceAnalysisValidationError,
)
from services.project_deadline import ProjectDeadline


SOURCE = """
def first(value: int) -> int:
    return value + 1

async def async_target(value: int) -> int:
    return value

def outer(value: int) -> int:
    def nested() -> int:
        return value
    return nested()

class Handler:
    @property
    def method(self, value: int) -> int:
        return value

def last(value: int) -> int:
    return value - 1
"""


def _diagnostic(
    source_file: Path,
    function_name: str,
    status: PipelineRunStatus,
) -> PipelineDiagnosticResult:
    return PipelineDiagnosticResult(
        status=status,
        source_file=source_file,
        function_name=function_name,
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=0.1,
        funnel=PipelineFunnelSnapshot(
            pre_concrete_scenario_count=2,
            concrete_validation_accepted_count=1,
            concrete_validation_rejected_count=1,
            final_scenario_count=1,
            rl_executed_test_count=(
                1 if status is PipelineRunStatus.COMPLETED else None
            ),
            q_table_state_count=(
                1 if status is PipelineRunStatus.COMPLETED else None
            ),
        ),
        line_coverage_percent=(
            100.0 if status is PipelineRunStatus.COMPLETED else None
        ),
        branch_coverage_percent=(
            100.0 if status is PipelineRunStatus.COMPLETED else None
        ),
        error_category=(
            None if status is PipelineRunStatus.COMPLETED else status.value
        ),
        error_message=(
            None if status is PipelineRunStatus.COMPLETED else "stopped"
        ),
    )


class FakeTrainingService:
    def __init__(
        self,
        outcomes: dict[str, PipelineDiagnosticResult | BaseException],
        calls: list[dict[str, Any]],
    ) -> None:
        self._outcomes = outcomes
        self._calls = calls

    def run_with_diagnostics(self, **arguments: Any) -> Any:
        self._calls.append(arguments)
        outcome = self._outcomes[arguments["function_name"]]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _orchestrator(
    outcomes: dict[str, PipelineDiagnosticResult | BaseException],
    calls: list[dict[str, Any]],
) -> SourceAnalysisOrchestrator:
    return SourceAnalysisOrchestrator(
        training_service_factory=lambda: FakeTrainingService(outcomes, calls)
    )


def _run_arguments(
    source_file: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "module_path": "fixture_target",
        "output_root": output_root,
        "max_visits_per_node": 3,
        "episode_count": 1,
        "epsilon": 0.0,
        "learning_rate": 0.5,
        "discount_factor": 0.9,
        "random_seed": 42,
        "overwrite": True,
        "timeout_seconds": 5.0,
        "per_function_timeout_seconds": 2.0,
    }


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixture_target.py"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def test_single_function_selection_and_timeout_forwarding(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(
            source_file, "first", PipelineRunStatus.COMPLETED
        )
    }

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name="first",
        all_functions=False,
        maximum_functions=1,
    )

    assert result.function_results[0].target.name == "first"
    assert result.function_results[0].status is FunctionRunStatus.COMPLETED
    assert all(
        item.status is FunctionRunStatus.SKIPPED_SELECTION
        for item in result.function_results[1:]
    )
    assert calls[0]["pipeline_timeout_seconds"] == 2.0
    assert result.status is ProjectRunStatus.COMPLETED


def test_explicit_qualified_selection_runs_only_exact_targets_and_reports_rest(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "qualified_fixture.py"
    source_file.write_text(
        "class First:\n"
        "    def execute(self, value: int) -> int:\n"
        "        return value\n\n"
        "class Second:\n"
        "    def execute(self, value: int) -> int:\n"
        "        return value + 1\n\n"
        "def helper(value: int) -> int:\n"
        "    return value - 1\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.COMPLETED)
        for name in ("Second.execute", "helper")
    }
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (
            QualifiedTargetSelector("fixture_target", "Second.execute"),
            QualifiedTargetSelector("fixture_target", "helper"),
        ),
    )

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=False,
        target_selection=selection,
    )

    assert [call["function_name"] for call in calls] == [
        "Second.execute",
        "helper",
    ]
    statuses = {
        item.target.qualified_name: item.status
        for item in result.function_results
    }
    assert statuses["First.execute"] is FunctionRunStatus.SKIPPED_SELECTION
    assert statuses["Second.execute"] is FunctionRunStatus.COMPLETED
    assert statuses["helper"] is FunctionRunStatus.COMPLETED
    skipped = next(
        item
        for item in result.function_results
        if item.target.qualified_name == "First.execute"
    )
    assert skipped.skip_reason == "TARGET_NOT_SELECTED"
    assert result.selection_skipped_function_count == 1
    assert result.status is ProjectRunStatus.COMPLETED


def test_explicit_selection_distinguishes_methods_in_the_same_class(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "same_class.py"
    source_file.write_text(
        "class Processor:\n"
        "    def first(self, value: int) -> int:\n"
        "        return value + 1\n\n"
        "    def second(self, value: int) -> int:\n"
        "        return value - 1\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("same_class", "Processor.second"),),
    )

    result = _orchestrator(
        {
            "Processor.second": _diagnostic(
                source_file,
                "Processor.second",
                PipelineRunStatus.COMPLETED,
            )
        },
        calls,
    ).run(
        source_file=source_file,
        module_path="same_class",
        function_name=None,
        all_functions=False,
        target_selection=selection,
        output_root=tmp_path / "same_class_output",
        max_visits_per_node=3,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        overwrite=True,
        timeout_seconds=5.0,
        per_function_timeout_seconds=2.0,
    )

    statuses = {
        item.target.qualified_name: item.status
        for item in result.function_results
    }
    assert statuses["Processor.second"] is FunctionRunStatus.COMPLETED
    assert statuses["Processor.first"] is FunctionRunStatus.SKIPPED_SELECTION
    assert [call["function_name"] for call in calls] == ["Processor.second"]


def test_explicit_selection_limit_counts_only_supported_selected_targets(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        "last": _diagnostic(source_file, "last", PipelineRunStatus.COMPLETED)
    }
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (
            QualifiedTargetSelector("fixture_target", "async_target"),
            QualifiedTargetSelector("fixture_target", "last"),
            QualifiedTargetSelector("fixture_target", "first"),
        ),
    )

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=False,
        target_selection=selection,
        maximum_functions=1,
    )

    statuses = {
        item.target.qualified_name: item.status
        for item in result.function_results
    }
    assert statuses["async_target"] is FunctionRunStatus.UNSUPPORTED
    assert statuses["last"] is FunctionRunStatus.COMPLETED
    assert statuses["first"] is FunctionRunStatus.SKIPPED_LIMIT
    assert [call["function_name"] for call in calls] == ["last"]


def test_explicit_selection_rejects_unknown_exact_target(
    source_file: Path,
    tmp_path: Path,
) -> None:
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("fixture_target", "Missing.execute"),),
    )

    with pytest.raises(SourceAnalysisValidationError, match="bulunamadı"):
        _orchestrator({}, []).run(
            **_run_arguments(source_file, tmp_path / "output"),
            function_name=None,
            all_functions=False,
            target_selection=selection,
        )


def test_strategy_comparison_flag_and_timeout_are_forwarded(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(source_file, "first", PipelineRunStatus.COMPLETED)
    }

    _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name="first",
        all_functions=False,
        run_strategy_comparison=True,
        comparison_timeout_seconds=1.5,
    )

    assert calls[0]["run_strategy_comparison"] is True
    assert calls[0]["comparison_timeout_seconds"] == 1.5


def test_all_functions_preserves_discovery_and_unsupported_results(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.COMPLETED)
        for name in ("first", "outer", "last")
    }

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert [target.qualified_name for target in result.discovered_targets] == [
        "first",
        "async_target",
        "outer",
        "outer.nested",
        "Handler.method",
        "last",
    ]
    assert [call["function_name"] for call in calls] == [
        "first",
        "outer",
        "last",
    ]
    assert result.unsupported_count == 3
    assert result.status is ProjectRunStatus.PARTIAL


def test_function_limit_preserves_source_order_and_explicit_skip_results(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.COMPLETED)
        for name in ("first", "outer", "last")
    }

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
        maximum_functions=2,
        run_greedy_baseline=True,
        run_strategy_comparison=True,
    )

    assert [item.target.qualified_name for item in result.function_results] == [
        "first",
        "async_target",
        "outer",
        "outer.nested",
        "Handler.method",
        "last",
    ]
    assert [call["function_name"] for call in calls] == ["first", "outer"]
    assert all(call["run_greedy_baseline"] is True for call in calls)
    assert all(call["run_strategy_comparison"] is True for call in calls)
    assert result.unsupported_count == 3
    assert result.limit_skipped_count == 1
    assert result.executed_function_count == 2
    assert result.status is ProjectRunStatus.PARTIAL
    limited = result.function_results[-1]
    assert limited.target.name == "last"
    assert limited.status is FunctionRunStatus.SKIPPED_LIMIT
    assert limited.skip_reason == "FUNCTION_LIMIT_EXCEEDED"
    assert limited.diagnostic is None
    assert limited.scenario_count is None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["summary"]["limit_skipped_function_count"] == 1
    assert report["functions"][-1]["status"] == "SKIPPED_LIMIT"


@pytest.mark.parametrize("maximum_functions", (True, 0, -1))
def test_function_limit_validation_is_preserved(
    source_file: Path,
    tmp_path: Path,
    maximum_functions: object,
) -> None:
    with pytest.raises(SourceAnalysisValidationError, match="maximum_functions"):
        _orchestrator({}, []).run(
            **_run_arguments(source_file, tmp_path / "output"),
            function_name=None,
            all_functions=True,
            maximum_functions=maximum_functions,
        )


def test_missing_function_lists_available_targets(
    source_file: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceAnalysisValidationError) as error:
        _orchestrator({}, []).run(
            **_run_arguments(source_file, tmp_path / "output"),
            function_name="missing",
            all_functions=False,
        )

    assert "missing" in str(error.value)
    assert "first" not in str(error.value)
    assert "last" not in str(error.value)


@pytest.mark.parametrize(
    "module_name,class_name,method_name,function_name,literal",
    (
        ("orbit_module", "Orbiter", "measure", "fallback", 3),
        ("harbor_module", "Navigator", "route", "reserve", 17),
        ("garden_module", "Cultivator", "harvest", "store", 29),
    ),
)
def test_explicit_selection_is_invariant_under_safe_target_renaming(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    method_name: str,
    function_name: str,
    literal: int,
) -> None:
    source_file = tmp_path / f"{module_name}.py"
    qualified_name = f"{class_name}.{method_name}"
    source_file.write_text(
        f"class {class_name}:\n"
        f"    def {method_name}(self, value: int) -> int:\n"
        f"        return value + {literal}\n\n"
        f"def {function_name}(value: int) -> int:\n"
        f"    return value - {literal}\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector(module_name, qualified_name),),
    )

    result = _orchestrator(
        {
            qualified_name: _diagnostic(
                source_file,
                qualified_name,
                PipelineRunStatus.COMPLETED,
            )
        },
        calls,
    ).run(
        source_file=source_file,
        module_path=module_name,
        function_name=None,
        all_functions=False,
        target_selection=selection,
        output_root=tmp_path / f"{module_name}_output",
        max_visits_per_node=3,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        overwrite=True,
        timeout_seconds=5.0,
        per_function_timeout_seconds=2.0,
    )

    assert [call["function_name"] for call in calls] == [qualified_name]
    statuses = {
        item.target.qualified_name: item.status
        for item in result.function_results
    }
    assert statuses[qualified_name] is FunctionRunStatus.COMPLETED
    assert statuses[function_name] is FunctionRunStatus.SKIPPED_SELECTION


def test_conflicting_target_selection_is_rejected(
    source_file: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceAnalysisValidationError, match="aynı anda"):
        _orchestrator({}, []).run(
            **_run_arguments(source_file, tmp_path / "output"),
            function_name="first",
            all_functions=True,
        )


@pytest.mark.parametrize(
    ("first_status", "last_status", "expected"),
    (
        (
            PipelineRunStatus.COMPLETED,
            PipelineRunStatus.FAILED,
            ProjectRunStatus.PARTIAL,
        ),
        (
            PipelineRunStatus.TIMED_OUT,
            PipelineRunStatus.COMPLETED,
            ProjectRunStatus.PARTIAL,
        ),
        (
            PipelineRunStatus.FAILED,
            PipelineRunStatus.FAILED,
            ProjectRunStatus.FAILED,
        ),
    ),
)
def test_function_results_are_isolated(
    source_file: Path,
    tmp_path: Path,
    first_status: PipelineRunStatus,
    last_status: PipelineRunStatus,
    expected: ProjectRunStatus,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(source_file, "first", first_status),
        "outer": _diagnostic(
            source_file, "outer", PipelineRunStatus.COMPLETED
        ),
        "last": _diagnostic(source_file, "last", last_status),
    }

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert len(calls) == 3
    assert result.status is expected or result.status is ProjectRunStatus.PARTIAL


@pytest.mark.parametrize(
    "error",
    (AssertionError("bug"), TypeError("bug"), RuntimeError("bug")),
)
def test_unexpected_programming_errors_propagate(
    source_file: Path,
    tmp_path: Path,
    error: BaseException,
) -> None:
    outcomes = {
        "first": error,
        "outer": _diagnostic(
            source_file, "outer", PipelineRunStatus.COMPLETED
        ),
        "last": _diagnostic(
            source_file, "last", PipelineRunStatus.COMPLETED
        ),
    }

    with pytest.raises(type(error), match="bug"):
        _orchestrator(outcomes, []).run(
            **_run_arguments(source_file, tmp_path / "output"),
            function_name=None,
            all_functions=True,
        )


def test_function_outputs_are_distinct_and_inside_root(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.COMPLETED)
        for name in ("first", "outer", "last")
    }
    output_root = tmp_path / "output"

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, output_root),
        function_name=None,
        all_functions=True,
    )

    directories = [Path(call["output_directory"]) for call in calls]
    assert len(set(directories)) == 3
    assert all(directory.is_relative_to(output_root.resolve()) for directory in directories)
    assert all(".." not in directory.parts for directory in directories)
    assert result.report_path.is_relative_to(output_root.resolve())


def test_project_report_is_atomic_json_without_sensitive_values(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(
            source_file, "first", PipelineRunStatus.COMPLETED
        )
    }
    output_root = tmp_path / "output"

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, output_root),
        function_name="first",
        all_functions=False,
    )

    value = json.loads(result.report_path.read_text(encoding="utf-8"))
    text = result.report_path.read_text(encoding="utf-8")
    assert value["project_status"] == "COMPLETED"
    assert value["aggregate_project_coverage"]["line_percent"] is None
    assert not tuple(output_root.glob("*.tmp"))
    assert "keyword_arguments" not in text
    assert "expected_result" not in text
    assert "actual_result" not in text
    assert "traceback" not in text.lower()


def test_two_runs_do_not_reuse_training_service_state(
    source_file: Path,
    tmp_path: Path,
) -> None:
    created_services: list[FakeTrainingService] = []
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(
            source_file, "first", PipelineRunStatus.COMPLETED
        )
    }

    def factory() -> FakeTrainingService:
        service = FakeTrainingService(outcomes, calls)
        created_services.append(service)
        return service

    orchestrator = SourceAnalysisOrchestrator(
        training_service_factory=factory
    )
    arguments = _run_arguments(source_file, tmp_path / "output")

    orchestrator.run(
        **arguments, function_name="first", all_functions=False
    )
    orchestrator.run(
        **arguments, function_name="first", all_functions=False
    )

    assert len(created_services) == 2


def test_function_limit_state_is_isolated_between_runs(
    source_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.COMPLETED)
        for name in ("first", "outer", "last")
    }
    orchestrator = _orchestrator(outcomes, calls)

    first = orchestrator.run(
        **_run_arguments(source_file, tmp_path / "first_output"),
        function_name=None,
        all_functions=True,
        maximum_functions=1,
    )
    second = orchestrator.run(
        **_run_arguments(source_file, tmp_path / "second_output"),
        function_name=None,
        all_functions=True,
        maximum_functions=1,
    )

    assert [call["function_name"] for call in calls] == ["first", "first"]
    assert first.limit_skipped_count == 2
    assert second.limit_skipped_count == 2
    assert [item.status for item in first.function_results] == [
        item.status for item in second.function_results
    ]


def test_all_failed_functions_produce_failed_project(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "failed.py"
    source_file.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    outcomes = {
        name: _diagnostic(source_file, name, PipelineRunStatus.FAILED)
        for name in ("first", "second")
    }

    result = _orchestrator(outcomes, []).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert result.status is ProjectRunStatus.FAILED
    assert result.failed_count == 2


def test_all_unsupported_functions_produce_explicit_results(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "unsupported.py"
    source_file.write_text(
        "async def async_target():\n    return 1\n\n"
        "class Handler:\n    @property\n    def method(self):\n        return 1\n",
        encoding="utf-8",
    )

    result = _orchestrator({}, []).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert result.status is ProjectRunStatus.FAILED
    assert result.unsupported_count == 2
    assert all(
        item.status is FunctionRunStatus.UNSUPPORTED
        for item in result.function_results
    )


def test_duplicate_top_level_names_are_not_run_ambiguously(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "duplicate.py"
    source_file.write_text(
        "def repeated():\n    return 1\n\ndef repeated():\n    return 2\n",
        encoding="utf-8",
    )

    result = _orchestrator({}, []).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert result.unsupported_count == 2
    assert all(
        item.skip_reason == "Duplicate top-level function names are ambiguous."
        for item in result.function_results
    )


def test_same_method_name_in_different_classes_uses_qualified_targets(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "qualified_methods.py"
    source_file.write_text(
        "class First:\n"
        "    def classify(self, value: int) -> int:\n"
        "        return value\n\n"
        "class Second:\n"
        "    def classify(self, value: int) -> int:\n"
        "        return value + 1\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    outcomes = {
        qualified_name: _diagnostic(
            source_file, qualified_name, PipelineRunStatus.COMPLETED
        )
        for qualified_name in ("First.classify", "Second.classify")
    }

    result = _orchestrator(outcomes, calls).run(
        **_run_arguments(source_file, tmp_path / "output"),
        function_name=None,
        all_functions=True,
    )

    assert [call["function_name"] for call in calls] == [
        "First.classify",
        "Second.classify",
    ]
    assert [item.target.qualified_name for item in result.function_results] == [
        "First.classify",
        "Second.classify",
    ]


def test_unsupported_method_target_dict_excludes_implicit_self(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "unsupported_target.py"
    source_file.write_text(
        "class Target:\n"
        "    @property\n"
        "    def value(self) -> int:\n"
        "        return 1\n",
        encoding="utf-8",
    )
    function = PythonAnalyzer().analyze_file(source_file).functions[0]

    payload = SourceAnalysisOrchestrator._to_target(function).to_dict()

    assert payload["qualified_name"] == "Target.value"
    assert payload["parameters"] == []
    assert payload["is_supported"] is False
    assert payload["unsupported_reason"] == (
        "Decorated instance methods are unsupported."
    )


class _DeadlineClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_project_deadline_preserves_completed_function_and_skips_unstarted_targets(
    source_file: Path,
    tmp_path: Path,
) -> None:
    clock = _DeadlineClock()
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(source_file, "first", PipelineRunStatus.COMPLETED),
    }

    class AdvancingTrainingService(FakeTrainingService):
        def run_with_diagnostics(self, **arguments: Any) -> Any:
            result = super().run_with_diagnostics(**arguments)
            clock.value = 3.0
            return result

    orchestrator = SourceAnalysisOrchestrator(
        training_service_factory=lambda: AdvancingTrainingService(outcomes, calls),
        clock=clock,
    )
    deadline = ProjectDeadline.start(2.0, clock=clock)

    result = orchestrator.run(
        **_run_arguments(source_file, tmp_path / "deadline_output"),
        function_name=None,
        all_functions=True,
        maximum_functions=1,
        project_deadline=deadline,
    )

    statuses = {item.target.qualified_name: item.status for item in result.function_results}
    assert statuses["first"] is FunctionRunStatus.COMPLETED
    assert statuses["outer"] is FunctionRunStatus.SKIPPED_DEADLINE
    assert statuses["last"] is FunctionRunStatus.SKIPPED_DEADLINE
    assert result.deadline_skipped_function_count == 2
    assert result.limit_skipped_function_count == 0
    assert result.executed_function_count == 1
    assert result.status is ProjectRunStatus.PARTIAL
    assert [call["function_name"] for call in calls] == ["first"]
    assert calls[0]["pipeline_timeout_seconds"] == pytest.approx(2.0)


def test_project_deadline_before_first_function_produces_timed_out_project(
    source_file: Path,
    tmp_path: Path,
) -> None:
    clock = _DeadlineClock()
    calls: list[dict[str, Any]] = []
    deadline = ProjectDeadline.start(1.0, clock=clock)
    clock.value = 1.0

    result = SourceAnalysisOrchestrator(
        training_service_factory=lambda: FakeTrainingService({}, calls),
        clock=clock,
    ).run(
        **_run_arguments(source_file, tmp_path / "expired_output"),
        function_name=None,
        all_functions=True,
        project_deadline=deadline,
    )

    assert result.status is ProjectRunStatus.TIMED_OUT
    assert result.deadline_skipped_function_count == 3
    assert result.executed_function_count == 0
    assert calls == []
    assert all(
        item.skip_reason == "PROJECT_DEADLINE_EXCEEDED"
        for item in result.function_results
        if item.status is FunctionRunStatus.SKIPPED_DEADLINE
    )


def test_active_worker_timeout_is_preserved_before_deadline_skips_later_targets(
    source_file: Path,
    tmp_path: Path,
) -> None:
    clock = _DeadlineClock()
    calls: list[dict[str, Any]] = []
    outcomes = {
        "first": _diagnostic(source_file, "first", PipelineRunStatus.TIMED_OUT),
    }

    class TimingOutTrainingService(FakeTrainingService):
        def run_with_diagnostics(self, **arguments: Any) -> Any:
            result = super().run_with_diagnostics(**arguments)
            clock.value = 2.0
            return result

    deadline = ProjectDeadline.start(2.0, clock=clock)
    result = SourceAnalysisOrchestrator(
        training_service_factory=lambda: TimingOutTrainingService(outcomes, calls),
        clock=clock,
    ).run(
        **_run_arguments(source_file, tmp_path / "active_timeout_output"),
        function_name=None,
        all_functions=True,
        project_deadline=deadline,
    )

    assert result.function_results[0].status is FunctionRunStatus.TIMED_OUT
    assert result.deadline_skipped_function_count == 2
    assert result.status is ProjectRunStatus.TIMED_OUT
