from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)
from models.project_analysis_result import (
    FunctionRunStatus,
    ProjectRunStatus,
)
from services.source_analysis_orchestrator import (
    SourceAnalysisOrchestrator,
    SourceAnalysisValidationError,
)


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
    )

    assert [item.target.name for item in result.function_results] == ["first"]
    assert calls[0]["pipeline_timeout_seconds"] == 2.0
    assert result.status is ProjectRunStatus.COMPLETED


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

    assert "first" in str(error.value)
    assert "last" in str(error.value)


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
        "class Handler:\n    def method(self):\n        return 1\n",
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
