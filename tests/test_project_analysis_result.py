from pathlib import Path

import pytest

from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)
from models.project_analysis_result import (
    FunctionAnalysisResult,
    FunctionRunStatus,
    FunctionSelectionMode,
    FunctionTarget,
    ProjectAnalysisResult,
    ProjectRunStatus,
)
from models.strategy_comparison_result import (
    StrategyComparisonResult,
    StrategyComparisonStatus,
    StrategyWinner,
)


def _target(name: str = "first") -> FunctionTarget:
    return FunctionTarget(
        name=name,
        qualified_name=name,
        start_line=1,
        end_line=2,
        parameters=("value",),
        return_annotation="int",
        is_async=False,
        is_nested=False,
        is_method=False,
        is_supported=True,
    )


def _diagnostic(
    status: PipelineRunStatus = PipelineRunStatus.COMPLETED,
) -> PipelineDiagnosticResult:
    return PipelineDiagnosticResult(
        status=status,
        source_file=Path("target.py"),
        function_name="first",
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=0.25,
        funnel=PipelineFunnelSnapshot(
            final_scenario_count=2,
            concrete_validation_accepted_count=2,
            concrete_validation_rejected_count=0,
            pre_concrete_scenario_count=2,
            rl_executed_test_count=2,
            q_table_state_count=2,
        ),
        line_coverage_percent=75.0,
        branch_coverage_percent=50.0,
    )


def _function_result(
    status: FunctionRunStatus,
    name: str = "first",
) -> FunctionAnalysisResult:
    return FunctionAnalysisResult(
        target=_target(name),
        status=status,
        diagnostic=_diagnostic(
            PipelineRunStatus.COMPLETED
            if status is FunctionRunStatus.COMPLETED
            else PipelineRunStatus.FAILED
        ),
        output_directory=Path("output") / name,
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    (
        ((FunctionRunStatus.COMPLETED,), ProjectRunStatus.COMPLETED),
        (
            (FunctionRunStatus.COMPLETED, FunctionRunStatus.FAILED),
            ProjectRunStatus.PARTIAL,
        ),
        ((FunctionRunStatus.PARTIAL,), ProjectRunStatus.PARTIAL),
        ((FunctionRunStatus.FAILED,), ProjectRunStatus.FAILED),
        ((FunctionRunStatus.UNSUPPORTED,), ProjectRunStatus.FAILED),
        ((FunctionRunStatus.TIMED_OUT,), ProjectRunStatus.TIMED_OUT),
        ((FunctionRunStatus.SKIPPED_DEADLINE,), ProjectRunStatus.TIMED_OUT),
        ((FunctionRunStatus.SKIPPED_SELECTION,), ProjectRunStatus.COMPLETED),
        (
            (
                FunctionRunStatus.COMPLETED,
                FunctionRunStatus.SKIPPED_SELECTION,
            ),
            ProjectRunStatus.COMPLETED,
        ),
    ),
)
def test_project_status_policy(
    statuses: tuple[FunctionRunStatus, ...],
    expected: ProjectRunStatus,
) -> None:
    assert ProjectAnalysisResult.derive_status(statuses) is expected


def test_project_result_serializes_null_aggregate_coverage() -> None:
    target = _target()
    function_result = _function_result(FunctionRunStatus.COMPLETED)
    result = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.ALL,
        discovered_targets=(target,),
        function_results=(function_result,),
        total_duration_seconds=0.5,
        status=ProjectRunStatus.COMPLETED,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    value = result.to_dict()

    assert value["summary"]["discovered_function_count"] == 1
    assert value["summary"]["completed_count"] == 1
    assert value["aggregate_project_coverage"] == {
        "line_percent": None,
        "branch_percent": None,
        "status": "UNMEASURED",
    }
    assert value["functions"][0]["line_coverage_percent"] == 75.0
    assert "keyword_arguments" not in str(value)
    assert "expected_result" not in str(value)
    assert "actual_result" not in str(value)
    assert "traceback" not in str(value).lower()


def test_function_result_serializes_optional_strategy_comparison() -> None:
    comparison = StrategyComparisonResult(
        source_file=Path("target.py"),
        function_name="first",
        status=StrategyComparisonStatus.NOT_COMPARABLE,
        comparable=False,
        non_comparable_reason="RL_TARGET_NOT_REACHED",
        scenario_pool_count=2,
        target_line_identities=(1,),
        target_line_percentage=100.0,
        target_branch_identities=(),
        target_branch_percentage=100.0,
        greedy_algorithm="GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION",
        greedy_selected_scenario_ids=("first",),
        greedy_coverage_preserved=False,
        greedy_duration_seconds=0.1,
        requested_rl_episode_count=1,
        completed_rl_episode_count=0,
        rl_hyperparameters=(("epsilon", 0.0),),
        random_seed=42,
        episode_traces=(),
        exact_target_reaching_episode_count=0,
        best_rl_episode_number=None,
        best_rl_ordered_scenario_ids=(),
        best_rl_executed_test_count=None,
        best_rl_unique_scenario_count=None,
        best_rl_duplicate_count=None,
        best_rl_total_reward=None,
        rl_coverage_preserved=False,
        rl_duration_seconds=None,
        winner=StrategyWinner.NOT_COMPARABLE,
        coverage_equality_verified=False,
    )
    function_result = FunctionAnalysisResult(
        target=_target(),
        status=FunctionRunStatus.COMPLETED,
        diagnostic=_diagnostic(),
        output_directory=Path("output/first"),
        strategy_comparison=comparison,
    )

    value = function_result.to_dict()

    assert value["strategy_comparison"]["winner"] == "NOT_COMPARABLE"
    assert "keyword_arguments" not in str(value)


def test_unsupported_target_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        FunctionTarget(
            name="nested",
            qualified_name="outer.nested",
            start_line=2,
            end_line=3,
            parameters=(),
            return_annotation=None,
            is_async=False,
            is_nested=True,
            is_method=False,
            is_supported=False,
        )


def test_limit_skipped_function_is_explicit_and_unmeasured() -> None:
    function_result = FunctionAnalysisResult(
        target=_target("limited"),
        status=FunctionRunStatus.SKIPPED_LIMIT,
        diagnostic=None,
        output_directory=Path("output/limited"),
        skip_reason="FUNCTION_LIMIT_EXCEEDED",
    )
    result = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.ALL,
        discovered_targets=(function_result.target,),
        function_results=(function_result,),
        total_duration_seconds=0.1,
        status=ProjectRunStatus.FAILED,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    value = result.to_dict()

    assert function_result.scenario_count is None
    assert function_result.rl_test_count is None
    assert value["functions"][0]["status"] == "SKIPPED_LIMIT"
    assert value["functions"][0]["line_coverage_percent"] is None
    assert value["functions"][0]["skip_reason"] == "FUNCTION_LIMIT_EXCEEDED"
    assert value["summary"]["limit_skipped_function_count"] == 1
    assert value["summary"]["skipped_function_count"] == 1
    assert value["summary"]["executed_function_count"] == 0


def test_existing_skipped_status_remains_distinct_from_limit_skip() -> None:
    skipped = FunctionAnalysisResult(
        target=_target("ordinary_skip"),
        status=FunctionRunStatus.SKIPPED,
        diagnostic=None,
        output_directory=Path("output/ordinary_skip"),
        skip_reason="ORDINARY_SKIP",
    )
    limited = FunctionAnalysisResult(
        target=_target("limited"),
        status=FunctionRunStatus.SKIPPED_LIMIT,
        diagnostic=None,
        output_directory=Path("output/limited"),
        skip_reason="FUNCTION_LIMIT_EXCEEDED",
    )
    result = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.ALL,
        discovered_targets=(skipped.target, limited.target),
        function_results=(skipped, limited),
        total_duration_seconds=0.1,
        status=ProjectRunStatus.FAILED,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    assert result.skipped_count == 1
    assert result.limit_skipped_count == 1
    assert result.limit_skipped_function_count == 1
    assert result.skipped_function_count == 2


def test_selection_skip_is_visible_but_not_selected_or_executed() -> None:
    skipped = FunctionAnalysisResult(
        target=_target("unselected"),
        status=FunctionRunStatus.SKIPPED_SELECTION,
        diagnostic=None,
        output_directory=Path("output/unselected"),
        skip_reason="TARGET_NOT_SELECTED",
    )
    result = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        discovered_targets=(skipped.target,),
        function_results=(skipped,),
        total_duration_seconds=0.1,
        status=ProjectRunStatus.COMPLETED,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    payload = result.to_dict()

    assert result.selected_function_count == 0
    assert result.executed_function_count == 0
    assert result.selection_skipped_function_count == 1
    assert payload["functions"][0]["status"] == "SKIPPED_SELECTION"
    assert payload["summary"]["selection_skipped_function_count"] == 1


def test_deadline_skip_is_visible_unexecuted_and_uses_controlled_reason() -> None:
    skipped = FunctionAnalysisResult(
        target=_target("deadline_target"),
        status=FunctionRunStatus.SKIPPED_DEADLINE,
        diagnostic=None,
        output_directory=Path("output/deadline_target"),
        skip_reason="PROJECT_DEADLINE_EXCEEDED",
    )
    result = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.ALL,
        discovered_targets=(skipped.target,),
        function_results=(skipped,),
        total_duration_seconds=0.1,
        status=ProjectRunStatus.TIMED_OUT,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    payload = result.to_dict()
    assert result.deadline_skipped_function_count == 1
    assert result.executed_function_count == 0
    assert result.skipped_function_count == 1
    assert payload["functions"][0]["status"] == "SKIPPED_DEADLINE"
    assert payload["summary"]["deadline_skipped_function_count"] == 1
