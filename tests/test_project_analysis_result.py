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
