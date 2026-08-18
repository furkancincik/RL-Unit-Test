from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
    PipelineStage,
)


def create_snapshot() -> PipelineFunnelSnapshot:
    return PipelineFunnelSnapshot(
        bounded_path_count=10,
        feasible_path_count=4,
        infeasible_path_count=5,
        unknown_path_count=1,
        candidate_generated_path_count=4,
        input_generation_accepted_count=3,
        input_generation_rejected_count=1,
        scenario_generation_accepted_count=3,
        scenario_generation_rejected_count=1,
        pre_concrete_scenario_count=3,
        concrete_validation_accepted_count=2,
        concrete_validation_rejected_count=1,
        final_scenario_count=2,
    )


def test_completed_diagnostic_is_immutable_and_serializable() -> None:
    result = PipelineDiagnosticResult(
        status=PipelineRunStatus.COMPLETED,
        source_file=Path("sample.py"),
        function_name="calculate",
        last_completed_stage=PipelineStage.REPORTING,
        stopped_stage=None,
        total_duration_seconds=1.25,
        stage_durations=((PipelineStage.SOURCE_ANALYSIS, 0.25),),
        funnel=create_snapshot(),
        scenario_rejection_counts=(("UNREACHABLE_INPUT", 1),),
        concrete_rejection_counts=(("RESULT_MISMATCH", 1),),
        line_coverage_percent=90.0,
        branch_coverage_percent=80.0,
        reachability_counts=(("FEASIBLE_UNCOVERED", 0),),
    )

    with pytest.raises(FrozenInstanceError):
        result.status = PipelineRunStatus.FAILED  # type: ignore[misc]

    serialized = result.to_dict()
    assert serialized["status"] == "COMPLETED"
    assert serialized["source_file"] == "sample.py"
    assert serialized["last_completed_stage"] == "REPORTING"
    assert serialized["stage_durations"] == {
        "SOURCE_ANALYSIS": 0.25,
    }
    assert serialized["funnel"]["final_scenario_count"] == 2
    json.dumps(serialized)


def test_timed_out_diagnostic_preserves_partial_snapshot() -> None:
    result = PipelineDiagnosticResult.timed_out(
        source_file=Path("sample.py"),
        function_name="calculate",
        stopped_stage=PipelineStage.RL_TRAINING,
        last_completed_stage=PipelineStage.COVERAGE_MEASUREMENT,
        total_duration_seconds=180.0,
        funnel=create_snapshot(),
        message="Orchestration timeout",
    )

    assert result.status is PipelineRunStatus.TIMED_OUT
    assert result.funnel.final_scenario_count == 2
    assert result.line_coverage_percent is None


@pytest.mark.parametrize("duration", (-0.1, float("nan")))
def test_diagnostic_rejects_invalid_durations(duration: float) -> None:
    with pytest.raises(ValueError):
        PipelineDiagnosticResult(
            status=PipelineRunStatus.PARTIAL,
            source_file=Path("sample.py"),
            function_name="calculate",
            last_completed_stage=PipelineStage.PATH_DISCOVERY,
            stopped_stage=PipelineStage.PATH_FEASIBILITY,
            total_duration_seconds=duration,
        )


def test_funnel_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="negatif"):
        PipelineFunnelSnapshot(bounded_path_count=-1)


def test_funnel_rejects_inconsistent_path_classification() -> None:
    with pytest.raises(ValueError, match="path sınıflandırma"):
        PipelineFunnelSnapshot(
            bounded_path_count=3,
            feasible_path_count=2,
            infeasible_path_count=2,
            unknown_path_count=0,
        )


def test_funnel_rejects_inconsistent_concrete_counts() -> None:
    with pytest.raises(ValueError, match="concrete validation"):
        PipelineFunnelSnapshot(
            pre_concrete_scenario_count=2,
            concrete_validation_accepted_count=2,
            concrete_validation_rejected_count=1,
        )


def test_unknown_metrics_remain_none() -> None:
    snapshot = PipelineFunnelSnapshot()
    result = PipelineDiagnosticResult(
        status=PipelineRunStatus.FAILED,
        source_file=Path("sample.py"),
        function_name="calculate",
        last_completed_stage=None,
        stopped_stage=PipelineStage.SOURCE_ANALYSIS,
        total_duration_seconds=0.0,
        funnel=snapshot,
    )

    assert snapshot.bounded_path_count is None
    assert result.line_coverage_percent is None
    assert result.to_dict()["funnel"]["bounded_path_count"] is None
