from datetime import UTC, datetime

from models.analysis_job_result import (
    AnalysisFunctionSummary,
    AnalysisJobStatus,
    AnalysisJobSummary,
)
from models.external_source_analysis_result import ExternalExecutionPolicy, ExternalSourceKind


def test_job_summary_is_utc_safe_and_contains_no_payload() -> None:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    summary = AnalysisJobSummary(
        job_id="opaque-id",
        source_kind=ExternalSourceKind.INLINE_PYTHON_SOURCE,
        analysis_policy=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        status=AnalysisJobStatus.QUEUED,
        created_at=created,
    )

    payload = summary.to_dict()

    assert payload["created_at"].endswith("+00:00")
    assert payload["status"] == "QUEUED"
    serialized = repr(payload)
    assert "source_code" not in serialized
    assert "file_bytes" not in serialized
    assert "traceback" not in serialized


def test_job_status_contains_all_lifecycle_states() -> None:
    assert {item.value for item in AnalysisJobStatus} == {
        "QUEUED", "RUNNING", "COMPLETED", "PARTIAL", "FAILED",
        "TIMED_OUT", "CANCELLED",
    }


def test_function_summary_keeps_pool_greedy_and_rl_coverage_distinct() -> None:
    summary = AnalysisFunctionSummary(
        qualified_name="target",
        status="COMPLETED",
        skip_reason=None,
        scenario_count=8,
        concrete_accepted_count=8,
        concrete_rejected_count=0,
        rl_test_count=3,
        q_table_state_count=3,
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        scenario_pool_line_coverage_percent=100.0,
        scenario_pool_branch_coverage_percent=100.0,
        greedy_line_coverage_percent=75.0,
        greedy_branch_coverage_percent=50.0,
        greedy_coverage_preserved=False,
        best_rl_line_coverage_percent=87.5,
        best_rl_branch_coverage_percent=80.0,
        best_rl_coverage_preserved=False,
        duration_seconds=1.0,
        stopped_stage=None,
        error_category=None,
        greedy_selected_count=2,
        rl_selected_count=3,
        strategy_winner="NOT_COMPARABLE",
        comparison_status="TARGET_NOT_REACHED",
        comparison_scenario_pool_count=8,
        greedy_reduction_percentage=75.0,
        rl_reduction_percentage=62.5,
        coverage_equality_verified=False,
        globally_minimal=False,
        rl_done_reason="ACTIONS_EXHAUSTED",
    )

    value = summary.to_dict()

    assert value["line_coverage_percent"] == 100.0
    assert value["scenario_pool_line_coverage_percent"] == 100.0
    assert value["greedy_line_coverage_percent"] == 75.0
    assert value["greedy_coverage_preserved"] is False
    assert value["best_rl_line_coverage_percent"] == 87.5
    assert value["best_rl_coverage_preserved"] is False
    assert value["globally_minimal"] is False
