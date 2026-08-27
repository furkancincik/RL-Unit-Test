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


def test_function_summary_exposes_normalized_input_rejection_funnel() -> None:
    summary = AnalysisFunctionSummary(
        qualified_name="target",
        status="COMPLETED",
        skip_reason=None,
        scenario_count=1,
        concrete_accepted_count=1,
        concrete_rejected_count=0,
        rl_test_count=1,
        q_table_state_count=1,
        line_coverage_percent=18.75,
        branch_coverage_percent=12.5,
        scenario_pool_line_coverage_percent=18.75,
        scenario_pool_branch_coverage_percent=12.5,
        greedy_line_coverage_percent=None,
        greedy_branch_coverage_percent=None,
        greedy_coverage_preserved=None,
        best_rl_line_coverage_percent=18.75,
        best_rl_branch_coverage_percent=12.5,
        best_rl_coverage_preserved=True,
        duration_seconds=1.0,
        stopped_stage=None,
        error_category=None,
        greedy_selected_count=None,
        rl_selected_count=1,
        strategy_winner=None,
        comparison_status=None,
        comparison_scenario_pool_count=None,
        greedy_reduction_percentage=None,
        rl_reduction_percentage=None,
        coverage_equality_verified=None,
        globally_minimal=None,
        rl_done_reason=None,
        bounded_path_count=19,
        input_generation_accepted_count=1,
        input_generation_rejected_count=18,
        input_rejection_categories=(
            ("UNSUPPORTED_INPUT_SYNTHESIS", 18),
        ),
    )

    value = summary.to_dict()

    assert value["status"] == "COMPLETED"
    assert value["bounded_path_count"] == 19
    assert value["input_generation_accepted_count"] == 1
    assert value["input_generation_rejected_count"] == 18
    assert value["input_rejection_categories"] == [
        {"category": "UNSUPPORTED_INPUT_SYNTHESIS", "count": 18}
    ]
    assert value["scenario_count"] == 1
    assert value["concrete_accepted_count"] == 1
    assert value["concrete_rejected_count"] == 0
    assert value["line_coverage_percent"] == 18.75
    assert value["branch_coverage_percent"] == 12.5


def test_function_summary_legacy_defaults_have_empty_input_rejections() -> None:
    summary = AnalysisFunctionSummary(
        qualified_name="legacy",
        status="COMPLETED",
        skip_reason=None,
        scenario_count=None,
        concrete_accepted_count=None,
        concrete_rejected_count=None,
        rl_test_count=None,
        q_table_state_count=None,
        line_coverage_percent=None,
        branch_coverage_percent=None,
        scenario_pool_line_coverage_percent=None,
        scenario_pool_branch_coverage_percent=None,
        greedy_line_coverage_percent=None,
        greedy_branch_coverage_percent=None,
        greedy_coverage_preserved=None,
        best_rl_line_coverage_percent=None,
        best_rl_branch_coverage_percent=None,
        best_rl_coverage_preserved=None,
        duration_seconds=None,
        stopped_stage=None,
        error_category=None,
        greedy_selected_count=None,
        rl_selected_count=None,
        strategy_winner=None,
        comparison_status=None,
        comparison_scenario_pool_count=None,
        greedy_reduction_percentage=None,
        rl_reduction_percentage=None,
        coverage_equality_verified=None,
        globally_minimal=None,
        rl_done_reason=None,
    )

    value = summary.to_dict()

    assert value["bounded_path_count"] == 0
    assert value["input_generation_accepted_count"] == 0
    assert value["input_generation_rejected_count"] == 0
    assert value["input_rejection_categories"] == []


def test_function_summary_sorts_categories_and_omits_unknown_internal_values() -> None:
    summary = AnalysisFunctionSummary(
        qualified_name="target",
        status="COMPLETED",
        skip_reason=None,
        scenario_count=1,
        concrete_accepted_count=1,
        concrete_rejected_count=0,
        rl_test_count=1,
        q_table_state_count=1,
        line_coverage_percent=50.0,
        branch_coverage_percent=25.0,
        scenario_pool_line_coverage_percent=50.0,
        scenario_pool_branch_coverage_percent=25.0,
        greedy_line_coverage_percent=None,
        greedy_branch_coverage_percent=None,
        greedy_coverage_preserved=None,
        best_rl_line_coverage_percent=None,
        best_rl_branch_coverage_percent=None,
        best_rl_coverage_preserved=None,
        duration_seconds=1.0,
        stopped_stage=None,
        error_category=None,
        greedy_selected_count=None,
        rl_selected_count=None,
        strategy_winner=None,
        comparison_status=None,
        comparison_scenario_pool_count=None,
        greedy_reduction_percentage=None,
        rl_reduction_percentage=None,
        coverage_equality_verified=None,
        globally_minimal=None,
        rl_done_reason=None,
        bounded_path_count=4,
        input_generation_accepted_count=1,
        input_generation_rejected_count=3,
        input_rejection_categories=(
            ("UNSUPPORTED_INPUT_SYNTHESIS", 1),
            ("PRIVATE_INTERNAL_DETAIL", 99),
            ("UNREACHABLE_INPUT", 2),
        ),
    )

    assert summary.to_dict()["input_rejection_categories"] == [
        {"category": "UNREACHABLE_INPUT", "count": 2},
        {"category": "UNSUPPORTED_INPUT_SYNTHESIS", "count": 1},
    ]
