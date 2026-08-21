from datetime import UTC, datetime

from models.analysis_job_result import AnalysisJobStatus, AnalysisJobSummary
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
