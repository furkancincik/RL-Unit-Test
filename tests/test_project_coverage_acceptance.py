from __future__ import annotations

from pathlib import Path

from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalExecutionPolicy,
    ExternalSourceAnalysisRequest,
    UploadedPythonFile,
)
from services.external_source_analysis_service import ExternalSourceAnalysisService


def test_sample_complex_five_functions_have_exact_combined_project_coverage(
    tmp_path: Path,
) -> None:
    source = Path("datasets/sample_complex_code.py")
    result = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile(source.name, source.read_bytes()),
            ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS,
            ExternalAnalysisConfiguration(
                output_root=tmp_path / "sample_complex_combined",
                maximum_functions_per_module=5,
                max_visits_per_node=3,
                episode_count=1,
                epsilon=0.0,
                random_seed=42,
                pytest_coverage_timeout_seconds=30.0,
                per_function_pipeline_timeout_seconds=120.0,
                run_greedy_baseline=True,
            ),
        )
    )

    combined = result.project_coverage
    assert combined is not None
    assert combined.scope.discovered_function_count == 5
    assert combined.scope.completed_function_count == 5
    assert combined.full_pytest_exit_code == 0
    assert combined.minimized_pytest_exit_code == 0
    assert combined.coverage_preserved is True
    assert combined.target_line_identities == combined.minimized_covered_line_identities
    assert combined.target_branch_identities == combined.minimized_covered_branch_identities
    assert combined.final_selected_count <= combined.full_scenario_count
    assert combined.globally_minimal is False
