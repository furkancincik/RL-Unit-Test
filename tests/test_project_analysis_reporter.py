from pathlib import Path

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
from services.project_analysis_report_service import (
    ProjectAnalysisReportFormatter,
)


def test_project_formatter_reports_unmeasured_values_without_fake_zero() -> None:
    target = FunctionTarget(
        name="target",
        qualified_name="target",
        start_line=1,
        end_line=2,
        parameters=(),
        return_annotation=None,
        is_async=False,
        is_nested=False,
        is_method=False,
        is_supported=True,
    )
    diagnostic = PipelineDiagnosticResult(
        status=PipelineRunStatus.TIMED_OUT,
        source_file=Path("target.py"),
        function_name="target",
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=1.0,
        funnel=PipelineFunnelSnapshot(),
        error_message="Global timeout",
    )
    function_result = FunctionAnalysisResult(
        target=target,
        status=FunctionRunStatus.TIMED_OUT,
        diagnostic=diagnostic,
        output_directory=Path("output/target"),
    )
    project = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.SINGLE,
        discovered_targets=(target,),
        function_results=(function_result,),
        total_duration_seconds=1.0,
        status=ProjectRunStatus.TIMED_OUT,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    report = ProjectAnalysisReportFormatter().format(project)

    assert "PROJE ANALİZ ÖZETİ" in report
    assert "FONKSİYON SONUÇLARI" in report
    assert "Aggregate coverage    : Ölçülmedi" in report
    assert "Line/branch       : Ölçülmedi / Ölçülmedi" in report
    assert "Sınırlandırılmış yol: 0" in report
    assert "Girdi üretimi kabul/red: 0 / 0" in report
    assert "Red kategorileri    : Yok" in report
    assert "%0.00" not in report


def test_project_formatter_reports_completed_input_rejections_and_low_coverage() -> None:
    target = FunctionTarget(
        name="target",
        qualified_name="target",
        start_line=1,
        end_line=2,
        parameters=(),
        return_annotation=None,
        is_async=False,
        is_nested=False,
        is_method=False,
        is_supported=True,
    )
    diagnostic = PipelineDiagnosticResult(
        status=PipelineRunStatus.COMPLETED,
        source_file=Path("target.py"),
        function_name="target",
        last_completed_stage=None,
        stopped_stage=None,
        total_duration_seconds=1.0,
        funnel=PipelineFunnelSnapshot(
            bounded_path_count=19,
            input_generation_accepted_count=1,
            input_generation_rejected_count=18,
            pre_concrete_scenario_count=1,
            concrete_validation_accepted_count=1,
            concrete_validation_rejected_count=0,
            final_scenario_count=1,
        ),
        scenario_rejection_counts=(
            ("UNSUPPORTED_INPUT_SYNTHESIS", 18),
        ),
        line_coverage_percent=18.75,
        branch_coverage_percent=12.5,
    )
    function_result = FunctionAnalysisResult(
        target=target,
        status=FunctionRunStatus.COMPLETED,
        diagnostic=diagnostic,
        output_directory=Path("output/target"),
    )
    project = ProjectAnalysisResult(
        source_file=Path("target.py"),
        module_path="target",
        selection_mode=FunctionSelectionMode.SINGLE,
        discovered_targets=(target,),
        function_results=(function_result,),
        total_duration_seconds=1.0,
        status=ProjectRunStatus.COMPLETED,
        output_root=Path("output"),
        report_path=Path("output/project_analysis_report.json"),
    )

    report = ProjectAnalysisReportFormatter().format(project)

    assert "Proje durumu          : COMPLETED" in report
    assert "Sınırlandırılmış yol: 19" in report
    assert "Girdi üretimi kabul/red: 1 / 18" in report
    assert "Red kategorileri    : UNSUPPORTED_INPUT_SYNTHESIS: 18" in report
    assert "Scenario          : 1" in report
    assert "Concrete kabul/red: 1 / 0" in report
    assert "Line/branch       : %18.75 / %12.50" in report
