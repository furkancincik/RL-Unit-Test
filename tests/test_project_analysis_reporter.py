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
    assert "%0.00" not in report
