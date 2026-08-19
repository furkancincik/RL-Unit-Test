from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
    LineReachabilityEvidence,
    LineReachabilityStatus,
)
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
    PipelineStage,
)

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.data_flow_analyzer import DataFlowAnalyzer
from cfg.path_analyzer import CFGPathAnalyzer, ExecutionPath
from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathFeasibilityAnalyzer,
)
from cfg.path_state_analyzer import PathStateAnalyzer
from evaluator.dqm import DQMScore, DecisionQualityMatrix
from generator.input_candidate_generator import InputCandidateGenerator
from generator.scenario_generator import (
    Scenario,
    ScenarioGenerator,
    ScenarioRejection,
    ScenarioRejectionCategory,
    ScenarioRejectionStage,
)
from rl.scenario_suite_coverage_transition import (
    ScenarioSuiteCoverageTransition,
)
from rl.training_report_formatter import TrainingReportFormatter
from rl.training_session import (
    TrainingSession,
    TrainingSessionResult,
)
from rl.training_statistics import (
    EpisodeStatistics,
    TrainingStatistics,
)
from services.coverage_reachability_service import (
    CoverageReachabilityService,
)
from services.pipeline_timeout_service import GlobalPipelineTimeoutRunner
from services.real_rl_training_service import (
    RealRLTrainingResult,
    RealRLTrainingService,
)
from services.scenario_coverage_minimization_service import (
    ScenarioCoverageMinimizationService,
)


def create_source_file(
    tmp_path: Path,
) -> Path:
    """Testlerde kullanılacak örnek Python dosyasını oluşturur."""
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 50:\n"
        "        return 'Başarılı'\n"
        "    return 'Başarısız'\n",
        encoding="utf-8",
    )

    return source_file


def create_function_analysis() -> Mock:
    """PythonAnalyzer fonksiyon sonucunu temsil eden mock oluşturur."""
    function = Mock()

    function.name = "calculate_score"
    function.parameters = ["score"]
    function.parameter_types = {
        "score": "int",
    }
    function.branch_count = 1
    function.line_number = 1
    function.end_line_number = 4

    return function


def create_graph() -> Mock:
    """CFG sonucunu temsil eden mock oluşturur."""
    graph = Mock()

    graph.function_name = "calculate_score"
    graph.nodes = (
        Mock(node_id=1),
        Mock(node_id=2),
        Mock(node_id=3),
        Mock(node_id=4),
    )

    return graph


def create_path() -> ExecutionPath:
    """Örnek yürütme yolu oluşturur."""
    return ExecutionPath(
        node_ids=[1, 2, 3, 4],
        edge_labels=[None, "True", None],
    )


def create_score() -> DQMScore:
    """Örnek DQM sonucu oluşturur."""
    return DQMScore(
        path_index=1,
        path_length=4,
        decision_edge_count=1,
        contains_loop=False,
        contains_exception=False,
        raw_score=10.0,
        normalized_score=100.0,
        priority_level="High",
    )


def create_scenario() -> Scenario:
    """Gerçek RL servis testinde kullanılacak senaryoyu oluşturur."""
    return Scenario(
        scenario_id="calculate_score_scenario_001",
        name="calculate_score yürütme yolu 1",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 2, 3, 4),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=False,
        description="Gerçek RL eğitim servisi testi.",
        keyword_arguments=(("score", 50),),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_session_result() -> TrainingSessionResult:
    """Tamamlanmış eğitim oturumu sonucu oluşturur."""
    episode = EpisodeStatistics(
        episode_number=1,
        step_count=1,
        total_reward=109.9,
        final_coverage_percentage=100.0,
        full_coverage=True,
        executed_test_count=1,
    )

    return TrainingSessionResult(
        episodes=(episode,),
        requested_episode_count=1,
        completed_episode_count=1,
    )


def create_function_coverage_result(
    tmp_path: Path,
    *,
    line_coverage_percent: float = 100.0,
    branch_coverage_percent: float = 100.0,
) -> FunctionCoverageResult:
    """Mock eğitim akışı için kontrollü coverage sonucu oluşturur."""
    source_file = create_source_file(tmp_path).resolve()
    test_file = (
        tmp_path
        / "test_calculate_score_scenario_suite.py"
    ).resolve()

    file_coverage = CoverageResult(
        source_file=source_file,
        test_file=test_file,
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=4,
        missing_line_count=0,
        total_line_count=4,
        covered_branch_count=2,
        missing_branch_count=0,
        total_branch_count=2,
        test_exit_code=0,
        duration_seconds=0.01,
    )

    return FunctionCoverageResult(
        source_file=source_file,
        test_file=test_file,
        function_name="calculate_score",
        start_line=1,
        end_line=4,
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_lines=(
            (1, 2, 3, 4)
            if line_coverage_percent == 100.0
            else (1, 2, 3)
        ),
        missing_lines=(
            ()
            if line_coverage_percent == 100.0
            else (4,)
        ),
        covered_branch_count=2,
        missing_branch_count=0,
        test_exit_code=0,
        duration_seconds=0.01,
        file_coverage=file_coverage,
    )


def create_reachability_result(
    coverage_result: FunctionCoverageResult,
) -> FunctionCoverageReachabilityResult:
    """Coverage sonucu için kontrollü erişilebilirlik sonucu oluşturur."""
    covered_lines = set(
        coverage_result.covered_lines
    )

    line_evidence = tuple(
        LineReachabilityEvidence(
            line_number=line_number,
            status=(
                LineReachabilityStatus.COVERED
                if line_number in covered_lines
                else LineReachabilityStatus.UNRESOLVED
            ),
        )
        for line_number in sorted(
            covered_lines
            | set(coverage_result.missing_lines)
        )
    )

    return FunctionCoverageReachabilityResult(
        coverage_result=coverage_result,
        line_evidence=line_evidence,
        analyzed_path_count=1,
        max_visits_per_node=3,
        path_metadata_complete=True,
    )


@pytest.fixture(autouse=True)
def provide_mock_transition_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    TrainingSession.run mocklandığında transition çağrılmadığı için
    servis testlerine kontrollü son coverage sonucu sağlar.
    """
    coverage_result = create_function_coverage_result(
        tmp_path
    )

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "last_coverage_result",
        property(
            lambda self: coverage_result
        ),
    )

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "measure_scenarios",
        lambda self, scenarios: coverage_result,
    )


def create_dependencies() -> tuple[
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
    Mock,
]:
    """RealRLTrainingService için mock bağımlılıklar oluşturur."""
    analyzer = Mock(spec=PythonAnalyzer)
    cfg_builder = Mock(spec=ControlFlowGraphBuilder)
    path_analyzer = Mock(spec=CFGPathAnalyzer)
    data_flow_analyzer = Mock(spec=DataFlowAnalyzer)
    path_state_analyzer = Mock(spec=PathStateAnalyzer)
    path_feasibility_analyzer = Mock(spec=PathFeasibilityAnalyzer)
    input_candidate_generator = Mock(spec=InputCandidateGenerator)
    dqm = Mock(spec=DecisionQualityMatrix)
    scenario_generator = Mock(spec=ScenarioGenerator)
    report_formatter = Mock(spec=TrainingReportFormatter)
    coverage_reachability_service = Mock(
        spec=CoverageReachabilityService,
    )

    function = create_function_analysis()
    graph = create_graph()
    path = create_path()
    score = create_score()
    scenario = create_scenario()

    analysis_result = Mock()
    analysis_result.functions = [function]

    data_flow_result = Mock()
    path_state = Mock()

    feasibility_result = Mock()
    feasibility_result.status = FeasibilityStatus.FEASIBLE
    feasibility_result.constraints = ()
    feasibility_result.relational_constraints = ()

    candidate = Mock()
    candidate.values = (("score", 50),)
    candidate.value_dict = {"score": 50}
    candidate.path_input_value_dict = {
        "score": 50,
    }

    analyzer.analyze_file.return_value = analysis_result
    cfg_builder.build_from_file.return_value = [graph]
    path_analyzer.find_paths.return_value = [path]
    data_flow_analyzer.analyze_file.return_value = data_flow_result
    path_state_analyzer.analyze_file.return_value = path_state
    path_feasibility_analyzer.analyze_paths.return_value = (
        feasibility_result,
    )
    input_candidate_generator.generate.return_value = candidate
    dqm.evaluate_paths.return_value = [score]
    scenario_generator.generate.return_value = [scenario]
    report_formatter.format_session.return_value = "RL EĞİTİM OTURUMU"
    coverage_reachability_service.analyze.side_effect = (
        lambda **kwargs: create_reachability_result(
            kwargs["coverage_result"]
        )
    )

    return (
        analyzer,
        cfg_builder,
        path_analyzer,
        data_flow_analyzer,
        path_state_analyzer,
        path_feasibility_analyzer,
        input_candidate_generator,
        dqm,
        scenario_generator,
        report_formatter,
        coverage_reachability_service,
    )


def create_service(
    scenario_minimization_service: ScenarioCoverageMinimizationService | None = None,
) -> tuple[
    RealRLTrainingService,
    tuple[Mock, ...],
]:
    """Mock bağımlılıklarla servis oluşturur."""
    dependencies = create_dependencies()

    service = RealRLTrainingService(
        analyzer=dependencies[0],
        cfg_builder=dependencies[1],
        path_analyzer=dependencies[2],
        data_flow_analyzer=dependencies[3],
        path_state_analyzer=dependencies[4],
        path_feasibility_analyzer=dependencies[5],
        input_candidate_generator=dependencies[6],
        dqm=dependencies[7],
        scenario_generator=dependencies[8],
        report_formatter=dependencies[9],
        coverage_reachability_service=dependencies[10],
        scenario_minimization_service=scenario_minimization_service,
    )

    return service, dependencies


@patch.object(
    TrainingSession,
    "run",
)
def test_run_returns_real_rl_training_result(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert isinstance(
        result,
        RealRLTrainingResult,
    )

    assert isinstance(
        result.final_coverage_result,
        FunctionCoverageResult,
    )
    assert result.function_coverage is not None
    assert result.file_coverage.line_coverage_percent == 100.0
    assert result.has_full_function_coverage is True
    assert result.has_full_file_coverage is True
    assert isinstance(
        result.reachability_result,
        FunctionCoverageReachabilityResult,
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_executes_analysis_pipeline(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, dependencies = create_service()

    source_file = create_source_file(
        tmp_path
    )

    service.run(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    analyzer = dependencies[0]
    cfg_builder = dependencies[1]
    path_analyzer = dependencies[2]
    dqm = dependencies[7]

    analyzer.analyze_file.assert_called_once_with(
        source_file.resolve()
    )

    cfg_builder.build_from_file.assert_called_once_with(
        source_file.resolve()
    )

    graph = cfg_builder.build_from_file.return_value[0]

    path_analyzer.find_paths.assert_called_once_with(
        graph,
        max_visits_per_node=3,
    )

    data_flow_analyzer = dependencies[3]
    path_state_analyzer = dependencies[4]
    path_feasibility_analyzer = dependencies[5]
    input_candidate_generator = dependencies[6]
    coverage_reachability_service = dependencies[10]

    data_flow_analyzer.analyze_file.assert_called_once()
    path_state_analyzer.analyze_file.assert_called_once()
    path_feasibility_analyzer.analyze_paths.assert_called_once()
    input_candidate_generator.generate.assert_called_once()

    dqm.evaluate_paths.assert_called_once()

    feasibility_call = (
        path_feasibility_analyzer
        .analyze_paths
        .call_args
        .kwargs
    )

    assert feasibility_call["parameter_types"] == {
        "score": "int",
    }

    reachability_call = (
        coverage_reachability_service.analyze
        .call_args
        .kwargs
    )

    assert isinstance(
        reachability_call["coverage_result"],
        FunctionCoverageResult,
    )
    assert reachability_call["paths"] == (
        path_analyzer.find_paths.return_value[0],
    )
    assert reachability_call["feasibility_results"] is (
        path_feasibility_analyzer
        .analyze_paths
        .return_value
    )
    assert reachability_call["max_visits_per_node"] == 3


@patch.object(
    TrainingSession,
    "run",
)
def test_run_generates_scenarios_with_parameters(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, dependencies = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    scenario_generator = dependencies[8]

    call_arguments = (
        scenario_generator.generate
        .call_args
        .kwargs
    )

    assert (
        call_arguments["function_name"]
        == "calculate_score"
    )
    assert call_arguments["parameter_names"] == (
        "score",
    )
    assert call_arguments["parameter_types"] == {
        "score": "int",
    }
    assert call_arguments["candidate_values_by_path"] == {
        1: {
            "score": 50,
        },
    }


@patch.object(
    TrainingSession,
    "run",
)
def test_run_executes_requested_episode_count(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=3,
    )

    assert (
        mock_run.call_args.kwargs["episode_count"]
        == 3
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_formats_training_report(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    session_result = create_session_result()
    mock_run.return_value = session_result

    service, dependencies = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    report_formatter = dependencies[9]

    report_formatter.format_session.assert_called_once()

    call_kwargs = (
        report_formatter.format_session
        .call_args
        .kwargs
    )

    assert call_kwargs["result"] is session_result
    assert call_kwargs["function_name"] == "calculate_score"
    assert isinstance(
        call_kwargs["coverage_result"],
        FunctionCoverageResult,
    )
    assert isinstance(
        call_kwargs["reachability_result"],
        FunctionCoverageReachabilityResult,
    )
    assert (
        call_kwargs["reachability_result"]
        is result.reachability_result
    )

    assert result.report == "RL EĞİTİM OTURUMU"


@patch.object(
    TrainingSession,
    "run",
)
def test_result_contains_generated_scenarios(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert result.scenario_count == 1
    assert result.scenarios[0].scenario_id == (
        "calculate_score_scenario_001"
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_result_exposes_session_information(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert result.success is True
    assert result.completed_episode_count == 1
    assert result.function_name == "calculate_score"
    assert result.module_path == "sample_code"


@patch.object(TrainingSession, "run")
def test_successful_run_exposes_completed_diagnostic(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    service, _ = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert result.diagnostic.status is PipelineRunStatus.COMPLETED
    assert result.diagnostic.last_completed_stage is PipelineStage.REPORTING
    assert result.diagnostic.funnel.bounded_path_count == 1
    assert result.diagnostic.funnel.final_scenario_count == 1
    assert result.diagnostic.line_coverage_percent == 100.0


def test_diagnostic_run_returns_partial_when_all_scenarios_rejected(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    dependencies[8].generate.return_value = []

    result = service.run_with_diagnostics(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    assert isinstance(result, PipelineDiagnosticResult)
    assert result.status is PipelineRunStatus.PARTIAL
    assert result.stopped_stage is PipelineStage.SCENARIO_GENERATION
    assert result.funnel.bounded_path_count == 1
    assert result.funnel.final_scenario_count is None


@pytest.mark.parametrize("error", (AssertionError(), TypeError(), RuntimeError()))
def test_diagnostic_run_does_not_swallow_programming_errors(
    error: Exception,
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    dependencies[0].analyze_file.side_effect = error

    with pytest.raises(type(error)):
        service.run_with_diagnostics(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_unexpected_error_diagnostic_does_not_expose_raw_message(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    dependencies[0].analyze_file.side_effect = RuntimeError(
        "secret input payload"
    )

    with pytest.raises(RuntimeError, match="secret input payload"):
        service.run_with_diagnostics(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    diagnostic = service.last_diagnostic_result
    assert diagnostic is not None
    assert diagnostic.exception_type == "RuntimeError"
    assert diagnostic.error_message == "Beklenmeyen pipeline hatası."
    assert "secret input payload" not in diagnostic.to_dict().values()


@pytest.mark.parametrize(
    "pipeline_timeout_seconds",
    (True, "1", object()),
)
def test_run_rejects_invalid_pipeline_timeout_type(
    pipeline_timeout_seconds: object,
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    with pytest.raises(TypeError, match="pipeline_timeout_seconds"):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            pipeline_timeout_seconds=pipeline_timeout_seconds,  # type: ignore[arg-type]
        )
    dependencies[0].analyze_file.assert_not_called()


@pytest.mark.parametrize(
    "pipeline_timeout_seconds",
    (0.0, -1.0, float("nan"), float("inf")),
)
def test_run_rejects_invalid_pipeline_timeout_value(
    pipeline_timeout_seconds: float,
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    with pytest.raises(ValueError, match="pipeline_timeout_seconds"):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            pipeline_timeout_seconds=pipeline_timeout_seconds,
        )
    dependencies[0].analyze_file.assert_not_called()


@patch.object(TrainingSession, "run")
def test_none_pipeline_timeout_preserves_in_process_run(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    service, dependencies = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
        pipeline_timeout_seconds=None,
    )

    assert isinstance(result, RealRLTrainingResult)
    dependencies[0].analyze_file.assert_called_once()


def test_global_timeout_rejects_injected_pipeline_dependencies(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    with pytest.raises(ValueError, match="varsayılan production"):
        service.run_with_diagnostics(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            pipeline_timeout_seconds=1.0,
        )

    dependencies[0].analyze_file.assert_not_called()


def test_global_timeout_keeps_coverage_timeout_meaning(tmp_path: Path) -> None:
    runner = Mock(spec=GlobalPipelineTimeoutRunner)
    runner.run.return_value = PipelineDiagnosticResult.timed_out(
        source_file=Path("sample.py"),
        function_name="calculate_score",
        stopped_stage=PipelineStage.COVERAGE_MEASUREMENT,
        last_completed_stage=PipelineStage.CONCRETE_VALIDATION,
        total_duration_seconds=2.0,
        funnel=PipelineFunnelSnapshot(),
        pipeline_timeout_seconds=2.0,
    )
    service = RealRLTrainingService(global_timeout_runner=runner)

    service.run_with_diagnostics(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        timeout_seconds=7.5,
        pipeline_timeout_seconds=2.0,
    )

    assert runner.run.call_args.kwargs["run_arguments"]["timeout_seconds"] == 7.5


@patch.object(TrainingSession, "run")
def test_diagnostic_state_does_not_leak_between_runs(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    service, dependencies = create_service()
    dependencies[8].generate.return_value = []

    partial = service.run_with_diagnostics(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )
    dependencies[8].generate.return_value = [create_scenario()]
    completed = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert partial.status is PipelineRunStatus.PARTIAL
    assert completed.diagnostic.status is PipelineRunStatus.COMPLETED
    assert completed.diagnostic.error_message is None


@patch.object(TrainingSession, "run")
def test_diagnostic_preserves_structured_and_concrete_rejections(
    mock_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    service, dependencies = create_service()
    accepted = create_scenario()
    rejected = Scenario(
        scenario_id="calculate_score_scenario_002",
        name="concrete mismatch",
        path_index=2,
        priority_rank=2,
        priority_level="Medium",
        dqm_score=25.0,
        node_ids=(1, 2, 4),
        edge_labels=(None, "False"),
        contains_loop=False,
        contains_exception=False,
        description="Concrete mismatch",
        keyword_arguments=(("score", 10),),
        expected_result="Başarılı",
    )
    dependencies[8].generate.return_value = [accepted, rejected]
    dependencies[8].rejections = (
        ScenarioRejection(
            path_index=3,
            stage=ScenarioRejectionStage.PATH_INPUT_GENERATION,
            category=ScenarioRejectionCategory.UNSUPPORTED_INPUT_SYNTHESIS,
            message="Güvenli özet",
            exception_type="UnsupportedInputSynthesisError",
        ),
    )

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert result.diagnostic is not None
    assert result.diagnostic.scenario_rejection_counts == (
        ("UNSUPPORTED_INPUT_SYNTHESIS", 1),
    )
    assert result.diagnostic.concrete_rejection_counts == (
        ("EXECUTION_MISMATCH", 1),
    )
    assert result.diagnostic.funnel.pre_concrete_scenario_count == 2
    assert result.diagnostic.funnel.final_scenario_count == 1


def test_all_concrete_rejections_produce_partial_diagnostic(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()
    scenario = create_scenario()
    dependencies[8].generate.return_value = [
        Scenario(
            scenario_id=scenario.scenario_id,
            name=scenario.name,
            path_index=scenario.path_index,
            priority_rank=scenario.priority_rank,
            priority_level=scenario.priority_level,
            dqm_score=scenario.dqm_score,
            node_ids=scenario.node_ids,
            edge_labels=scenario.edge_labels,
            contains_loop=scenario.contains_loop,
            contains_exception=scenario.contains_exception,
            description=scenario.description,
            keyword_arguments=(("score", 10),),
            expected_result="Başarılı",
        )
    ]

    result = service.run_with_diagnostics(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    assert isinstance(result, PipelineDiagnosticResult)
    assert result.stopped_stage is PipelineStage.CONCRETE_VALIDATION
    assert result.funnel.concrete_validation_accepted_count == 0
    assert result.funnel.concrete_validation_rejected_count == 1
    assert result.funnel.final_scenario_count == 0


@patch.object(TrainingSession, "run")
def test_rl_failure_preserves_completed_coverage_diagnostic(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.side_effect = RuntimeError("trainer invariant")
    service, _ = create_service()

    with pytest.raises(RuntimeError, match="trainer invariant"):
        service.run_with_diagnostics(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            episode_count=1,
        )

    diagnostic = service.last_diagnostic_result
    assert diagnostic is not None
    assert diagnostic.status is PipelineRunStatus.PARTIAL
    assert diagnostic.last_completed_stage is PipelineStage.COVERAGE_MEASUREMENT
    assert diagnostic.stopped_stage is PipelineStage.RL_TRAINING
    assert diagnostic.line_coverage_percent == 100.0
    assert diagnostic.funnel.rl_executed_test_count is None


def test_run_rejects_missing_function(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    analysis_result = Mock()
    analysis_result.functions = []

    dependencies[0].analyze_file.return_value = (
        analysis_result
    )

    with pytest.raises(
        ValueError,
        match=(
            "Kaynak dosyada belirtilen "
            "fonksiyon bulunamadı"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_missing_graph(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    dependencies[1].build_from_file.return_value = []

    with pytest.raises(
        ValueError,
        match=(
            "Fonksiyon için Control Flow Graph "
            "bulunamadı"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_empty_paths(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    dependencies[2].find_paths.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için yürütme yolu bulunamadı",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_empty_dqm_scores(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    dependencies[7].evaluate_paths.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için uygulanabilir DQM sonucu üretilemedi",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_empty_scenarios(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    dependencies[8].generate.return_value = []

    with pytest.raises(
        ValueError,
        match=(
            "Fonksiyon için test senaryosu "
            "üretilemedi"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_invalid_episode_count(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match=(
            "episode_count 1 veya "
            "daha büyük olmalıdır"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            episode_count=0,
        )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_passes_custom_max_visits_to_path_analyzer(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    """Dışarıdan verilen path bütçesi CFG analizörüne aktarılır."""
    mock_run.return_value = create_session_result()

    service, dependencies = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        max_visits_per_node=5,
        episode_count=1,
    )

    graph = dependencies[
        1
    ].build_from_file.return_value[0]

    dependencies[2].find_paths.assert_called_once_with(
        graph,
        max_visits_per_node=5,
    )


@pytest.mark.parametrize(
    "max_visits_per_node",
    (
        True,
        3.0,
        "3",
        object(),
    ),
)
def test_run_rejects_invalid_max_visits_type_before_analysis(
    tmp_path: Path,
    max_visits_per_node: object,
) -> None:
    """Path bütçesi yalnızca gerçek integer değer kabul eder."""
    service, dependencies = create_service()

    with pytest.raises(
        TypeError,
        match=(
            "max_visits_per_node bir tam sayı "
            "olmalıdır"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            max_visits_per_node=(  # type: ignore[arg-type]
                max_visits_per_node
            ),
        )

    dependencies[0].analyze_file.assert_not_called()


@pytest.mark.parametrize(
    "max_visits_per_node",
    (
        0,
        -1,
    ),
)
def test_run_rejects_non_positive_max_visits_before_analysis(
    tmp_path: Path,
    max_visits_per_node: int,
) -> None:
    """Sıfır ve negatif path ziyaret sınırları reddedilir."""
    service, dependencies = create_service()

    with pytest.raises(
        ValueError,
        match=(
            "max_visits_per_node 1 veya daha büyük "
            "olmalıdır"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            max_visits_per_node=max_visits_per_node,
        )

    dependencies[0].analyze_file.assert_not_called()


def test_run_rejects_invalid_epsilon(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="epsilon 0 ile 1 arasında olmalıdır",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            epsilon=1.5,
        )


def test_run_rejects_invalid_learning_rate(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        TypeError,
        match="learning_rate sayısal olmalıdır",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            learning_rate=True,
        )


def test_run_rejects_invalid_timeout(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match=(
            "timeout_seconds sonlu ve "
            "sıfırdan büyük olmalıdır"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            timeout_seconds=0.0,
        )


def test_run_rejects_invalid_module_path(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample/code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_run_rejects_invalid_function_name(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python fonksiyon adı",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate-score",
            output_directory=tmp_path,
        )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_uses_scenario_pool_baseline_as_environment_target(
    mock_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    baseline_coverage = create_function_coverage_result(
        tmp_path,
        line_coverage_percent=75.0,
        branch_coverage_percent=50.0,
    )

    measured_scenarios: list[
        tuple[Scenario, ...]
    ] = []

    def measure_baseline(
        transition: ScenarioSuiteCoverageTransition,
        scenarios: tuple[Scenario, ...],
    ) -> FunctionCoverageResult:
        measured_scenarios.append(scenarios)
        return baseline_coverage

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "measure_scenarios",
        measure_baseline,
    )

    service, _ = create_service()

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    environment = mock_run.call_args.kwargs[
        "environment"
    ]

    assert (
        environment.target_coverage_percentage
        == pytest.approx(75.0)
    )
    assert measured_scenarios == [
        result.scenarios
    ]


@patch.object(TrainingSession, "run")
def test_run_keeps_greedy_baseline_disabled_by_default(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    minimizer = Mock(spec=ScenarioCoverageMinimizationService)
    service, _ = create_service(minimizer)

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    minimizer.minimize.assert_not_called()
    assert result.minimization_result is None


@patch.object(TrainingSession, "run")
def test_run_optionally_minimizes_concrete_valid_pool_with_existing_target(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    minimization_result = Mock()
    minimizer = Mock(spec=ScenarioCoverageMinimizationService)
    minimizer.minimize.return_value = minimization_result
    service, _ = create_service(minimizer)

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
        run_greedy_baseline=True,
        greedy_timeout_seconds=8.0,
    )

    call = minimizer.minimize.call_args.kwargs
    assert call["scenarios"] == result.scenarios
    assert call["full_pool_coverage"].function_name == "calculate_score"
    assert call["minimization_timeout_seconds"] == 8.0
    assert result.minimization_result is minimization_result


@patch.object(TrainingSession, "run")
def test_run_does_not_swallow_unexpected_minimizer_runtime_error(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()
    minimizer = Mock(spec=ScenarioCoverageMinimizationService)
    minimizer.minimize.side_effect = RuntimeError("unexpected minimizer failure")
    service, _ = create_service(minimizer)

    with pytest.raises(RuntimeError, match="unexpected minimizer failure"):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            episode_count=1,
            run_greedy_baseline=True,
        )


@pytest.mark.parametrize("value", (0.0, True, "slow"))
def test_run_rejects_invalid_greedy_timeout(
    tmp_path: Path,
    value: object,
) -> None:
    service, _ = create_service()

    with pytest.raises((TypeError, ValueError), match="greedy_timeout_seconds"):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            greedy_timeout_seconds=value,  # type: ignore[arg-type]
        )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_accepts_non_full_scenario_pool_coverage_target(
    mock_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    baseline_coverage = create_function_coverage_result(
        tmp_path,
        line_coverage_percent=62.5,
        branch_coverage_percent=50.0,
    )

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "measure_scenarios",
        lambda self, scenarios: baseline_coverage,
    )

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    environment = mock_run.call_args.kwargs[
        "environment"
    ]

    assert (
        environment.target_coverage_percentage
        == pytest.approx(62.5)
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_rejects_zero_scenario_pool_coverage_before_training(
    mock_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zero_coverage = create_function_coverage_result(
        tmp_path,
        line_coverage_percent=0.0,
        branch_coverage_percent=0.0,
    )

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "measure_scenarios",
        lambda self, scenarios: zero_coverage,
    )

    service, _ = create_service()

    with pytest.raises(
        RuntimeError,
        match=(
            "pozitif bir coverage "
            "hedefi üretemedi"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            episode_count=1,
        )

    mock_run.assert_not_called()


@patch.object(
    TrainingSession,
    "run",
)
def test_run_baseline_uses_only_executable_scenarios(
    mock_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    executable_scenario = create_scenario()

    rejected_scenario = Scenario(
        scenario_id="calculate_score_scenario_rejected",
        name="calculate_score reddedilen yol",
        path_index=2,
        priority_rank=2,
        priority_level="Medium",
        dqm_score=50.0,
        node_ids=(1, 2, 4),
        edge_labels=(None, "False"),
        contains_loop=False,
        contains_exception=False,
        description="Concrete validation tarafından elenecek senaryo.",
        keyword_arguments=(("score", 10),),
        expected_result="Yanlış beklenen sonuç",
        expected_exception=None,
    )

    service, dependencies = create_service()

    dependencies[8].generate.return_value = [
        executable_scenario,
        rejected_scenario,
    ]

    measured_scenarios: list[
        tuple[Scenario, ...]
    ] = []

    baseline_coverage = create_function_coverage_result(
        tmp_path,
        line_coverage_percent=75.0,
        branch_coverage_percent=50.0,
    )

    def measure_baseline(
        transition: ScenarioSuiteCoverageTransition,
        scenarios: tuple[Scenario, ...],
    ) -> FunctionCoverageResult:
        measured_scenarios.append(scenarios)
        return baseline_coverage

    monkeypatch.setattr(
        ScenarioSuiteCoverageTransition,
        "measure_scenarios",
        measure_baseline,
    )

    result = service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=1,
    )

    assert result.scenarios == (
        executable_scenario,
    )
    assert measured_scenarios == [
        (
            executable_scenario,
        )
    ]


@patch.object(
    TrainingSession,
    "run",
)
def test_run_passes_default_epsilon_decay_configuration_to_session(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=3,
    )

    call_kwargs = mock_run.call_args.kwargs

    assert call_kwargs["episode_count"] == 3
    assert (
        call_kwargs["epsilon_decay_rate"]
        == pytest.approx(0.95)
    )
    assert (
        call_kwargs["minimum_epsilon"]
        == pytest.approx(0.05)
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_passes_custom_epsilon_decay_configuration_to_session(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=2,
        epsilon=0.40,
        epsilon_decay_rate=0.80,
        minimum_epsilon=0.10,
    )

    call_kwargs = mock_run.call_args.kwargs

    assert call_kwargs["episode_count"] == 2
    assert (
        call_kwargs["epsilon_decay_rate"]
        == pytest.approx(0.80)
    )
    assert (
        call_kwargs["minimum_epsilon"]
        == pytest.approx(0.10)
    )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_allows_disabling_epsilon_decay(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        episode_count=2,
        epsilon=0.25,
        epsilon_decay_rate=None,
        minimum_epsilon=0.05,
    )

    call_kwargs = mock_run.call_args.kwargs

    assert (
        call_kwargs["epsilon_decay_rate"]
        is None
    )
    assert (
        call_kwargs["minimum_epsilon"]
        == pytest.approx(0.05)
    )


@pytest.mark.parametrize(
    "epsilon_decay_rate",
    (
        -0.1,
        1.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_run_rejects_invalid_epsilon_decay_rate(
    tmp_path: Path,
    epsilon_decay_rate: float,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            epsilon_decay_rate=epsilon_decay_rate,
        )


@pytest.mark.parametrize(
    "epsilon_decay_rate",
    (
        True,
        "0.95",
        object(),
    ),
)
def test_run_rejects_invalid_epsilon_decay_rate_type(
    tmp_path: Path,
    epsilon_decay_rate: object,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        TypeError,
        match="epsilon_decay_rate sayısal olmalıdır",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            epsilon_decay_rate=(  # type: ignore[arg-type]
                epsilon_decay_rate
            ),
        )


@pytest.mark.parametrize(
    "minimum_epsilon",
    (
        -0.1,
        1.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_run_rejects_invalid_minimum_epsilon(
    tmp_path: Path,
    minimum_epsilon: float,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            minimum_epsilon=minimum_epsilon,
        )


@pytest.mark.parametrize(
    "minimum_epsilon",
    (
        True,
        "0.05",
        object(),
    ),
)
def test_run_rejects_invalid_minimum_epsilon_type(
    tmp_path: Path,
    minimum_epsilon: object,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        TypeError,
        match="minimum_epsilon sayısal olmalıdır",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            minimum_epsilon=(  # type: ignore[arg-type]
                minimum_epsilon
            ),
        )


def test_run_rejects_minimum_epsilon_above_initial_epsilon(
    tmp_path: Path,
) -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match=(
            "minimum_epsilon başlangıç epsilon "
            "değerinden büyük olamaz"
        ),
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            epsilon=0.10,
            epsilon_decay_rate=0.95,
            minimum_epsilon=0.20,
        )


@patch.object(
    TrainingSession,
    "run",
)
def test_run_allows_minimum_above_epsilon_when_decay_disabled(
    mock_run: Mock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = create_session_result()

    service, _ = create_service()

    service.run(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.50,
    )

    assert (
        mock_run.call_args.kwargs[
            "epsilon_decay_rate"
        ]
        is None
    )


# ============================================================
# Feasibility + candidate pipeline integration tests
# ============================================================


def test_run_filters_infeasible_score_before_scenario_generation(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    infeasible_result = Mock()
    infeasible_result.status = FeasibilityStatus.INFEASIBLE
    infeasible_result.constraints = ()
    infeasible_result.relational_constraints = ()

    dependencies[5].analyze_paths.return_value = (
        infeasible_result,
    )

    with pytest.raises(
        ValueError,
        match="FEASIBLE veya UNKNOWN yürütme yolu bulunamadı",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    dependencies[8].generate.assert_not_called()


def test_run_keeps_unknown_path_without_candidate_values(
    tmp_path: Path,
) -> None:
    # Training'e kadar ilerlemeyi gerektirmeden ScenarioGenerator çağrısını
    # yakalamak için generate sonrasında kontrollü olarak durduruyoruz.
    service, dependencies = create_service()

    unknown_result = Mock()
    unknown_result.status = FeasibilityStatus.UNKNOWN
    unknown_result.constraints = ()
    unknown_result.relational_constraints = ()

    dependencies[5].analyze_paths.return_value = (
        unknown_result,
    )
    dependencies[8].generate.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için test senaryosu üretilemedi",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    call_kwargs = dependencies[8].generate.call_args.kwargs
    assert call_kwargs["candidate_values_by_path"] == {}
    dependencies[6].generate.assert_not_called()


def test_run_passes_feasible_candidate_values_to_scenario_generator(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    dependencies[8].generate.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için test senaryosu üretilemedi",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    call_kwargs = dependencies[8].generate.call_args.kwargs

    assert call_kwargs["candidate_values_by_path"] == {
        1: {
            "score": 50,
        },
    }

    dependencies[6].generate.assert_called_once()


def test_run_does_not_forward_structural_truthiness_placeholder(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    candidate = dependencies[
        6
    ].generate.return_value

    candidate.values = (
        ("values", True),
    )
    candidate.value_dict = {
        "values": True,
    }
    candidate.path_input_value_dict = {}

    dependencies[8].generate.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için test senaryosu üretilemedi",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    call_kwargs = (
        dependencies[8]
        .generate
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["candidate_values_by_path"]
        == {}
    )


def test_run_filters_dqm_scores_for_infeasible_paths(
    tmp_path: Path,
) -> None:
    service, dependencies = create_service()

    second_path = ExecutionPath(
        node_ids=[1, 2, 4],
        edge_labels=[None, "False"],
    )
    dependencies[2].find_paths.return_value = [
        create_path(),
        second_path,
    ]

    feasible_result = Mock()
    feasible_result.status = FeasibilityStatus.FEASIBLE
    feasible_result.constraints = ()
    feasible_result.relational_constraints = ()

    infeasible_result = Mock()
    infeasible_result.status = FeasibilityStatus.INFEASIBLE
    infeasible_result.constraints = ()
    infeasible_result.relational_constraints = ()

    dependencies[5].analyze_paths.return_value = (
        feasible_result,
        infeasible_result,
    )
    dependencies[4].analyze_file.side_effect = [
        Mock(),
        Mock(),
    ]

    second_score = DQMScore(
        path_index=2,
        path_length=3,
        decision_edge_count=1,
        contains_loop=False,
        contains_exception=False,
        raw_score=5.0,
        normalized_score=50.0,
        priority_level="Medium",
    )
    dependencies[7].evaluate_paths.return_value = [
        create_score(),
        second_score,
    ]
    dependencies[8].generate.return_value = []

    with pytest.raises(
        ValueError,
        match="Fonksiyon için test senaryosu üretilemedi",
    ):
        service.run(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )

    call_kwargs = dependencies[8].generate.call_args.kwargs
    assert [score.path_index for score in call_kwargs["scores"]] == [1]
