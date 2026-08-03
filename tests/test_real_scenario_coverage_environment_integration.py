from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_coverage_transition import (
    ScenarioCoverageTransition,
)
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)
from services.scenario_coverage_service import (
    ScenarioCoverageResult,
    ScenarioCoverageService,
)


def create_source_file(
    tmp_path: Path,
) -> Path:
    """Entegrasyon testinde kullanılacak kaynak dosyayı oluşturur."""
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 85:\n"
        "        return 'Başarılı'\n"
        "    return 'Başarısız'\n",
        encoding="utf-8",
    )

    return source_file


def create_scenario(
    *,
    scenario_id: str,
    path_index: int,
    priority_rank: int,
    score: int,
    expected_result: str,
) -> Scenario:
    """Entegrasyon testlerinde kullanılacak Scenario oluşturur."""
    return Scenario(
        scenario_id=scenario_id,
        name=f"calculate_score yürütme yolu {path_index}",
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 3, 4, 2),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=False,
        description="Gerçek coverage geçiş entegrasyon testi.",
        keyword_arguments=(("score", score),),
        expected_result=expected_result,
        expected_exception=None,
    )


def create_initial_state() -> CoverageState:
    """RL ortamının başlangıç coverage durumunu oluşturur."""
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4),
        uncovered_branches=2,
    )


def create_coverage_result(
    *,
    source_file: Path,
    line_coverage_percent: float,
    branch_coverage_percent: float,
    covered_line_count: int,
    missing_line_count: int,
    total_line_count: int,
    covered_branch_count: int,
    missing_branch_count: int,
    total_branch_count: int,
) -> CoverageResult:
    """Kontrollü CoverageResult oluşturur."""
    return CoverageResult(
        source_file=source_file,
        test_file=Path("generated_test.py"),
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=0,
        duration_seconds=0.10,
    )


def create_mock_service(
    *,
    scenarios: list[Scenario],
    coverage_results: list[CoverageResult],
) -> Mock:
    """
    Her senaryo çağrısında sıradaki CoverageResult değerini
    döndüren mock servis oluşturur.
    """
    service = Mock(
        spec=ScenarioCoverageService,
    )

    service.measure_scenario.side_effect = [
        ScenarioCoverageResult(
            scenario=scenario,
            test_file=Path(
                f"test_{scenario.scenario_id}.py"
            ),
            coverage=coverage_result,
        )
        for scenario, coverage_result in zip(
            scenarios,
            coverage_results,
            strict=True,
        )
    ]

    return service


def test_environment_resolves_action_and_uses_real_coverage_transition(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    scenarios = [
        create_scenario(
            scenario_id="calculate_score_scenario_001",
            path_index=1,
            priority_rank=1,
            score=85,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="calculate_score_scenario_002",
            path_index=2,
            priority_rank=2,
            score=50,
            expected_result="Başarısız",
        ),
    ]

    coverage_results = [
        create_coverage_result(
            source_file=source_file,
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=2,
            missing_line_count=2,
            total_line_count=4,
            covered_branch_count=1,
            missing_branch_count=1,
            total_branch_count=2,
        ),
        create_coverage_result(
            source_file=source_file,
            line_coverage_percent=100.0,
            branch_coverage_percent=100.0,
            covered_line_count=4,
            missing_line_count=0,
            total_line_count=4,
            covered_branch_count=2,
            missing_branch_count=0,
            total_branch_count=2,
        ),
    ]

    service = create_mock_service(
        scenarios=scenarios,
        coverage_results=coverage_results,
    )

    mapper = ScenarioActionMapper(
        scenarios
    )

    scenario_transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    transition_adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=scenario_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=transition_adapter,
    )

    first_step = environment.step(
        mapper.actions[0]
    )

    assert first_step.state.coverage_percentage == 50.0
    assert first_step.state.executed_tests == 1
    assert first_step.state.missing_line_count == 2
    assert first_step.state.uncovered_branches == 1
    assert first_step.reward > 0.0
    assert first_step.done is False

    second_step = environment.step(
        mapper.actions[1]
    )

    assert second_step.state.coverage_percentage == 100.0
    assert second_step.state.executed_tests == 2
    assert second_step.state.missing_lines == ()
    assert second_step.state.uncovered_branches == 0
    assert second_step.state.is_fully_covered is True
    assert second_step.done is True
    assert environment.is_done is True


def test_environment_passes_correct_scenario_to_coverage_service(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    scenarios = [
        create_scenario(
            scenario_id="calculate_score_scenario_001",
            path_index=1,
            priority_rank=1,
            score=85,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="calculate_score_scenario_002",
            path_index=2,
            priority_rank=2,
            score=50,
            expected_result="Başarısız",
        ),
    ]

    coverage_results = [
        create_coverage_result(
            source_file=source_file,
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=2,
            missing_line_count=2,
            total_line_count=4,
            covered_branch_count=1,
            missing_branch_count=1,
            total_branch_count=2,
        ),
    ]

    service = create_mock_service(
        scenarios=[scenarios[1]],
        coverage_results=coverage_results,
    )

    mapper = ScenarioActionMapper(
        scenarios
    )

    scenario_transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=scenario_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    environment.step(
        mapper.actions[1]
    )

    service.measure_scenario.assert_called_once()

    called_scenario = (
        service.measure_scenario
        .call_args.kwargs["scenario"]
    )

    assert called_scenario is scenarios[1]


def test_environment_removes_action_after_real_coverage_transition(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="calculate_score_scenario_001",
        path_index=1,
        priority_rank=1,
        score=85,
        expected_result="Başarılı",
    )

    coverage_result = create_coverage_result(
        source_file=source_file,
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=2,
        missing_line_count=2,
        total_line_count=4,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
    )

    service = create_mock_service(
        scenarios=[scenario],
        coverage_results=[coverage_result],
    )

    mapper = ScenarioActionMapper(
        [scenario]
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    selected_action = mapper.actions[0]

    environment.step(
        selected_action
    )

    assert selected_action not in environment.available_actions
    assert environment.available_actions == ()


def test_environment_reset_restores_real_transition_state(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="calculate_score_scenario_001",
        path_index=1,
        priority_rank=1,
        score=85,
        expected_result="Başarılı",
    )

    coverage_result = create_coverage_result(
        source_file=source_file,
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=2,
        missing_line_count=2,
        total_line_count=4,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
    )

    service = create_mock_service(
        scenarios=[scenario],
        coverage_results=[coverage_result],
    )

    mapper = ScenarioActionMapper(
        [scenario]
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=transition,
    )

    initial_state = create_initial_state()

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=mapper.actions,
        transition_function=adapter,
    )

    environment.step(
        mapper.actions[0]
    )

    reset_state = environment.reset()

    assert reset_state is initial_state
    assert environment.current_state is initial_state
    assert environment.available_actions == mapper.actions
    assert environment.is_done is False