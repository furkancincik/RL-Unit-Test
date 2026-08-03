from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_suite_coverage_transition import (
    ScenarioSuiteCoverageTransition,
)
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)
from services.scenario_suite_coverage_service import (
    ScenarioSuiteCoverageResult,
    ScenarioSuiteCoverageService,
)


def create_source_file(
    tmp_path: Path,
) -> Path:
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 85:\n"
        "        return 'Başarılı'\n"
        "    if score >= 50:\n"
        "        return 'Orta'\n"
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
        description="Çok episode kümülatif environment testi.",
        keyword_arguments=(("score", score),),
        expected_result=expected_result,
        expected_exception=None,
    )


def create_scenarios() -> tuple[Scenario, ...]:
    return (
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
            expected_result="Orta",
        ),
    )


def create_initial_state() -> CoverageState:
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def create_coverage_result(
    *,
    line_coverage_percent: float,
    branch_coverage_percent: float,
    covered_line_count: int,
    missing_line_count: int,
    total_line_count: int,
    covered_branch_count: int,
    missing_branch_count: int,
    total_branch_count: int,
) -> CoverageResult:
    return CoverageResult(
        source_file=Path("sample_code.py"),
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
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


def create_suite_result(
    *,
    scenarios: tuple[Scenario, ...],
    coverage: CoverageResult,
) -> ScenarioSuiteCoverageResult:
    return ScenarioSuiteCoverageResult(
        scenarios=scenarios,
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
        coverage=coverage,
    )


def test_environment_reset_clears_cumulative_transition_state(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    first_episode_result = create_suite_result(
        scenarios=(scenarios[0],),
        coverage=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
        ),
    )

    second_episode_result = create_suite_result(
        scenarios=(scenarios[0],),
        coverage=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
        ),
    )

    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )
    service.measure_scenarios.side_effect = [
        first_episode_result,
        second_episode_result,
    ]

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    suite_transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=suite_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
        episode_reset_callback=suite_transition.reset,
    )

    first_step = environment.step(
        mapper.actions[0]
    )

    assert first_step.state.coverage_percentage == 50.0
    assert suite_transition.selected_scenarios == (
        scenarios[0],
    )

    reset_state = environment.reset()

    assert reset_state.coverage_percentage == 0.0
    assert reset_state.executed_tests == 0
    assert suite_transition.selected_scenarios == ()
    assert environment.available_actions == mapper.actions

    second_step = environment.step(
        mapper.actions[0]
    )

    assert second_step.state.coverage_percentage == 50.0
    assert second_step.state.executed_tests == 1
    assert suite_transition.selected_scenarios == (
        scenarios[0],
    )


def test_each_episode_rebuilds_cumulative_suite_from_empty(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    single_scenario_result = create_suite_result(
        scenarios=(scenarios[1],),
        coverage=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
        ),
    )

    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )
    service.measure_scenarios.side_effect = [
        single_scenario_result,
        single_scenario_result,
    ]

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    suite_transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=suite_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
        episode_reset_callback=suite_transition.reset,
    )

    environment.step(
        mapper.actions[1]
    )

    first_episode_call = (
        service.measure_scenarios
        .call_args_list[0]
        .kwargs["scenarios"]
    )

    environment.reset()

    environment.step(
        mapper.actions[1]
    )

    second_episode_call = (
        service.measure_scenarios
        .call_args_list[1]
        .kwargs["scenarios"]
    )

    assert first_episode_call == (
        scenarios[1],
    )
    assert second_episode_call == (
        scenarios[1],
    )


def test_environment_can_complete_two_separate_episodes(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    full_coverage_result = create_suite_result(
        scenarios=(scenarios[0],),
        coverage=create_coverage_result(
            line_coverage_percent=100.0,
            branch_coverage_percent=100.0,
            covered_line_count=6,
            missing_line_count=0,
            total_line_count=6,
            covered_branch_count=4,
            missing_branch_count=0,
            total_branch_count=4,
        ),
    )

    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )
    service.measure_scenarios.side_effect = [
        full_coverage_result,
        full_coverage_result,
    ]

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    suite_transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=suite_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
        episode_reset_callback=suite_transition.reset,
    )

    first_episode_step = environment.step(
        mapper.actions[0]
    )

    assert first_episode_step.done is True
    assert environment.is_done is True

    environment.reset()

    assert environment.is_done is False

    second_episode_step = environment.step(
        mapper.actions[0]
    )

    assert second_episode_step.done is True
    assert environment.is_done is True