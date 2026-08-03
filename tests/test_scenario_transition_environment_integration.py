from __future__ import annotations

from generator.scenario_generator import Scenario
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)


def create_scenario(
    *,
    scenario_id: str,
    path_index: int,
    priority_rank: int,
    expected_result: str,
) -> Scenario:
    """Entegrasyon testlerinde kullanılacak Scenario nesnesini üretir."""
    return Scenario(
        scenario_id=scenario_id,
        name=f"Senaryo {scenario_id}",
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 2, 3),
        edge_labels=(None, "True"),
        contains_loop=False,
        contains_exception=False,
        description="Environment entegrasyon testi.",
        keyword_arguments=(("score", 85),),
        expected_result=expected_result,
        expected_exception=None,
    )


def create_initial_state() -> CoverageState:
    """Entegrasyon testinin başlangıç coverage durumunu oluşturur."""
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def test_environment_uses_adapter_to_resolve_selected_scenario() -> None:
    scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=2,
            priority_rank=2,
            expected_result="Orta",
        ),
    ]

    mapper = ScenarioActionMapper(scenarios)

    selected_scenarios: list[Scenario] = []

    def apply_scenario(
        state: CoverageState,
        scenario: Scenario,
    ) -> CoverageState:
        selected_scenarios.append(scenario)

        return CoverageState(
            coverage_percentage=50.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(4, 5, 6),
            uncovered_branches=2,
        )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=apply_scenario,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    result = environment.step(
        mapper.actions[1]
    )

    assert selected_scenarios == [
        scenarios[1],
    ]
    assert result.state.coverage_percentage == 50.0
    assert result.state.executed_tests == 1
    assert environment.current_state is result.state


def test_environment_removes_used_action_after_scenario_transition() -> None:
    scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=2,
            priority_rank=2,
            expected_result="Orta",
        ),
    ]

    mapper = ScenarioActionMapper(scenarios)

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: CoverageState(
                coverage_percentage=50.0,
                executed_tests=state.executed_tests + 1,
                missing_lines=(4, 5, 6),
                uncovered_branches=2,
            )
        ),
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    selected_action = mapper.actions[0]

    environment.step(selected_action)

    assert selected_action not in environment.available_actions
    assert len(environment.available_actions) == 1


def test_environment_calculates_reward_for_scenario_transition() -> None:
    scenario = create_scenario(
        scenario_id="scenario_001",
        path_index=1,
        priority_rank=1,
        expected_result="Başarılı",
    )

    mapper = ScenarioActionMapper([scenario])

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, selected_scenario: CoverageState(
                coverage_percentage=25.0,
                executed_tests=state.executed_tests + 1,
                missing_lines=(3, 4, 5, 6),
                uncovered_branches=3,
            )
        ),
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    result = environment.step(
        mapper.actions[0]
    )

    assert result.reward > 0.0


def test_environment_marks_episode_done_when_all_actions_are_used() -> None:
    scenario = create_scenario(
        scenario_id="scenario_001",
        path_index=1,
        priority_rank=1,
        expected_result="Başarılı",
    )

    mapper = ScenarioActionMapper([scenario])

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, selected_scenario: CoverageState(
                coverage_percentage=25.0,
                executed_tests=state.executed_tests + 1,
                missing_lines=(3, 4, 5, 6),
                uncovered_branches=3,
            )
        ),
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
    )

    result = environment.step(
        mapper.actions[0]
    )

    assert result.done is True
    assert environment.is_done is True
    assert environment.available_actions == ()


def test_environment_reset_restores_mapper_actions() -> None:
    scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=2,
            priority_rank=2,
            expected_result="Orta",
        ),
    ]

    mapper = ScenarioActionMapper(scenarios)

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, selected_scenario: CoverageState(
                coverage_percentage=50.0,
                executed_tests=state.executed_tests + 1,
                missing_lines=(4, 5, 6),
                uncovered_branches=2,
            )
        ),
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