from __future__ import annotations

import pytest

from generator.scenario_generator import Scenario
from rl.action import Action
from rl.coverage_state import CoverageState
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)


def create_scenario(
    *,
    scenario_id: str = "scenario_001",
    path_index: int = 1,
    priority_rank: int = 1,
) -> Scenario:
    """
    ScenarioTransitionAdapter testlerinde kullanılacak
    örnek Scenario nesnesini oluşturur.
    """
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
        description="Adapter testi için oluşturuldu.",
        keyword_arguments=(("score", 85),),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_state() -> CoverageState:
    """
    Testlerde kullanılacak başlangıç coverage durumunu oluşturur.
    """
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def test_adapter_resolves_action_to_scenario() -> None:
    scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=2,
            priority_rank=2,
        ),
    ]

    mapper = ScenarioActionMapper(scenarios)

    received_scenarios: list[Scenario] = []

    def transition_function(
        state: CoverageState,
        scenario: Scenario,
    ) -> CoverageState:
        received_scenarios.append(scenario)

        return CoverageState(
            coverage_percentage=25.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(3, 4, 5, 6),
            uncovered_branches=3,
        )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=transition_function,
    )

    result = adapter(
        create_state(),
        mapper.actions[1],
    )

    assert received_scenarios == [
        scenarios[1],
    ]
    assert result.coverage_percentage == 25.0
    assert result.executed_tests == 1


def test_adapter_passes_current_state_to_transition() -> None:
    scenario = create_scenario()
    mapper = ScenarioActionMapper([scenario])
    initial_state = create_state()

    received_states: list[CoverageState] = []

    def transition_function(
        state: CoverageState,
        selected_scenario: Scenario,
    ) -> CoverageState:
        received_states.append(state)

        return CoverageState(
            coverage_percentage=10.0,
            executed_tests=1,
            missing_lines=(2, 3, 4, 5, 6),
            uncovered_branches=4,
        )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=transition_function,
    )

    adapter(
        initial_state,
        mapper.actions[0],
    )

    assert received_states == [
        initial_state,
    ]


def test_adapter_returns_transition_result() -> None:
    scenario = create_scenario()
    mapper = ScenarioActionMapper([scenario])

    expected_state = CoverageState(
        coverage_percentage=50.0,
        executed_tests=1,
        missing_lines=(4, 5, 6),
        uncovered_branches=2,
    )

    def transition_function(
        state: CoverageState,
        selected_scenario: Scenario,
    ) -> CoverageState:
        return expected_state

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=transition_function,
    )

    result = adapter(
        create_state(),
        mapper.actions[0],
    )

    assert result is expected_state


def test_adapter_exposes_mapper() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: state
        ),
    )

    assert adapter.mapper is mapper


def test_adapter_rejects_invalid_mapper() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "mapper must be a "
            "ScenarioActionMapper instance"
        ),
    ):
        ScenarioTransitionAdapter(
            mapper="invalid",  # type: ignore[arg-type]
            transition_function=(
                lambda state, scenario: state
            ),
        )


def test_adapter_rejects_non_callable_transition() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    with pytest.raises(
        TypeError,
        match="transition_function must be callable",
    ):
        ScenarioTransitionAdapter(
            mapper=mapper,
            transition_function="invalid",  # type: ignore[arg-type]
        )


def test_adapter_rejects_invalid_state() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: state
        ),
    )

    with pytest.raises(
        TypeError,
        match="state must be a CoverageState instance",
    ):
        adapter(
            "invalid",  # type: ignore[arg-type]
            mapper.actions[0],
        )


def test_adapter_rejects_invalid_action() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: state
        ),
    )

    with pytest.raises(
        TypeError,
        match="action must be an Action instance",
    ):
        adapter(
            create_state(),
            "invalid",  # type: ignore[arg-type]
        )


def test_adapter_rejects_unmapped_action() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: state
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "action does not correspond "
            "to a mapped scenario"
        ),
    ):
        adapter(
            create_state(),
            Action(scenario_index=5),
        )


def test_adapter_rejects_invalid_transition_result() -> None:
    mapper = ScenarioActionMapper(
        [create_scenario()]
    )

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=(
            lambda state, scenario: "invalid"
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "transition_function must return "
            "a CoverageState instance"
        ),
    ):
        adapter(
            create_state(),
            mapper.actions[0],
        )