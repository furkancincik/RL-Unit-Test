import pytest

from generator.scenario_generator import Scenario
from rl.action import Action
from rl.scenario_action_mapper import ScenarioActionMapper


def create_scenario(
    scenario_id: str = "sample_scenario_001",
    name: str = "Sample scenario",
    path_index: int = 1,
    priority_rank: int = 1,
    priority_level: str = "High",
    dqm_score: float = 0.9,
    node_ids: tuple[int, ...] = (1, 2, 3),
    edge_labels: tuple[str, ...] = ("start", "true"),
    contains_loop: bool = False,
    contains_exception: bool = False,
    description: str = "Sample scenario description.",
    keyword_arguments: tuple[tuple[str, object], ...] = (),
    expected_result: object = None,
    expected_exception: str | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        name=name,
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level=priority_level,
        dqm_score=dqm_score,
        node_ids=node_ids,
        edge_labels=edge_labels,
        contains_loop=contains_loop,
        contains_exception=contains_exception,
        description=description,
        keyword_arguments=keyword_arguments,
        expected_result=expected_result,
        expected_exception=expected_exception,
    )


def create_scenarios() -> tuple[Scenario, ...]:
    return (
        create_scenario(
            scenario_id="sample_scenario_001",
            name="First scenario",
            path_index=1,
            priority_rank=1,
            dqm_score=0.9,
        ),
        create_scenario(
            scenario_id="sample_scenario_002",
            name="Second scenario",
            path_index=2,
            priority_rank=2,
            priority_level="Medium",
            dqm_score=0.6,
            node_ids=(1, 4, 5),
            edge_labels=("start", "false"),
        ),
        create_scenario(
            scenario_id="sample_scenario_003",
            name="Third scenario",
            path_index=3,
            priority_rank=3,
            priority_level="Low",
            dqm_score=0.3,
            node_ids=(1, 6, 7),
            edge_labels=("start", "loop"),
            contains_loop=True,
        ),
    )


def test_mapper_can_be_created() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    assert len(mapper) == 3
    assert mapper.scenarios == scenarios


def test_mapper_accepts_list_of_scenarios() -> None:
    scenarios = list(create_scenarios())

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    assert mapper.scenarios == tuple(scenarios)


def test_mapper_accepts_generator_of_scenarios() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=(
            scenario
            for scenario in scenarios
        ),
    )

    assert mapper.scenarios == scenarios


def test_mapper_creates_zero_based_actions() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    assert mapper.actions == (
        Action(scenario_index=0),
        Action(scenario_index=1),
        Action(scenario_index=2),
    )


def test_mapper_action_count_matches_scenario_count() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    assert len(mapper.actions) == len(mapper.scenarios)


def test_mapper_returns_first_scenario_for_first_action() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_scenario(
        action=Action(scenario_index=0),
    )

    assert result == scenarios[0]


def test_mapper_returns_middle_scenario() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_scenario(
        action=Action(scenario_index=1),
    )

    assert result == scenarios[1]


def test_mapper_returns_last_scenario_for_last_action() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_scenario(
        action=Action(scenario_index=2),
    )

    assert result == scenarios[2]


def test_mapper_returns_action_for_first_scenario() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_action(
        scenario=scenarios[0],
    )

    assert result == Action(scenario_index=0)


def test_mapper_returns_action_for_middle_scenario() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_action(
        scenario=scenarios[1],
    )

    assert result == Action(scenario_index=1)


def test_mapper_returns_action_for_last_scenario() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    result = mapper.get_action(
        scenario=scenarios[2],
    )

    assert result == Action(scenario_index=2)


def test_mapper_round_trip_from_scenario_to_action() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    for scenario in scenarios:
        action = mapper.get_action(
            scenario=scenario,
        )

        mapped_scenario = mapper.get_scenario(
            action=action,
        )

        assert mapped_scenario == scenario


def test_mapper_round_trip_from_action_to_scenario() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    for action in mapper.actions:
        scenario = mapper.get_scenario(
            action=action,
        )

        mapped_action = mapper.get_action(
            scenario=scenario,
        )

        assert mapped_action == action


def test_mapper_contains_registered_scenario() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    assert scenarios[0] in mapper
    assert scenarios[1] in mapper
    assert scenarios[2] in mapper


def test_mapper_does_not_contain_unregistered_scenario() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    unknown_scenario = create_scenario(
        scenario_id="unknown_scenario",
        name="Unknown scenario",
        path_index=99,
    )

    assert unknown_scenario not in mapper


def test_mapper_contains_returns_false_for_invalid_type() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    assert "invalid" not in mapper
    assert 1 not in mapper
    assert None not in mapper


def test_empty_mapper_can_be_created() -> None:
    mapper = ScenarioActionMapper(
        scenarios=(),
    )

    assert len(mapper) == 0
    assert mapper.scenarios == ()
    assert mapper.actions == ()


def test_empty_mapper_does_not_contain_scenario() -> None:
    mapper = ScenarioActionMapper(
        scenarios=(),
    )

    assert create_scenario() not in mapper


def test_empty_mapper_rejects_action_lookup() -> None:
    mapper = ScenarioActionMapper(
        scenarios=(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "action does not correspond "
            "to a mapped scenario."
        ),
    ):
        mapper.get_scenario(
            action=Action(scenario_index=0),
        )


def test_mapper_rejects_non_iterable_scenarios() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "scenarios must be an iterable of "
            "Scenario instances."
        ),
    ):
        ScenarioActionMapper(
            scenarios=123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "scenarios",
    [
        ("invalid",),
        (1,),
        (None,),
        (
            create_scenario(),
            "invalid",
        ),
    ],
)
def test_mapper_rejects_invalid_scenario_items(
    scenarios: tuple[object, ...],
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "scenarios must contain only "
            "Scenario instances."
        ),
    ):
        ScenarioActionMapper(
            scenarios=scenarios,  # type: ignore[arg-type]
        )


def test_mapper_rejects_duplicate_scenario_ids() -> None:
    scenario = create_scenario()

    with pytest.raises(
        ValueError,
        match="scenarios cannot contain duplicate scenario_id values.",
    ):
        ScenarioActionMapper(
            scenarios=(
                scenario,
                scenario,
            ),
        )


def test_mapper_rejects_equal_scenarios_with_duplicate_ids() -> None:
    first_scenario = create_scenario()
    second_scenario = create_scenario()

    assert first_scenario is not second_scenario
    assert first_scenario == second_scenario

    with pytest.raises(
        ValueError,
        match="scenarios cannot contain duplicate scenario_id values.",
    ):
        ScenarioActionMapper(
            scenarios=(
                first_scenario,
                second_scenario,
            ),
        )


@pytest.mark.parametrize(
    "action",
    [
        "invalid",
        0,
        None,
        object(),
    ],
)
def test_get_scenario_rejects_invalid_action_type(
    action: object,
) -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    with pytest.raises(
        TypeError,
        match="action must be an Action instance.",
    ):
        mapper.get_scenario(
            action=action,  # type: ignore[arg-type]
        )


def test_get_scenario_rejects_out_of_range_action() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "action does not correspond "
            "to a mapped scenario."
        ),
    ):
        mapper.get_scenario(
            action=Action(scenario_index=3),
        )


def test_get_scenario_rejects_much_larger_action_index() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "action does not correspond "
            "to a mapped scenario."
        ),
    ):
        mapper.get_scenario(
            action=Action(scenario_index=100),
        )


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid",
        0,
        None,
        object(),
    ],
)
def test_get_action_rejects_invalid_scenario_type(
    scenario: object,
) -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "scenario must be a "
            "Scenario instance."
        ),
    ):
        mapper.get_action(
            scenario=scenario,  # type: ignore[arg-type]
        )


def test_get_action_rejects_unregistered_scenario() -> None:
    mapper = ScenarioActionMapper(
        scenarios=create_scenarios(),
    )

    unknown_scenario = create_scenario(
        scenario_id="unknown_scenario_999",
        name="Unknown scenario",
        path_index=999,
        priority_rank=999,
    )

    with pytest.raises(
        ValueError,
        match=(
            "scenario is not registered "
            "in this mapper."
        ),
    ):
        mapper.get_action(
            scenario=unknown_scenario,
        )


def test_mapper_preserves_original_scenario_order() -> None:
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=reversed(scenarios),
    )

    assert mapper.scenarios == tuple(
        reversed(scenarios)
    )

    assert mapper.get_scenario(
        action=Action(scenario_index=0),
    ) == scenarios[2]


def test_mapper_is_not_affected_by_original_list_changes() -> None:
    scenarios = list(create_scenarios())

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    scenarios.clear()

    assert len(mapper) == 3
    assert len(mapper.scenarios) == 3
    assert len(mapper.actions) == 3


def test_mapper_accepts_scenario_with_list_argument() -> None:
    scenario = create_scenario(
        scenario_id="list_scenario_001",
        keyword_arguments=(
            ("items", [10, 20, 30]),
        ),
        expected_result="Başarılı",
    )

    mapper = ScenarioActionMapper(
        scenarios=(scenario,),
    )

    assert len(mapper) == 1
    assert mapper.get_action(scenario) == Action(
        scenario_index=0
    )
    assert mapper.get_scenario(
        Action(scenario_index=0)
    ) is scenario
    assert scenario in mapper


def test_mapper_accepts_scenario_with_dictionary_result() -> None:
    scenario = create_scenario(
        scenario_id="dictionary_scenario_001",
        keyword_arguments=(
            ("payload", {"score": 85}),
        ),
        expected_result={
            "status": "accepted",
            "details": ["high", "priority"],
        },
    )

    mapper = ScenarioActionMapper(
        scenarios=(scenario,),
    )

    assert mapper.get_action(scenario) == Action(
        scenario_index=0
    )
    assert scenario in mapper


def test_mapper_uses_scenario_id_for_equivalent_lookup() -> None:
    registered_scenario = create_scenario(
        scenario_id="shared_scenario_001",
        keyword_arguments=(
            ("items", [1, 2]),
        ),
    )
    equivalent_reference = create_scenario(
        scenario_id="shared_scenario_001",
        name="Different display name",
        path_index=99,
        keyword_arguments=(
            ("items", [999]),
        ),
    )

    mapper = ScenarioActionMapper(
        scenarios=(registered_scenario,),
    )

    assert mapper.get_action(
        equivalent_reference
    ) == Action(scenario_index=0)
    assert equivalent_reference in mapper


def test_mapper_rejects_different_scenarios_with_same_id() -> None:
    first_scenario = create_scenario(
        scenario_id="duplicate_id",
        name="First",
        keyword_arguments=(
            ("items", [1]),
        ),
    )
    second_scenario = create_scenario(
        scenario_id="duplicate_id",
        name="Second",
        path_index=2,
        keyword_arguments=(
            ("items", [2]),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "scenarios cannot contain duplicate "
            "scenario_id values."
        ),
    ):
        ScenarioActionMapper(
            scenarios=(
                first_scenario,
                second_scenario,
            ),
        )
