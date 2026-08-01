import re

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.scenario_generator import Scenario, ScenarioGenerator

SOURCE_FILE = "datasets/sample_code.py"


def create_scenarios() -> tuple[
    object,
    list,
    list,
    list[Scenario],
]:
    """
    Testlerde kullanılacak senaryoları üretir.
    """
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    scenario_generator = ScenarioGenerator()

    function = analyzer.analyze_file(
        SOURCE_FILE
    ).functions[0]

    graph = cfg_builder.build_from_file(
        SOURCE_FILE
    )[0]

    paths = path_analyzer.find_paths(graph)

    scores = dqm.evaluate_paths(
        function=function,
        paths=paths,
    )

    scenarios = scenario_generator.generate(
        function_name=function.name,
        paths=paths,
        scores=scores,
        parameter_names=tuple(function.parameters),
    )

    return function, paths, scores, scenarios


def test_generate_creates_scenario_for_each_dqm_score() -> None:
    _, _, scores, scenarios = create_scenarios()

    assert len(scenarios) == len(scores)
    assert len(scenarios) == 3


def test_generate_preserves_dqm_priority_order() -> None:
    _, _, scores, scenarios = create_scenarios()

    assert scenarios[0].priority_rank == 1
    assert scenarios[0].priority_level == (
        scores[0].priority_level
    )
    assert scenarios[0].dqm_score == (
        scores[0].normalized_score
    )

    assert scenarios[1].priority_rank == 2
    assert scenarios[2].priority_rank == 3


def test_generate_maps_execution_path_information() -> None:
    function, paths, scores, scenarios = create_scenarios()

    first_score = scores[0]
    first_path = paths[first_score.path_index - 1]
    first_scenario = scenarios[0]

    assert first_scenario.scenario_id == (
        "calculate_score_scenario_001"
    )

    assert function.name in first_scenario.name

    assert first_scenario.path_index == (
        first_score.path_index
    )

    assert first_scenario.node_ids == tuple(
        first_path.node_ids
    )

    assert first_scenario.edge_labels == tuple(
        first_path.edge_labels
    )


def test_generate_creates_real_input_for_each_scenario() -> None:
    _, _, _, scenarios = create_scenarios()

    assert all(
        scenario.keyword_arguments
        for scenario in scenarios
    )

    assert all(
        "score" in scenario.keyword_argument_dict
        for scenario in scenarios
    )


def test_generate_creates_expected_result_for_each_path() -> None:
    _, _, _, scenarios = create_scenarios()

    result_by_path = {
        scenario.path_index: scenario.expected_result
        for scenario in scenarios
    }

    assert result_by_path[1] == "Başarılı"
    assert result_by_path[2] == "Orta"
    assert result_by_path[3] == "Başarısız"


def test_generate_creates_inputs_matching_each_path() -> None:
    _, _, _, scenarios = create_scenarios()

    scenario_by_path = {
        scenario.path_index: scenario
        for scenario in scenarios
    }

    success_score = (
        scenario_by_path[1]
        .keyword_argument_dict["score"]
    )

    middle_score = (
        scenario_by_path[2]
        .keyword_argument_dict["score"]
    )

    failure_score = (
        scenario_by_path[3]
        .keyword_argument_dict["score"]
    )

    assert success_score >= 85

    assert middle_score < 85
    assert middle_score >= 50

    assert failure_score < 50


def test_generate_scenario_does_not_expect_exception() -> None:
    _, _, _, scenarios = create_scenarios()

    assert all(
        scenario.expected_exception is None
        for scenario in scenarios
    )

    assert all(
        scenario.expects_exception is False
        for scenario in scenarios
    )


def test_generate_includes_expected_result_in_description() -> None:
    _, _, _, scenarios = create_scenarios()

    for scenario in scenarios:
        assert "beklenen sonuç" in scenario.description


def test_generate_rejects_empty_function_name() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Fonksiyon adı boş olamaz."
        ),
    ):
        generator.generate(
            function_name=" ",
            paths=[],
            scores=[],
            parameter_names=(),
        )


def test_generate_rejects_non_tuple_parameter_names() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        TypeError,
        match="parameter_names bir tuple olmalıdır",
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=["score"],  # type: ignore[arg-type]
        )


def test_generate_rejects_empty_parameter_name() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "parameter_names yalnızca boş olmayan "
            "string değerler içermelidir"
        ),
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=("",),
        )


def test_generate_rejects_duplicate_parameter_names() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "parameter_names tekrar eden değer "
            "içeremez"
        ),
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=(
                "score",
                "score",
            ),
        )