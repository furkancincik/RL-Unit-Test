import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.scenario_generator import TestScenarioGenerator

SOURCE_FILE = "datasets/sample_code.py"


def create_scenarios():
    """Testlerde kullanılacak senaryoları üretir."""
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    scenario_generator = TestScenarioGenerator()

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
    )

    return function, paths, scores, scenarios


def test_generate_creates_scenario_for_each_dqm_score() -> None:
    _, _, scores, scenarios = create_scenarios()

    assert len(scenarios) == len(scores)
    assert len(scenarios) == 3


def test_generate_preserves_dqm_priority_order() -> None:
    _, _, scores, scenarios = create_scenarios()

    assert scenarios[0].priority_rank == 1
    assert scenarios[0].priority_level == scores[0].priority_level
    assert scenarios[0].dqm_score == scores[0].normalized_score

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
    assert first_scenario.path_index == first_score.path_index
    assert first_scenario.node_ids == tuple(first_path.node_ids)
    assert first_scenario.edge_labels == tuple(
        first_path.edge_labels
    )


def test_generate_rejects_empty_function_name() -> None:
    generator = TestScenarioGenerator()

    with pytest.raises(
        ValueError,
        match="Fonksiyon adı boş olamaz",
    ):
        generator.generate(
            function_name=" ",
            paths=[],
            scores=[],
        )