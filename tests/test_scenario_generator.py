import re
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer, ExecutionPath
from evaluator.dqm import DQMScore, DecisionQualityMatrix
from generator.path_input_generator import (
    GeneratedTestInput,
    PathInputGenerator,
    UnreachablePathError,
)
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario, ScenarioGenerator

SOURCE_FILE = "datasets/sample_code.py"


def create_scenarios() -> tuple[
    object,
    list,
    list,
    list[Scenario],
]:
    """Testlerde kullanılacak senaryoları üretir."""
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


def create_mock_path(
    path_index: int,
) -> ExecutionPath:
    """İzole ScenarioGenerator testleri için metadata içeren yol üretir."""
    return ExecutionPath(
        node_ids=[
            path_index * 10 + 1,
            path_index * 10 + 2,
            path_index * 10 + 3,
        ],
        edge_labels=[
            None,
            None,
        ],
        node_labels=[
            "START",
            f"return {path_index}",
            "END",
        ],
        node_types=[
            "start",
            "return",
            "end",
        ],
        line_numbers=[
            1,
            2,
            3,
        ],
    )


def create_mock_score(
    path_index: int,
    normalized_score: float,
) -> DQMScore:
    """İzole testlerde kullanılacak DQM sonucu üretir."""
    return DQMScore(
        path_index=path_index,
        path_length=3,
        decision_edge_count=0,
        contains_loop=False,
        contains_exception=False,
        raw_score=normalized_score,
        normalized_score=normalized_score,
        priority_level="High",
    )


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


def test_generate_skips_unreachable_path() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.side_effect = [
        GeneratedTestInput(
            keyword_arguments=(("score", 90),),
            expected_result="Başarılı",
        ),
        UnreachablePathError(
            "Çelişkili yol."
        ),
        GeneratedTestInput(
            keyword_arguments=(("score", 10),),
            expected_result="Başarısız",
        ),
    ]

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    paths = [
        create_mock_path(1),
        create_mock_path(2),
        create_mock_path(3),
    ]

    scores = [
        create_mock_score(1, 100.0),
        create_mock_score(2, 90.0),
        create_mock_score(3, 80.0),
    ]

    scenarios = generator.generate(
        function_name="calculate_score",
        paths=paths,
        scores=scores,
        parameter_names=("score",),
    )

    assert len(scenarios) == 2
    assert [scenario.path_index for scenario in scenarios] == [
        1,
        3,
    ]
    assert generator.skipped_path_indices == (2,)
    assert generator.skipped_path_count == 1


def test_generate_reassigns_continuous_priority_ranks_after_skip() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.side_effect = [
        UnreachablePathError(
            "Çelişkili yol."
        ),
        GeneratedTestInput(
            keyword_arguments=(("score", 75),),
            expected_result="Orta",
        ),
    ]

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    scenarios = generator.generate(
        function_name="calculate_score",
        paths=[
            create_mock_path(1),
            create_mock_path(2),
        ],
        scores=[
            create_mock_score(1, 100.0),
            create_mock_score(2, 90.0),
        ],
        parameter_names=("score",),
    )

    assert len(scenarios) == 1
    assert scenarios[0].priority_rank == 1
    assert scenarios[0].scenario_id == (
        "calculate_score_scenario_001"
    )
    assert scenarios[0].path_index == 2


def test_generate_resets_skipped_paths_between_calls() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.side_effect = [
        UnreachablePathError(
            "Çelişkili yol."
        ),
        GeneratedTestInput(
            keyword_arguments=(("score", 85),),
            expected_result="Başarılı",
        ),
    ]

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    generator.generate(
        function_name="calculate_score",
        paths=[create_mock_path(1)],
        scores=[create_mock_score(1, 100.0)],
        parameter_names=("score",),
    )

    assert generator.skipped_path_indices == (1,)

    generator.generate(
        function_name="calculate_score",
        paths=[create_mock_path(1)],
        scores=[create_mock_score(1, 100.0)],
        parameter_names=("score",),
    )

    assert generator.skipped_path_indices == ()
    assert generator.skipped_path_count == 0


def test_generate_propagates_non_unreachable_path_errors() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )
    path_input_generator.generate.side_effect = ValueError(
        "Desteklenmeyen koşul ifadesi."
    )

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    with pytest.raises(
        ValueError,
        match="Desteklenmeyen koşul ifadesi",
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[create_mock_path(1)],
            scores=[create_mock_score(1, 100.0)],
            parameter_names=("score",),
        )


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

# ============================================================
# ScenarioGenerator candidate_values_by_path integration tests
# ============================================================


def test_generate_forwards_candidate_values_to_matching_path() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.return_value = (
        GeneratedTestInput(
            keyword_arguments=(("score", 90),),
            expected_result=1,
        )
    )

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    generator.generate(
        function_name="calculate_score",
        paths=[create_mock_path(1)],
        scores=[create_mock_score(1, 100.0)],
        parameter_names=("score",),
        candidate_values_by_path={
            1: {
                "score": 90,
            },
        },
    )

    path_input_generator.generate.assert_called_once_with(
        path=create_mock_path(1),
        parameter_names=("score",),
        parameter_types={},
        candidate_values={
            "score": 90,
        },
    )


def test_generate_uses_none_when_path_has_no_candidate_values() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.return_value = (
        GeneratedTestInput(
            keyword_arguments=(("score", 50),),
            expected_result=1,
        )
    )

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    generator.generate(
        function_name="calculate_score",
        paths=[create_mock_path(1)],
        scores=[create_mock_score(1, 100.0)],
        parameter_names=("score",),
        candidate_values_by_path={
            2: {
                "score": 10,
            },
        },
    )

    call_kwargs = (
        path_input_generator
        .generate
        .call_args
        .kwargs
    )

    assert call_kwargs["candidate_values"] is None


def test_generate_forwards_different_candidates_by_path() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.side_effect = [
        GeneratedTestInput(
            keyword_arguments=(("score", 90),),
            expected_result=1,
        ),
        GeneratedTestInput(
            keyword_arguments=(("score", 40),),
            expected_result=2,
        ),
    ]

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    generator.generate(
        function_name="calculate_score",
        paths=[
            create_mock_path(1),
            create_mock_path(2),
        ],
        scores=[
            create_mock_score(1, 100.0),
            create_mock_score(2, 90.0),
        ],
        parameter_names=("score",),
        candidate_values_by_path={
            1: {
                "score": 90,
            },
            2: {
                "score": 40,
            },
        },
    )

    calls = (
        path_input_generator
        .generate
        .call_args_list
    )

    assert calls[0].kwargs[
        "candidate_values"
    ] == {
        "score": 90,
    }

    assert calls[1].kwargs[
        "candidate_values"
    ] == {
        "score": 40,
    }


def test_generate_rejects_invalid_candidate_values_by_path_type() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        TypeError,
        match="candidate_values_by_path bir dict veya None olmalıdır.",
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=(),
            candidate_values_by_path=[  # type: ignore[arg-type]
                (1, {"score": 90}),
            ],
        )


def test_generate_rejects_invalid_candidate_path_index() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "candidate_values_by_path anahtarları "
            "pozitif int path index"
        ),
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=(),
            candidate_values_by_path={
                0: {
                    "score": 90,
                },
            },
        )


def test_generate_rejects_non_dict_candidate_values() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        TypeError,
        match=(
            "candidate_values_by_path değerleri "
            "dict olmalıdır."
        ),
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=(),
            candidate_values_by_path={
                1: 90,  # type: ignore[dict-item]
            },
        )


def test_generate_rejects_empty_candidate_variable_name() -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "candidate_values_by_path içindeki "
            "değişken adları boş olmayan string"
        ),
    ):
        generator.generate(
            function_name="calculate_score",
            paths=[],
            scores=[],
            parameter_names=(),
            candidate_values_by_path={
                1: {
                    "": 90,
                },
            },
        )


def test_real_cfg_derived_average_scenario_passes_concrete_validation(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "derived_average.py"
    source_file.write_text(
        """
def evaluate(items: list[int]) -> str:
    total = 0
    count = 0
    for item in items:
        if item < 0:
            continue
        total += item
        count += 1
    average = total / count
    if average >= 50:
        return "high"
    return "low"
""".strip(),
        encoding="utf-8",
    )
    function = PythonAnalyzer().analyze_file(source_file).functions[0]
    graph = next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(source_file)
        if graph.function_name == "evaluate"
    )
    paths = CFGPathAnalyzer().find_paths(graph, max_visits_per_node=3)
    scores = DecisionQualityMatrix().evaluate_paths(function, paths)
    scenarios = ScenarioGenerator().generate(
        function_name="evaluate",
        paths=paths,
        scores=scores,
        parameter_names=("items",),
        parameter_types={"items": "list[int]"},
    )
    scenario = next(
        scenario for scenario in scenarios if scenario.expected_result == "high"
    )
    target = runpy.run_path(str(source_file))["evaluate"]

    actual = target(**scenario.keyword_argument_dict)

    assert actual == "high"
    assert actual == scenario.expected_result
    assert set(scenario.keyword_argument_dict) == {"items"}


def test_ultracomplex_derived_value_lines_have_concrete_scenarios() -> None:
    source_file = Path("datasets/sample_ultracomplex_code.py")
    function = next(
        function
        for function in PythonAnalyzer().analyze_file(source_file).functions
        if function.name == "process_order"
    )
    graph = next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(source_file)
        if graph.function_name == "process_order"
    )
    paths = CFGPathAnalyzer().find_paths(graph, max_visits_per_node=3)
    scores = DecisionQualityMatrix().evaluate_paths(function, paths)
    scenarios = ScenarioGenerator().generate(
        function_name="process_order",
        paths=paths,
        scores=scores,
        parameter_names=(
            "amount", "stock", "customer_type", "coupon",
            "failed_attempts", "items",
        ),
        parameter_types=function.parameter_types,
    )
    target = runpy.run_path(str(source_file))["process_order"]
    accepted_lines: set[int] = set()

    for scenario in scenarios:
        path = paths[scenario.path_index - 1]
        target_lines = {101, 107} & set(path.line_numbers)
        if not target_lines:
            continue
        actual = target(**scenario.keyword_argument_dict)
        if actual == scenario.expected_result:
            accepted_lines.update(target_lines)

    assert accepted_lines == {101, 107}


def test_real_cfg_round_scenario_matches_execution_and_pytest_assertion(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "round_usage.py"
    source_file.write_text(
        """
def usage(current: int, limit: int) -> float:
    if current < 0:
        return 0.0
    if limit <= 0:
        return 0.0
    percentage = current / limit * 100
    return round(percentage, 2)
""".strip(),
        encoding="utf-8",
    )
    function = PythonAnalyzer().analyze_file(source_file).functions[0]
    graph = ControlFlowGraphBuilder().build_from_file(source_file)[0]
    paths = CFGPathAnalyzer().find_paths(graph, max_visits_per_node=3)
    scores = DecisionQualityMatrix().evaluate_paths(function, paths)
    scenarios = ScenarioGenerator().generate(
        function_name="usage",
        paths=paths,
        scores=scores,
        parameter_names=tuple(function.parameters),
        parameter_types=function.parameter_types,
    )
    dynamic_path_indices = {
        index
        for index, path in enumerate(paths, start=1)
        if any("return round(" in step.node_label for step in path.steps)
    }
    scenario = next(
        scenario
        for scenario in scenarios
        if scenario.path_index in dynamic_path_indices
    )
    target = runpy.run_path(str(source_file))["usage"]

    actual = target(**scenario.keyword_argument_dict)
    generated_pytest = PytestGenerator().generate(
        module_path="round_usage",
        function_name="usage",
        scenarios=[scenario],
    )

    assert actual == scenario.expected_result
    assert f"assert result == {scenario.expected_result!r}" in generated_pytest


def test_robustness_category_usage_round_scenarios_match_execution() -> None:
    source_file = Path("datasets/sample_robustness_code.py")
    function = next(
        function
        for function in PythonAnalyzer().analyze_file(source_file).functions
        if function.name == "calculate_category_usage"
    )
    graph = next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(source_file)
        if graph.function_name == "calculate_category_usage"
    )
    paths = CFGPathAnalyzer().find_paths(graph, max_visits_per_node=3)
    scores = DecisionQualityMatrix().evaluate_paths(function, paths)
    scenarios = ScenarioGenerator().generate(
        function_name=function.name,
        paths=paths,
        scores=scores,
        parameter_names=tuple(function.parameters),
        parameter_types=function.parameter_types,
    )
    dynamic_path_indices = {
        index
        for index, path in enumerate(paths, start=1)
        if any("return round(" in step.node_label for step in path.steps)
    }
    target = runpy.run_path(str(source_file))["calculate_category_usage"]
    dynamic_scenarios = [
        scenario
        for scenario in scenarios
        if scenario.path_index in dynamic_path_indices
    ]

    assert dynamic_scenarios
    assert all(
        target(**scenario.keyword_argument_dict) == scenario.expected_result
        for scenario in dynamic_scenarios
    )
