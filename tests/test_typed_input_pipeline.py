from __future__ import annotations

from unittest.mock import Mock

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore
from generator.path_input_generator import (
    GeneratedTestInput,
    PathInputGenerator,
)
from generator.scenario_generator import ScenarioGenerator


def create_execution_path(
    *,
    node_labels: list[str],
    node_types: list[str],
    edge_labels: list[str | None],
) -> ExecutionPath:
    """Typed-input testleri için metadata içeren yol oluşturur."""
    node_count = len(node_labels)

    return ExecutionPath(
        node_ids=list(range(1, node_count + 1)),
        edge_labels=edge_labels,
        node_labels=node_labels,
        node_types=node_types,
        line_numbers=list(range(1, node_count + 1)),
    )


def create_score() -> DQMScore:
    """ScenarioGenerator aktarım testinde kullanılacak DQM skoru."""
    return DQMScore(
        path_index=1,
        path_length=3,
        decision_edge_count=1,
        contains_loop=False,
        contains_exception=False,
        raw_score=100.0,
        normalized_score=100.0,
        priority_level="High",
    )


def test_analyzer_extracts_parameter_type_hints(
    tmp_path,
) -> None:
    source_file = tmp_path / "typed_sample.py"

    source_file.write_text(
        """
def process(
    amount: float,
    customer_type: str,
    items: list[int],
    enabled: bool,
) -> str:
    return "ok"
""".strip(),
        encoding="utf-8",
    )

    function = PythonAnalyzer().analyze_file(
        source_file
    ).functions[0]

    assert function.parameter_types == {
        "amount": "float",
        "customer_type": "str",
        "items": "list[int]",
        "enabled": "bool",
    }

    assert function.typed_parameter_count == 4


def test_analyzer_omits_untyped_parameters(
    tmp_path,
) -> None:
    source_file = tmp_path / "partly_typed.py"

    source_file.write_text(
        """
def process(score: int, payload):
    return score
""".strip(),
        encoding="utf-8",
    )

    function = PythonAnalyzer().analyze_file(
        source_file
    ).functions[0]

    assert function.parameters == [
        "score",
        "payload",
    ]

    assert function.parameter_types == {
        "score": "int",
    }

    assert function.typed_parameter_count == 1


@pytest.mark.parametrize(
    (
        "parameter_type",
        "expected_value",
    ),
    (
        ("str", ""),
        ("float", 0.0),
        ("bool", False),
        ("list[int]", []),
        ("tuple[int, ...]", ()),
        ("dict[str, int]", {}),
        ("set[int]", set()),
    ),
)
def test_generate_uses_typed_default_values(
    parameter_type: str,
    expected_value: object,
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("value",),
        parameter_types={
            "value": parameter_type,
        },
    )

    assert (
        result.keyword_argument_dict["value"]
        == expected_value
    )


def test_generate_creates_truthy_list_for_false_not_condition(
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "return 'Dolu'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("items",),
        parameter_types={
            "items": "list[int]",
        },
    )

    assert result.keyword_argument_dict["items"] == [0]


def test_generate_creates_empty_list_for_true_not_condition(
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "return 'Boş'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("items",),
        parameter_types={
            "items": "list[int]",
        },
    )

    assert result.keyword_argument_dict["items"] == []


def test_generate_creates_truthy_string_for_name_condition(
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "name",
            "return 'Var'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("name",),
        parameter_types={
            "name": "str",
        },
    )

    assert result.keyword_argument_dict["name"] == "value"


def test_generate_creates_empty_string_for_false_name_condition(
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "name",
            "return 'Yok'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("name",),
        parameter_types={
            "name": "str",
        },
    )

    assert result.keyword_argument_dict["name"] == ""


def test_generate_rejects_unknown_parameter_type_name() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "parameter_types bilinmeyen "
            "parametre içeriyor"
        ),
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("score",),
            parameter_types={
                "items": "list[int]",
            },
        )


def test_scenario_generator_forwards_parameter_types() -> None:
    path_input_generator = Mock(
        spec=PathInputGenerator,
    )

    path_input_generator.generate.return_value = (
        GeneratedTestInput(
            keyword_arguments=(
                ("items", [0]),
            ),
            expected_result="Tamamlandı",
        )
    )

    generator = ScenarioGenerator(
        path_input_generator=path_input_generator,
    )

    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            None,
        ],
    )

    scenarios = generator.generate(
        function_name="process",
        paths=[path],
        scores=[create_score()],
        parameter_names=("items",),
        parameter_types={
            "items": "list[int]",
        },
    )

    path_input_generator.generate.assert_called_once_with(
        path=path,
        parameter_names=("items",),
        parameter_types={
            "items": "list[int]",
        },
        candidate_values=None,
    )

    assert scenarios[0].keyword_argument_dict == {
        "items": [0],
    }


def test_scenario_generator_rejects_unknown_parameter_type(
) -> None:
    generator = ScenarioGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "parameter_types bilinmeyen "
            "parametre içeriyor"
        ),
    ):
        generator.generate(
            function_name="process",
            paths=[],
            scores=[],
            parameter_names=("score",),
            parameter_types={
                "items": "list[int]",
            },
        )