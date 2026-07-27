from __future__ import annotations

import pytest

from cfg.control_flow_graph import (
    CFGEdge,
    CFGNode,
    ControlFlowGraph,
)
from cfg.path_analyzer import ExecutionPath
from generator.input_generator import TestInputGenerator


def create_graph_and_path(
    condition: str,
    branch: str,
    node_type: str = "if",
) -> tuple[ControlFlowGraph, ExecutionPath]:
    """Verilen koşul ve dal yönü için örnek CFG ve yürütme yolu oluşturur."""

    graph = ControlFlowGraph(
        function_name="check_value",
        nodes=[
            CFGNode(
                node_id=1,
                label="START",
                node_type="start",
            ),
            CFGNode(
                node_id=2,
                label=condition,
                node_type=node_type,
                line_number=2,
            ),
            CFGNode(
                node_id=3,
                label="return True",
                node_type="return",
                line_number=3,
            ),
            CFGNode(
                node_id=4,
                label="END",
                node_type="end",
            ),
        ],
        edges=[
            CFGEdge(
                source_id=1,
                target_id=2,
            ),
            CFGEdge(
                source_id=2,
                target_id=3,
                label=branch,
            ),
            CFGEdge(
                source_id=3,
                target_id=4,
            ),
        ],
    )

    path = ExecutionPath(
        node_ids=[1, 2, 3, 4],
        edge_labels=[None, branch, None],
    )

    return graph, path


def test_generate_true_branch_for_greater_equal_condition() -> None:
    graph, path = create_graph_and_path(
        condition="score >= 50",
        branch="True",
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "score"
    assert generated_input.value == 50
    assert generated_input.condition == "score >= 50"
    assert generated_input.branch == "True"


def test_generate_false_branch_for_greater_equal_condition() -> None:
    graph, path = create_graph_and_path(
        condition="score >= 50",
        branch="False",
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "score"
    assert generated_input.value == 49
    assert generated_input.condition == "score >= 50"
    assert generated_input.branch == "False"


@pytest.mark.parametrize(
    ("condition", "branch", "expected_value"),
    [
        ("x > 10", "True", 11),
        ("x > 10", "False", 10),
        ("x >= 10", "True", 10),
        ("x >= 10", "False", 9),
        ("x < 10", "True", 9),
        ("x < 10", "False", 10),
        ("x <= 10", "True", 10),
        ("x <= 10", "False", 11),
        ("x == 10", "True", 10),
        ("x == 10", "False", 11),
        ("x != 10", "True", 11),
        ("x != 10", "False", 10),
    ],
)
def test_generate_values_for_supported_comparison_operators(
    condition: str,
    branch: str,
    expected_value: int,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "x"
    assert generated_input.value == expected_value
    assert generated_input.condition == condition
    assert generated_input.branch == branch


@pytest.mark.parametrize(
    "condition",
    [
        "x > y",
        "items[0] > 10",
        "x > 1 < 20",
    ],
)

def test_generate_returns_empty_result_for_unsupported_conditions(
    condition: str,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch="True",
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert result.values == ()


def test_generate_returns_empty_result_for_invalid_condition_syntax() -> None:
    graph, path = create_graph_and_path(
        condition="x >",
        branch="True",
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert result.values == ()


def test_generate_input_for_while_condition() -> None:
    graph, path = create_graph_and_path(
        condition="counter < 5",
        branch="True",
        node_type="while",
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "counter"
    assert generated_input.value == 4
    assert generated_input.condition == "counter < 5"
    assert generated_input.branch == "True"

def test_generate_inputs_for_multiple_conditions_in_same_path() -> None:
    graph = ControlFlowGraph(
        function_name="check_values",
        nodes=[
            CFGNode(
                node_id=1,
                label="START",
                node_type="start",
            ),
            CFGNode(
                node_id=2,
                label="x > 10",
                node_type="if",
                line_number=2,
            ),
            CFGNode(
                node_id=3,
                label="y < 5",
                node_type="if",
                line_number=3,
            ),
            CFGNode(
                node_id=4,
                label="return True",
                node_type="return",
                line_number=4,
            ),
            CFGNode(
                node_id=5,
                label="END",
                node_type="end",
            ),
        ],
        edges=[
            CFGEdge(
                source_id=1,
                target_id=2,
            ),
            CFGEdge(
                source_id=2,
                target_id=3,
                label="True",
            ),
            CFGEdge(
                source_id=3,
                target_id=4,
                label="False",
            ),
            CFGEdge(
                source_id=4,
                target_id=5,
            ),
        ],
    )

    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5],
        edge_labels=[None, "True", "False", None],
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 2

    first_input = result.values[0]
    second_input = result.values[1]

    assert first_input.parameter_name == "x"
    assert first_input.value == 11
    assert first_input.condition == "x > 10"
    assert first_input.branch == "True"

    assert second_input.parameter_name == "y"
    assert second_input.value == 5
    assert second_input.condition == "y < 5"
    assert second_input.branch == "False"


@pytest.mark.parametrize(
    ("condition", "branch", "expected_value"),
    [
        ("10 < x", "True", 11),
        ("10 < x", "False", 10),
        ("10 <= x", "True", 10),
        ("10 <= x", "False", 9),
        ("10 > x", "True", 9),
        ("10 > x", "False", 10),
        ("10 >= x", "True", 10),
        ("10 >= x", "False", 11),
        ("10 == x", "True", 10),
        ("10 == x", "False", 11),
        ("10 != x", "True", 11),
        ("10 != x", "False", 10),
    ],
)
def test_generate_values_when_constant_is_on_left(
    condition: str,
    branch: str,
    expected_value: int,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "x"
    assert generated_input.value == expected_value
    assert generated_input.condition == condition
    assert generated_input.branch == branch

@pytest.mark.parametrize(
    ("condition", "branch", "expected_value"),
    [
        ("x > -5", "True", -4),
        ("x > -5", "False", -5),
        ("x >= -5", "True", -5),
        ("x >= -5", "False", -6),
        ("x < -5", "True", -6),
        ("x < -5", "False", -5),
        ("x <= -5", "True", -5),
        ("x <= -5", "False", -4),
        ("x == -5", "True", -5),
        ("x == -5", "False", -4),
        ("x != -5", "True", -4),
        ("x != -5", "False", -5),
    ],
)
def test_generate_values_for_negative_numeric_constants(
    condition: str,
    branch: str,
    expected_value: int,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "x"
    assert generated_input.value == expected_value
    assert generated_input.condition == condition
    assert generated_input.branch == branch


@pytest.mark.parametrize(
    (
        "condition",
        "branch",
        "expected_parameter",
        "expected_value",
    ),
    [
        ("is_active == True", "True", "is_active", True),
        ("is_active == True", "False", "is_active", False),
        ("is_active != True", "True", "is_active", False),
        ("is_active != True", "False", "is_active", True),
        ("is_active == False", "True", "is_active", False),
        ("is_active == False", "False", "is_active", True),
        ("is_active != False", "True", "is_active", True),
        ("is_active != False", "False", "is_active", False),
    ],
)
def test_generate_values_for_boolean_constants(
    condition: str,
    branch: str,
    expected_parameter: str,
    expected_value: bool,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == expected_parameter
    assert generated_input.value is expected_value
    assert generated_input.condition == condition
    assert generated_input.branch == branch


@pytest.mark.parametrize(
    (
        "condition",
        "branch",
        "expected_value",
    ),
    [
        ("is_active", "True", True),
        ("is_active", "False", False),
        ("not is_active", "True", False),
        ("not is_active", "False", True),
    ],
)
def test_generate_values_for_direct_boolean_conditions(
    condition: str,
    branch: str,
    expected_value: bool,
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    assert len(result.values) == 1

    generated_input = result.values[0]

    assert generated_input.parameter_name == "is_active"
    assert generated_input.value is expected_value
    assert generated_input.condition == condition
    assert generated_input.branch == branch


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        (
            "True",
            {
                "x": True,
                "y": True,
            },
        ),
        (
            "False",
            {
                "x": False,
                "y": False,
            },
        ),
    ],
)
def test_generate_values_for_direct_boolean_and_condition(
    branch: str,
    expected: dict[str, bool],
) -> None:
    graph, path = create_graph_and_path(
        condition="x and y",
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    generated = {
        item.parameter_name: item.value
        for item in result.values
    }

    assert generated == expected


@pytest.mark.parametrize(
    (
        "condition",
        "branch",
        "expected",
    ),
    [
        (
            "x > 10 and y < 5",
            "True",
            {
                "x": 11,
                "y": 4,
            },
        ),
        (
            "x > 10 and y < 5",
            "False",
            {
                "x": 10,
                "y": 5,
            },
        ),
    ],
)
def test_generate_values_for_and_condition(
    condition: str,
    branch: str,
    expected: dict[str, int],
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    generated = {
        item.parameter_name: item.value
        for item in result.values
    }

    assert generated == expected

@pytest.mark.parametrize(
    ("condition", "branch", "expected"),
    [
        (
            "x > 10 and flag",
            "True",
            {
                "x": 11,
                "flag": True,
            },
        ),
        (
            "x > 10 and flag",
            "False",
            {
                "x": 10,
                "flag": False,
            },
        ),
        (
            "x > 10 and not flag",
            "True",
            {
                "x": 11,
                "flag": False,
            },
        ),
        (
            "x > 10 and not flag",
            "False",
            {
                "x": 10,
                "flag": True,
            },
        ),
    ],
)
def test_generate_values_for_mixed_and_condition(
    condition: str,
    branch: str,
    expected: dict[str, object],
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    generated = {
        item.parameter_name: item.value
        for item in result.values
    }

    assert generated == expected


@pytest.mark.parametrize(
    ("condition", "branch", "expected"),
    [
        (
            "x > 10 or y < 5",
            "True",
            {
                "x": 11,
                "y": 5,
            },
        ),
        (
            "x > 10 or y < 5",
            "False",
            {
                "x": 10,
                "y": 5,
            },
        ),
    ],
)
def test_generate_values_for_or_condition(
    condition: str,
    branch: str,
    expected: dict[str, int],
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    generated = {
        item.parameter_name: item.value
        for item in result.values
    }

    assert generated == expected


@pytest.mark.parametrize(
    ("condition", "branch", "expected"),
    [
        (
            "flag or is_admin",
            "True",
            {
                "flag": True,
                "is_admin": False,
            },
        ),
        (
            "flag or is_admin",
            "False",
            {
                "flag": False,
                "is_admin": False,
            },
        ),
        (
            "not flag or is_admin",
            "True",
            {
                "flag": False,
                "is_admin": False,
            },
        ),
        (
            "not flag or is_admin",
            "False",
            {
                "flag": True,
                "is_admin": False,
            },
        ),
    ],
)
def test_generate_values_for_boolean_or_condition(
    condition: str,
    branch: str,
    expected: dict[str, bool],
) -> None:
    graph, path = create_graph_and_path(
        condition=condition,
        branch=branch,
    )

    result = TestInputGenerator().generate(
        graph=graph,
        path=path,
    )

    generated = {
        item.parameter_name: item.value
        for item in result.values
    }

    assert generated == expected