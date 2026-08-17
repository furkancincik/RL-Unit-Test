from __future__ import annotations

from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import PathInputGenerator


def create_path(
    *,
    node_labels: list[str],
    node_types: list[str],
    edge_labels: list[str | None],
) -> ExecutionPath:
    return ExecutionPath(
        node_ids=list(range(1, len(node_labels) + 1)),
        edge_labels=edge_labels,
        node_labels=node_labels,
        node_types=node_types,
        line_numbers=list(range(1, len(node_labels) + 1)),
    )


def test_generate_propagates_loop_variable_constraint_to_items(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "item in items",
            "item < 0",
            "continue",
            "item in items",
            "return 'Geçerli ürün yok'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "continue",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "True",
            None,
            "Complete",
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

    items = result.keyword_argument_dict["items"]

    assert items
    assert all(item < 0 for item in items)
    assert result.expected_result == "Geçerli ürün yok"


def test_generate_propagates_false_loop_condition_to_items(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "item in items",
            "item < 0",
            "total += item",
            "item in items",
            "return 'Geçerli'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "False",
            None,
            "Complete",
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

    items = result.keyword_argument_dict["items"]

    assert items
    assert all(item >= 0 for item in items)


def test_generate_maps_constraints_to_separate_loop_iterations(
) -> None:
    """
    Aynı for düğümünün farklı ziyaretlerindeki koşulların
    iterable'ın karşılık gelen elemanlarına uygulanmasını doğrular.
    """
    path = ExecutionPath(
        node_ids=[
            1,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            6,
            7,
        ],
        edge_labels=[
            None,
            "Iterate",
            "True",
            "Continue",
            "Iterate",
            "False",
            "Next",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "item in items",
            "item < 0",
            "continue",
            "item in items",
            "item < 0",
            "total += item",
            "item in items",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "continue",
            "for",
            "if",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        line_numbers=list(range(1, 11)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("items",),
        parameter_types={
            "items": "list[int]",
        },
    )

    items = result.keyword_argument_dict["items"]

    assert len(items) == 2
    assert items[0] < 0
    assert items[1] >= 0
    assert result.expected_result == "Tamamlandı"


def test_generate_preserves_tuple_iterable_type(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "value in values",
            "value == 'VIP'",
            "value in values",
            "return 'Özel'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "True",
            "Complete",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={
            "values": "tuple[str, ...]",
        },
    )

    values = result.keyword_argument_dict["values"]

    assert isinstance(values, tuple)
    assert values == ("VIP",)


def test_generate_maps_three_iterations_to_distinct_elements(
) -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 7, 8],
        edge_labels=[
            None,
            "Iterate",
            "False",
            "Next",
            "Iterate",
            "True",
            "Continue",
            "Iterate",
            "False",
            "Next",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "value in values",
            "value < 0",
            "total += value",
            "value in values",
            "value < 0",
            "continue",
            "value in values",
            "value < 0",
            "total += value",
            "value in values",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "AugAssign",
            "for",
            "if",
            "continue",
            "for",
            "if",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        line_numbers=list(range(1, 14)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[int]"},
    )

    assert len(result.keyword_argument_dict["values"]) == 3
    assert result.keyword_argument_dict["values"][0] >= 0
    assert result.keyword_argument_dict["values"][1] < 0
    assert result.keyword_argument_dict["values"][2] >= 0


def test_generate_preserves_tuple_type_across_multiple_iterations(
) -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 2, 3, 5, 2, 6, 7],
        edge_labels=[
            None,
            "Iterate",
            "False",
            "Next",
            "Iterate",
            "True",
            "Continue",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "value in values",
            "value == 'VIP'",
            "seen += 1",
            "value in values",
            "value == 'VIP'",
            "continue",
            "value in values",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "AugAssign",
            "for",
            "if",
            "continue",
            "for",
            "return",
            "end",
        ],
        line_numbers=list(range(1, 11)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "tuple[str, ...]"},
    )

    assert result.keyword_argument_dict["values"] == (
        "__generated_value__",
        "VIP",
    )


def test_generate_keeps_same_target_separate_across_for_nodes(
) -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 2, 4, 5, 6, 5, 7, 8],
        edge_labels=[
            None,
            "Iterate",
            "True",
            "Complete",
            None,
            "Iterate",
            "False",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "entry in left_values",
            "entry < 0",
            "entry in left_values",
            "between = 1",
            "entry in right_values",
            "entry < 0",
            "entry in right_values",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "for",
            "Assign",
            "for",
            "if",
            "for",
            "return",
            "end",
        ],
        line_numbers=list(range(1, 11)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("left_values", "right_values"),
        parameter_types={
            "left_values": "list[int]",
            "right_values": "list[int]",
        },
    )

    assert result.keyword_argument_dict["left_values"][0] < 0
    assert result.keyword_argument_dict["right_values"][0] >= 0


def test_generate_restarts_inner_iteration_index_per_outer_activation(
) -> None:
    path = ExecutionPath(
        node_ids=[
            1, 2, 3, 4, 5, 4, 6, 2, 3, 4, 5, 4, 6, 2, 7, 8,
        ],
        edge_labels=[
            None,
            "Iterate",
            "False",
            "Iterate",
            "True",
            "Complete",
            "Next",
            "Iterate",
            "False",
            "Iterate",
            "True",
            "Complete",
            "Next",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "value in outer_values",
            "value >= 0",
            "value in inner_values",
            "value < 0",
            "value in inner_values",
            "count += 1",
            "value in outer_values",
            "value >= 0",
            "value in inner_values",
            "value < 0",
            "value in inner_values",
            "count += 1",
            "value in outer_values",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "if",
            "for",
            "if",
            "for",
            "AugAssign",
            "for",
            "if",
            "for",
            "if",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        line_numbers=list(range(1, 17)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("outer_values", "inner_values"),
        parameter_types={
            "outer_values": "list[int]",
            "inner_values": "list[int]",
        },
    )

    assert len(result.keyword_argument_dict["outer_values"]) == 2
    assert len(result.keyword_argument_dict["inner_values"]) == 1
    assert all(
        value < 0
        for value in result.keyword_argument_dict["inner_values"]
    )
