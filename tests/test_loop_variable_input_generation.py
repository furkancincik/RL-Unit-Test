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