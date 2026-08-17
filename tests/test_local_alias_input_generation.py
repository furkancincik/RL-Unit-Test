from __future__ import annotations

import pytest

from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import (
    PathInputGenerator,
    UnreachablePathError,
)


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


def test_generate_propagates_local_subscript_constraint_to_list(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "not items",
            "first_item = items[0]",
            "first_item < 0",
            "return 'Geçersiz ürün değeri'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "Assign",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
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

    items = result.keyword_argument_dict["items"]

    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[0] < 0
    assert result.expected_result == "Geçersiz ürün değeri"


def test_generate_preserves_required_collection_length(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "item in items",
            "item in items",
            "second_item = items[1]",
            "second_item >= 10",
            "return 'İkinci ürün geçerli'",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "for",
            "Assign",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "Next",
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

    items = result.keyword_argument_dict["items"]

    assert len(items) >= 2
    assert items[1] >= 10


def test_generate_supports_tuple_subscript_alias(
) -> None:
    path = create_path(
        node_labels=[
            "START",
            "first_value = values[0]",
            "first_value == 'VIP'",
            "return 'Özel'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "True",
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
    assert values[0] == "VIP"


def test_generate_preserves_alias_across_later_loop_iteration(
) -> None:
    """
    İlk eleman alias kısıtını sağlarken sonraki iterasyonun farklı
    bir koleksiyon elemanı üzerinde negatif olabilmesini doğrular.
    """
    path = ExecutionPath(
        node_ids=[
            1,
            2,
            3,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            8,
            9,
        ],
        edge_labels=[
            None,
            None,
            "False",
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
            "first_item = items[0]",
            "first_item < 0",
            "item in items",
            "item < 0",
            "valid_item_count += 1",
            "item in items",
            "item < 0",
            "continue",
            "item in items",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "if",
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
        line_numbers=list(range(1, 13)),
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
    assert items[0] >= 0
    assert items[1] < 0
    assert result.expected_result == "Tamamlandı"


def test_generate_rejects_conflict_between_alias_and_loop_item(
) -> None:
    """
    Tek elemanlı/tek iterasyonlu bir yolda items[0] hem
    negatif olmayan hem negatif olmaya zorlanamaz.
    """
    path = create_path(
        node_labels=[
            "START",
            "first_item = items[0]",
            "first_item < 0",
            "item in items",
            "item < 0",
            "continue",
            "item in items",
            "return 'Sonuç'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "if",
            "for",
            "if",
            "continue",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "False",
            "Iterate",
            "True",
            "Continue",
            "Complete",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="alias ve döngü kısıtları çelişiyor",
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("items",),
            parameter_types={
                "items": "list[int]",
            },
        )