from __future__ import annotations

import pytest

from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import PathInputGenerator


def create_execution_path(
    *,
    node_labels: list[str],
    node_types: list[str],
    edge_labels: list[str | None],
) -> ExecutionPath:
    """
    PathInputGenerator testlerinde kullanılacak yürütme yolunu oluşturur.
    """
    node_count = len(node_labels)

    return ExecutionPath(
        node_ids=list(range(1, node_count + 1)),
        edge_labels=edge_labels,
        node_labels=node_labels,
        node_types=node_types,
        line_numbers=list(range(1, node_count + 1)),
    )


def test_generate_creates_input_for_true_greater_equal_path() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 85",
            'return "Başarılı"',
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

    result = generator.generate(
        path=path,
        parameter_names=("score",),
    )

    assert result.keyword_argument_dict == {
        "score": 85,
    }
    assert result.expected_result == "Başarılı"
    assert result.expected_exception is None


def test_generate_combines_two_numeric_conditions() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 85",
            "score >= 50",
            'return "Orta"',
            "END",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            "True",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("score",),
    )

    generated_score = result.keyword_argument_dict["score"]

    assert generated_score < 85
    assert generated_score >= 50
    assert result.expected_result == "Orta"


def test_generate_creates_input_for_two_false_branches() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 85",
            "score >= 50",
            'return "Başarısız"',
            "END",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            "False",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("score",),
    )

    generated_score = result.keyword_argument_dict["score"]

    assert generated_score < 50
    assert result.expected_result == "Başarısız"


def test_generate_supports_equal_condition() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "status == 'active'",
            'return "Allowed"',
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

    result = generator.generate(
        path=path,
        parameter_names=("status",),
    )

    assert result.keyword_argument_dict == {
        "status": "active",
    }
    assert result.expected_result == "Allowed"


def test_generate_supports_false_equal_condition() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "status == 'active'",
            'return "Denied"',
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

    result = generator.generate(
        path=path,
        parameter_names=("status",),
    )

    assert result.keyword_argument_dict["status"] != "active"
    assert result.expected_result == "Denied"


def test_generate_supports_boolean_parameter() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "is_admin",
            'return "Admin"',
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

    result = generator.generate(
        path=path,
        parameter_names=("is_admin",),
    )

    assert result.keyword_argument_dict == {
        "is_admin": True,
    }


def test_generate_supports_negated_boolean_parameter() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "not is_active",
            'return "Inactive"',
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

    result = generator.generate(
        path=path,
        parameter_names=("is_active",),
    )

    assert result.keyword_argument_dict == {
        "is_active": False,
    }


def test_generate_uses_default_value_for_unconstrained_parameter() -> None:
    generator = PathInputGenerator()

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

    result = generator.generate(
        path=path,
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict == {
        "value": 0,
    }
    assert result.expected_result == 0


def test_generate_rejects_path_without_metadata() -> None:
    generator = PathInputGenerator()

    path = ExecutionPath(
        node_ids=[1, 2],
        edge_labels=[None],
    )

    with pytest.raises(
        ValueError,
        match="ExecutionPath düğüm metadata bilgilerini içermelidir",
    ):
        generator.generate(
            path=path,
            parameter_names=("score",),
        )


def test_generate_rejects_invalid_path_type() -> None:
    generator = PathInputGenerator()

    with pytest.raises(
        TypeError,
        match="path bir ExecutionPath örneği olmalıdır",
    ):
        generator.generate(  # type: ignore[arg-type]
            path="invalid",
            parameter_names=("score",),
        )


def test_generate_rejects_non_tuple_parameter_names() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "return 1",
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
        TypeError,
        match="parameter_names bir tuple olmalıdır",
    ):
        generator.generate(
            path=path,
            parameter_names=["score"],  # type: ignore[arg-type]
        )


def test_generate_rejects_duplicate_parameter_names() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "return 1",
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
        match="parameter_names tekrar eden değer içeremez",
    ):
        generator.generate(
            path=path,
            parameter_names=(
                "score",
                "score",
            ),
        )


def test_generate_rejects_unsupported_expression() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score + 1",
            "return 1",
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

    with pytest.raises(
        ValueError,
        match="Desteklenmeyen koşul ifadesi",
    ):
        generator.generate(
            path=path,
            parameter_names=("score",),
        )


def test_generate_rejects_dynamic_return_expression() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "return score * 2",
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
        match="Dinamik return ifadeleri henüz desteklenmiyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("score",),
        )