from __future__ import annotations

import pytest

from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import (
    PathInputGenerator,
    UnreachablePathError,
)


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


def test_generate_supports_dynamic_return_expression() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 5",
            "return score * 2",
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

    assert result.keyword_argument_dict["score"] == 5
    assert result.expected_result == 10


def test_generate_rejects_equal_value_below_minimum() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "attendance < 40",
            "attendance == 0",
            'return "Başvuruya katılmadı"',
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

    with pytest.raises(
        UnreachablePathError,
        match="Eşitlik değeri minimum kısıtını sağlamıyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("attendance",),
        )


def test_generate_rejects_equal_value_above_maximum() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score <= 50",
            "score == 75",
            'return "Ulaşılamaz"',
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
            "True",
            "True",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="Eşitlik değeri maksimum kısıtını sağlamıyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("score",),
        )


def test_generate_rejects_range_added_after_conflicting_equality() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "attendance == 0",
            "attendance < 40",
            'return "Ulaşılamaz"',
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
            "True",
            "False",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="Eşitlik değeri minimum kısıtını sağlamıyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("attendance",),
        )


def test_generate_rejects_forbidden_singleton_range() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "score <= 50",
            "score != 50",
            'return "Ulaşılamaz"',
            "END",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "True",
            "True",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="Tek mümkün değer eşitsizlik kısıtıyla yasaklandı",
    ):
        generator.generate(
            path=path,
            parameter_names=("score",),
        )

def test_generate_supports_true_and_condition() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50 and attendance >= 70",
            'return "Kabul"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "attendance"),
    )
    assert result.keyword_argument_dict == {
        "score": 50,
        "attendance": 70,
    }


def test_generate_supports_false_and_condition() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50 and attendance >= 70",
            'return "Red"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "False", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "attendance"),
    )
    values = result.keyword_argument_dict
    assert values["score"] < 50 or values["attendance"] < 70


def test_generate_supports_true_or_condition() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "score < 40 or project_score < 50",
            'return "Yetersiz"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "project_score"),
    )
    values = result.keyword_argument_dict
    assert values["score"] < 40 or values["project_score"] < 50


def test_generate_supports_false_or_condition() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "score < 40 or project_score < 50",
            'return "Yeterli"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "False", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "project_score"),
    )
    values = result.keyword_argument_dict
    assert values["score"] >= 40
    assert values["project_score"] >= 50


def test_generate_supports_nested_boolean_condition() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "(score >= 50 and attendance >= 70) or is_admin",
            'return "Allowed"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "attendance", "is_admin"),
    )
    values = result.keyword_argument_dict
    assert (
        values["score"] >= 50 and values["attendance"] >= 70
    ) or values["is_admin"] is True


def test_generate_supports_not_wrapped_boolean_operation() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "not (score < 50 or attendance < 70)",
            'return "Kabul"',
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "attendance"),
    )
    values = result.keyword_argument_dict
    assert values["score"] >= 50
    assert values["attendance"] >= 70


def test_generate_selects_non_conflicting_boolean_alternative() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "score < 40 or attendance < 70",
            'return "Alternatif"',
            "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "True", None],
    )
    result = generator.generate(
        path=path,
        parameter_names=("score", "attendance"),
    )
    values = result.keyword_argument_dict
    assert values["score"] >= 50
    assert values["attendance"] < 70


def test_generate_creates_zero_iteration_while_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "value > 0",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "while",
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
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict["value"] <= 0


def test_generate_creates_one_iteration_while_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "value > 0",
            "value -= 1",
            "value > 0",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Loop",
            "False",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict["value"] == 1


def test_generate_creates_two_iteration_while_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "value > 0",
            "value -= 1",
            "value > 0",
            "value -= 1",
            "value > 0",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict["value"] == 2


def test_generate_supports_incrementing_while_loop() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "value < 2",
            "value += 1",
            "value < 2",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Loop",
            "False",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict["value"] == 1


def test_generate_creates_empty_iterable_for_zero_iteration_for_loop() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "item in values",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Complete",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == []


def test_generate_creates_iterable_for_one_iteration_for_loop() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "item in values",
            "total += item",
            "item in values",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == [0]


def test_generate_creates_zero_division_exception_path_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = a / b",
            "except ZeroDivisionError",
            "return None",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("a", "b"),
    )

    assert result.keyword_argument_dict["b"] == 0
    assert result.expected_result is None
    assert result.expected_exception is None


def test_generate_creates_index_error_exception_path_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = values[0]",
            "except IndexError",
            "return None",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == []
    assert result.expected_result is None


def test_generate_creates_index_error_for_larger_index() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = values[2]",
            "except IndexError",
            "return None",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert len(result.keyword_argument_dict["values"]) == 2


def test_generate_creates_key_error_exception_path_input() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = data['name']",
            "except KeyError",
            "return None",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("data",),
    )

    assert result.keyword_argument_dict["data"] == {}
    assert result.expected_result is None


def test_generate_preserves_uncaught_raise_exception() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "raise ValueError('Geçersiz')",
            "END",
        ],
        node_types=[
            "start",
            "Raise",
            "end",
        ],
        edge_labels=[
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=(),
    )

    assert result.expected_exception == "ValueError"
    assert result.expected_result is None


def test_generate_supports_direct_parameter_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score == 75",
            "return score",
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

    assert result.expected_result == 75


def test_generate_supports_multiple_parameter_arithmetic_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score == 70",
            "bonus == 5",
            "return score + bonus",
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
            "True",
            "True",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("score", "bonus"),
    )

    assert result.expected_result == 75


def test_generate_supports_assignment_before_dynamic_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score == 10",
            "total = score + 5",
            "return total * 2",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "Assign",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("score",),
    )

    assert result.expected_result == 30


def test_generate_supports_augmented_assignment_before_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "total == 3",
            "total += 4",
            "return total",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "AugAssign",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("total",),
    )

    assert result.expected_result == 7


def test_generate_supports_subscript_dynamic_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "values = [10, 20]",
            "return values[0]",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=(),
    )

    assert result.expected_result == 10


def test_generate_supports_conditional_dynamic_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score == 80",
            "return score if score >= 50 else 0",
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

    assert result.expected_result == 80


def test_generate_rejects_function_call_in_dynamic_return() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "return len(values)",
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
        match="Dinamik return ifadesi güvenli biçimde hesaplanamadı",
    ):
        generator.generate(
            path=path,
            parameter_names=("values",),
        )


def test_generate_constant_return_skips_unresolved_loop_assignment() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "item in values",
            "total += item",
            "item in values",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == [0]
    assert result.expected_result == 0


def test_generate_constant_handler_return_skips_raising_assignment() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = a / b",
            "except ZeroDivisionError",
            "return None",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("a", "b"),
    )

    assert result.keyword_argument_dict["b"] == 0
    assert result.expected_result is None
    assert result.expected_exception is None


def test_generate_supports_constant_local_while_variable() -> None:
    generator = PathInputGenerator()

    path = ExecutionPath(
        node_ids=[
            1,
            3,
            4,
            5,
            4,
            5,
            4,
            6,
            2,
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
            None,
        ],
        node_labels=[
            "START",
            "remaining_checks = 2",
            "remaining_checks > 0",
            "remaining_checks -= 1",
            "remaining_checks > 0",
            "remaining_checks -= 1",
            "remaining_checks > 0",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        line_numbers=[
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            5,
            6,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("score",),
    )

    assert result.keyword_argument_dict["score"] == 0
    assert result.expected_result == "Tamamlandı"


def test_generate_supports_zero_iteration_local_while() -> None:
    generator = PathInputGenerator()

    path = ExecutionPath(
        node_ids=[1, 3, 4, 6, 2],
        edge_labels=[
            None,
            None,
            "False",
            None,
        ],
        node_labels=[
            "START",
            "remaining_checks = 0",
            "remaining_checks > 0",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "return",
            "end",
        ],
        line_numbers=[1, 2, 3, 5, 6],
    )

    result = generator.generate(
        path=path,
        parameter_names=(),
    )

    assert result.keyword_arguments == ()
    assert result.expected_result == "Tamamlandı"


def test_generate_rejects_infeasible_local_while_path() -> None:
    generator = PathInputGenerator()

    path = ExecutionPath(
        node_ids=[
            1,
            3,
            4,
            5,
            4,
            6,
            2,
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "False",
            None,
        ],
        node_labels=[
            "START",
            "remaining_checks = 2",
            "remaining_checks > 0",
            "remaining_checks -= 1",
            "remaining_checks > 0",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        line_numbers=[1, 2, 3, 4, 3, 5, 6],
    )

    with pytest.raises(
        UnreachablePathError,
        match="iterasyon sayısıyla uyuşmuyor",
    ):
        generator.generate(
            path=path,
            parameter_names=(),
        )


def test_generate_counts_for_and_while_iterations_separately() -> None:
    generator = PathInputGenerator()

    path = ExecutionPath(
        node_ids=[
            1,
            3,
            4,
            5,
            4,
            6,
            7,
            8,
            7,
            8,
            7,
            9,
            2,
        ],
        edge_labels=[
            None,
            None,
            "Iterate",
            "Next",
            "Complete",
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
            None,
        ],
        node_labels=[
            "START",
            "total = 0",
            "item in values",
            "total += item",
            "item in values",
            "remaining_checks = 2",
            "remaining_checks > 0",
            "remaining_checks -= 1",
            "remaining_checks > 0",
            "remaining_checks -= 1",
            "remaining_checks > 0",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "for",
            "AugAssign",
            "for",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        line_numbers=[
            1,
            2,
            3,
            4,
            3,
            5,
            6,
            7,
            6,
            7,
            6,
            8,
            9,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == [0]
    assert result.expected_result == 0


def test_count_loop_iterations_supports_distinct_visit_node_ids() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "value > 0",
            "value -= 1",
            "value > 0",
            "return 0",
            "END",
        ],
        node_types=[
            "start",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Loop",
            "False",
            None,
        ],
    )

    loop_step = path.loop_steps[0]

    assert (
        PathInputGenerator._count_loop_iterations(
            path=path,
            loop_step=loop_step,
        )
        == 1
    )


def test_generate_supports_true_in_condition_with_strings() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "coupon in ('SAVE10', 'SAVE5')",
            "return 'Geçerli'",
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
        parameter_names=("coupon",),
    )

    assert result.keyword_argument_dict["coupon"] in {
        "SAVE10",
        "SAVE5",
    }
    assert result.expected_result == "Geçerli"


def test_generate_supports_false_in_condition_with_strings() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "coupon in ('SAVE10', 'SAVE5')",
            "return 'Geçersiz'",
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
        parameter_names=("coupon",),
    )

    assert result.keyword_argument_dict["coupon"] not in {
        "SAVE10",
        "SAVE5",
    }
    assert result.expected_result == "Geçersiz"


def test_generate_supports_true_not_in_condition() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "coupon not in ('NONE', '')",
            "return 'Geçersiz kupon'",
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
        parameter_names=("coupon",),
    )

    assert result.keyword_argument_dict["coupon"] not in {
        "NONE",
        "",
    }


def test_generate_supports_false_not_in_condition() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "coupon not in ('NONE', '')",
            "return 'Kupon yok'",
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
        parameter_names=("coupon",),
    )

    assert result.keyword_argument_dict["coupon"] in {
        "NONE",
        "",
    }


def test_generate_combines_string_equality_and_membership() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "coupon != 'SAVE10'",
            "coupon in ('SAVE10', 'SAVE5')",
            "return 'SAVE5'",
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
            "True",
            "True",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("coupon",),
    )

    assert result.keyword_argument_dict["coupon"] == "SAVE5"


def test_generate_skips_zero_division_path_with_local_denominator() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "valid_count = 1",
            "try",
            "average = total / valid_count",
            "except ZeroDivisionError",
            "return 'Hata'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="yerel bir böleni sıfır yapmayı gerektiriyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("total",),
        )


def test_generate_still_supports_zero_division_parameter() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = total / divisor",
            "except ZeroDivisionError",
            "return 'Hata'",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("total", "divisor"),
    )

    assert result.keyword_argument_dict["divisor"] == 0
    assert result.expected_result == "Hata"


def test_generate_rejects_empty_for_input_when_path_requires_truthy_list(
) -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "item in items",
            "return 'Geçersiz kupon'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            "Complete",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="Boolean yol kısıtıyla çelişiyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("items",),
        )


def test_generate_accepts_non_empty_for_input_when_path_requires_truthy_list(
) -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "item in items",
            "total += item",
            "item in items",
            "return 'Tamamlandı'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "False",
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("items",),
    )

    assert result.keyword_argument_dict["items"] == [0]
    assert result.expected_result == "Tamamlandı"


def test_generate_rejects_non_empty_for_input_when_path_requires_falsy_list(
) -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "item in items",
            "total += item",
            "item in items",
            "return 'Ulaşılamaz'",
            "END",
        ],
        node_types=[
            "start",
            "if",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
    )

    with pytest.raises(
        UnreachablePathError,
        match="Boolean yol kısıtıyla çelişiyor",
    ):
        generator.generate(
            path=path,
            parameter_names=("items",),
        )


# ============================================================
# PathInputGenerator candidate_values integration tests
# ============================================================


def test_generate_uses_candidate_value_for_parameter() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "return score",
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
        candidate_values={
            "score": 75,
        },
    )

    assert result.keyword_argument_dict["score"] == 75
    assert result.expected_result == 75


def test_generate_filters_candidate_for_local_variable() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "return score",
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
        candidate_values={
            "score": 80,
            "local_value": 999,
        },
    )

    assert result.keyword_argument_dict == {
        "score": 80,
    }


def test_generate_coerces_candidate_collection_to_parameter_type() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "return values",
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
        parameter_names=("values",),
        parameter_types={
            "values": "tuple[int, ...]",
        },
        candidate_values={
            "values": [1, 2],
        },
    )

    assert result.keyword_argument_dict["values"] == (
        1,
        2,
    )
    assert isinstance(
        result.keyword_argument_dict["values"],
        tuple,
    )


def test_generate_loop_input_overrides_candidate_when_required() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "value > 0",
            "value -= 1",
            "value > 0",
            "return value",
            "END",
        ],
        node_types=[
            "start",
            "while",
            "AugAssign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "True",
            "Loop",
            "False",
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=("value",),
        candidate_values={
            "value": 100,
        },
    )

    # Bu path tam bir while iterasyonu gerektirir.
    assert result.keyword_argument_dict["value"] == 1


def test_generate_exception_input_overrides_candidate_when_required() -> None:
    generator = PathInputGenerator()

    path = create_execution_path(
        node_labels=[
            "START",
            "try",
            "result = total / divisor",
            "except ZeroDivisionError",
            "return 'Hata'",
            "END",
        ],
        node_types=[
            "start",
            "try",
            "Assign",
            "except",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            "Success",
            "Exception",
            None,
            None,
        ],
    )

    result = generator.generate(
        path=path,
        parameter_names=(
            "total",
            "divisor",
        ),
        candidate_values={
            "total": 10,
            "divisor": 5,
        },
    )

    assert result.keyword_argument_dict["total"] == 10
    assert result.keyword_argument_dict["divisor"] == 0
    assert result.expected_result == "Hata"


def test_generate_rejects_invalid_candidate_values_type() -> None:
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

    with pytest.raises(
        TypeError,
        match="candidate_values bir dict veya None olmalıdır.",
    ):
        generator.generate(
            path=path,
            parameter_names=("value",),
            candidate_values=[  # type: ignore[arg-type]
                ("value", 1),
            ],
        )


def test_generate_rejects_invalid_candidate_variable_name() -> None:
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

    with pytest.raises(
        ValueError,
        match="candidate_values anahtarları boş olmayan string",
    ):
        generator.generate(
            path=path,
            parameter_names=("value",),
            candidate_values={
                "": 1,
            },
        )


def test_generate_accepts_candidate_for_tighter_upper_bound_path() -> None:
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
        candidate_values={
            "score": 49,
        },
    )

    assert result.keyword_argument_dict["score"] == 49
    assert result.keyword_argument_dict["score"] < 85
    assert result.keyword_argument_dict["score"] < 50
    assert result.expected_result == "Başarısız"


# ============================================================
# Derived-value input synthesis tests
# ============================================================


@pytest.mark.parametrize(
    ("assignments", "condition", "parameters", "assertion"),
    (
        (("derived = amount",), "derived >= 100", ("amount",),
         lambda values: values["amount"] >= 100),
        (("derived = amount + fee",), "derived >= 100", ("amount", "fee"),
         lambda values: values["amount"] + values["fee"] >= 100),
        (("remaining = stock - required",), "remaining >= 10", ("stock", "required"),
         lambda values: values["stock"] - values["required"] >= 10),
        (("derived = amount * 2",), "derived >= 100", ("amount",),
         lambda values: values["amount"] * 2 >= 100),
        (("derived = amount / 2",), "derived >= 100", ("amount",),
         lambda values: values["amount"] / 2 >= 100),
        (("first = amount + 10", "second = first * 2"), "second >= 100", ("amount",),
         lambda values: (values["amount"] + 10) * 2 >= 100),
    ),
)
def test_generate_propagates_affine_local_condition_to_parameters(
    assignments: tuple[str, ...],
    condition: str,
    parameters: tuple[str, ...],
    assertion: object,
) -> None:
    path = create_execution_path(
        node_labels=["START", *assignments, condition, "return 1", "END"],
        node_types=["start", *("Assign" for _ in assignments), "if", "return", "end"],
        edge_labels=[None, *(None for _ in assignments), "True", None],
    )

    generated = PathInputGenerator().generate(path, parameters)
    values = generated.keyword_argument_dict

    assert assertion(values)  # type: ignore[operator]
    assert not set(values) - set(parameters)


def create_average_path(
    *,
    condition: str = "average >= 50",
) -> ExecutionPath:
    return ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 7, 8, 9, 10],
        edge_labels=[None, None, None, "Iterate", None, "Next", "Iterate",
                     None, "Next", "Complete", None, "True", None],
        node_labels=[
            "START", "total = 0", "count = 0", "item in items",
            "total += item", "count += 1", "item in items", "total += item",
            "count += 1", "item in items", "average = total / count", condition,
            "return 1", "END",
        ],
        node_types=[
            "start", "Assign", "Assign", "for", "AugAssign", "AugAssign",
            "for", "AugAssign", "AugAssign", "for", "Assign", "if", "return", "end",
        ],
        line_numbers=list(range(1, 15)),
    )


@pytest.mark.parametrize(
    ("parameter_type", "expected_type"),
    (("list[int]", list), ("tuple[int, ...]", tuple)),
)
def test_generate_propagates_average_constraint_to_typed_collection(
    parameter_type: str,
    expected_type: type,
) -> None:
    generated = PathInputGenerator().generate(
        create_average_path(),
        ("items",),
        {"items": parameter_type},
    )
    items = generated.keyword_argument_dict["items"]

    assert isinstance(items, expected_type)
    assert len(items) == 2
    assert sum(items) / len(items) >= 50
    assert "average" not in generated.keyword_argument_dict
    assert "total" not in generated.keyword_argument_dict
    assert "count" not in generated.keyword_argument_dict


def test_generate_excludes_continued_item_from_average() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 4, 5, 7, 8, 4, 9, 10, 11, 12],
        edge_labels=[None, None, None, "Iterate", "True", "Continue",
                     "Iterate", "False", None, "Next", "Complete", None,
                     "True", None],
        node_labels=[
            "START", "total = 0", "count = 0", "item in items", "item < 0",
            "continue", "item in items", "item < 0", "total += item",
            "count += 1", "item in items", "average = total / count",
            "average >= 50", "return 1", "END",
        ],
        node_types=[
            "start", "Assign", "Assign", "for", "if", "continue", "for",
            "if", "AugAssign", "AugAssign", "for", "Assign", "if", "return", "end",
        ],
        line_numbers=list(range(1, 16)),
    )

    generated = PathInputGenerator().generate(
        path, ("items",), {"items": "list[int]"}
    )
    items = generated.keyword_argument_dict["items"]

    assert items[0] < 0
    assert items[1] >= 50


def test_generate_preserves_subscript_alias_while_satisfying_average() -> None:
    path = create_average_path()
    path.node_ids.insert(1, 20)
    path.node_labels.insert(1, "first = items[0]")
    path.node_types.insert(1, "Assign")
    path.line_numbers.insert(1, 1)
    path.edge_labels.insert(1, None)
    path.node_ids.insert(2, 21)
    path.node_labels.insert(2, "first < 0")
    path.node_types.insert(2, "if")
    path.line_numbers.insert(2, 2)
    path.edge_labels.insert(2, "False")

    generated = PathInputGenerator().generate(
        path, ("items",), {"items": "list[int]"}
    )
    items = generated.keyword_argument_dict["items"]

    assert items[0] >= 0
    assert sum(items) / len(items) >= 50


@pytest.mark.parametrize(
    ("assignment", "parameters"),
    (
        ("derived = left * right", ("left", "right")),
        ("derived = amount / divisor", ("amount", "divisor")),
        ("derived = items[index]", ("items", "index")),
    ),
)
def test_generate_rejects_unsafe_derived_provenance(
    assignment: str,
    parameters: tuple[str, ...],
) -> None:
    path = create_execution_path(
        node_labels=["START", assignment, "derived >= 10", "return 1", "END"],
        node_types=["start", "Assign", "if", "return", "end"],
        edge_labels=[None, None, "True", None],
    )

    with pytest.raises(
        UnreachablePathError,
        match="desteklenmeyen provenance",
    ):
        PathInputGenerator().generate(path, parameters)

def test_generate_defers_name_to_name_comparison() -> None:
    """
    Değişkenler arası ilişki literal olarak yorumlanmamalıdır.

    Somut değerler upstream relational witness katmanından geldiğinde
    PathInputGenerator bu değerleri bozmadan kullanmalıdır.
    """
    path = create_execution_path(
        node_labels=[
            "START",
            "lower < upper",
            "return 'ordered'",
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
        parameter_names=(
            "lower",
            "upper",
        ),
        candidate_values={
            "lower": 1,
            "upper": 2,
        },
    )

    assert result.keyword_argument_dict == {
        "lower": 1,
        "upper": 2,
    }
    assert result.expected_result == "ordered"


def test_generate_replays_for_target_before_dynamic_return() -> None:
    """For hedefi, döngü gövdesindeki atamadan önce bağlanır."""
    path = create_execution_path(
        node_labels=[
            "START",
            "total = 0",
            "value in values",
            "total += value",
            "value in values",
            "return total",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={
            "values": "list[int]",
        },
    )

    values = result.keyword_argument_dict["values"]

    assert values == [0]
    assert result.expected_result == sum(values)


def test_extract_expected_result_replays_successive_for_values() -> None:
    """Aynı for düğümünün her ziyareti iterator'ı ilerletir."""
    path = ExecutionPath(
        node_ids=[
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            5,
            6,
        ],
        edge_labels=[
            None,
            None,
            "Iterate",
            "Next",
            "Iterate",
            "Next",
            "Complete",
            None,
        ],
        node_labels=[
            "START",
            "total = 0",
            "value in values",
            "total += value",
            "value in values",
            "total += value",
            "value in values",
            "return total",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "for",
            "AugAssign",
            "for",
            "AugAssign",
            "for",
            "return",
            "end",
        ],
        line_numbers=[
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            5,
            6,
        ],
    )

    result = PathInputGenerator._extract_expected_result(
        path=path,
        keyword_arguments=(("values", [2, 3]),),
    )

    assert result == 5


def test_extract_expected_result_supports_for_target_unpacking() -> None:
    """For hedefindeki tuple unpacking yerel ortama aktarılır."""
    path = create_execution_path(
        node_labels=[
            "START",
            "total = 0",
            "(left, right) in pairs",
            "total += left",
            "total += right",
            "return total",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "for",
            "AugAssign",
            "AugAssign",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "Iterate",
            None,
            None,
            None,
        ],
    )

    result = PathInputGenerator._extract_expected_result(
        path=path,
        keyword_arguments=(
            (
                "pairs",
                [(2, 3)],
            ),
        ),
    )

    assert result == 5


def test_generate_supports_formatted_string_return() -> None:
    """Dinamik f-string return, Python format spec kurallarını korur."""
    path = create_execution_path(
        node_labels=[
            "START",
            "score == 12.5",
            "return f'Score: {score:.2f}'",
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
        parameter_names=("score",),
        parameter_types={
            "score": "float",
        },
    )

    assert result.keyword_argument_dict["score"] == 12.5
    assert result.expected_result == "Score: 12.50"


def test_extract_expected_result_formats_replayed_loop_total() -> None:
    """For replay sonucu f-string içinde formatlanabilir."""
    path = create_execution_path(
        node_labels=[
            "START",
            "total = 0",
            "value in values",
            "total += value",
            "return f'Total: {total:.2f}'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "for",
            "AugAssign",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "Iterate",
            None,
            None,
        ],
    )

    result = PathInputGenerator._extract_expected_result(
        path=path,
        keyword_arguments=(("values", [7]),),
    )

    assert result == "Total: 7.00"
