from __future__ import annotations

import pytest

from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import (
    PathInputGenerator,
    UnreachablePathError,
    UnsupportedInputSynthesisError,
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


def create_caught_exception_path(
    *,
    source_label: str,
    handler_label: str,
    return_label: str = "return None",
) -> ExecutionPath:
    """Exception-handler input synthesis testleri için metadata yolu oluşturur."""
    return create_execution_path(
        node_labels=[
            "START",
            "try",
            source_label,
            handler_label,
            return_label,
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


def test_generate_isolates_unsupported_attribute_truthiness() -> None:
    generator = PathInputGenerator()
    path = create_execution_path(
        node_labels=[
            "START",
            "not custom_object.attribute",
            "return False",
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(
        UnsupportedInputSynthesisError,
        match="Desteklenmeyen koşul ifadesi",
    ):
        generator.generate(
            path=path,
            parameter_names=("custom_object",),
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


def test_generate_uses_first_synthesizable_tuple_exception_in_source_order() -> None:
    result = PathInputGenerator().generate(
        path=create_caught_exception_path(
            source_label="result = values[0]",
            handler_label="except (KeyError, IndexError)",
        ),
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == {}


def test_generate_uses_supported_member_after_unsupported_tuple_member() -> None:
    result = PathInputGenerator().generate(
        path=create_caught_exception_path(
            source_label="result = values[0]",
            handler_label="except (ValueError, IndexError)",
        ),
        parameter_names=("values",),
    )

    assert result.keyword_argument_dict["values"] == []


def test_generate_supports_safe_attribute_tuple_exception_member() -> None:
    result = PathInputGenerator().generate(
        path=create_caught_exception_path(
            source_label="result = data['name']",
            handler_label="except (errors.KeyError, errors.IndexError)",
        ),
        parameter_names=("data",),
    )

    assert result.keyword_argument_dict["data"] == {}


def test_generate_rejects_tuple_without_synthesizable_exception_member() -> None:
    with pytest.raises(
        UnsupportedInputSynthesisError,
        match="ValueError.*TypeError|TypeError.*ValueError",
    ):
        PathInputGenerator().generate(
            path=create_caught_exception_path(
                source_label="result = transform(value)",
                handler_label="except (ValueError, TypeError)",
            ),
            parameter_names=("value",),
        )


@pytest.mark.parametrize(
    "handler_label",
    (
        "except factory()",
        "except errors[0]",
        "except (*errors,)",
        "except factory().IndexError",
    ),
)
def test_generate_rejects_unsafe_dynamic_exception_handler_expression(
    handler_label: str,
) -> None:
    with pytest.raises(UnsupportedInputSynthesisError):
        PathInputGenerator().generate(
            path=create_caught_exception_path(
                source_label="result = values[0]",
                handler_label=handler_label,
            ),
            parameter_names=("values",),
        )


@pytest.mark.parametrize(
    "handler_label",
    ("except ValueError", "except module.CustomError"),
)
def test_generate_preserves_safe_single_exception_handler_extraction(
    handler_label: str,
) -> None:
    result = PathInputGenerator().generate(
        path=create_caught_exception_path(
            source_label="result = transform(value)",
            handler_label=handler_label,
        ),
        parameter_names=("value",),
    )

    assert result.keyword_argument_dict["value"] == 0


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


@pytest.mark.parametrize(
    (
        "parameter_type",
        "expected_type",
        "member",
        "parameter_name",
    ),
    (
        ("dict[str, bool]", dict, "flag", "settings"),
        ("list[int]", list, 7, "codes"),
        ("tuple[str, ...]", tuple, "token", "labels"),
        ("set[float]", set, 2.5, "measurements"),
    ),
)
def test_generate_supports_literal_membership_in_typed_parameter_collection(
    parameter_type: str,
    expected_type: type[object],
    member: object,
    parameter_name: str,
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            f"{member!r} in {parameter_name}",
            "return True",
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=(parameter_name,),
        parameter_types={parameter_name: parameter_type},
    )

    collection = result.keyword_argument_dict[parameter_name]
    assert type(collection) is expected_type
    assert member in collection


def test_generate_supports_false_literal_membership_in_dict_parameter() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "'feature' in options",
            "return False",
            "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "False", None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("options",),
        parameter_types={"options": "dict[str, bool]"},
    )

    assert "feature" not in result.keyword_argument_dict["options"]


def test_generate_rejects_conflicting_literal_collection_membership() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "'feature' in options",
            "'feature' not in options",
            "return True",
            "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "True", None],
    )

    with pytest.raises(UnreachablePathError, match="koleksiyon üyelik"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("options",),
            parameter_types={"options": "dict[str, bool]"},
        )


def test_generate_materializes_membership_without_mutating_candidate_seed() -> None:
    candidate_seed: dict[str, bool] = {}
    path = create_execution_path(
        node_labels=["START", "'enabled' in flags", "return True", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("flags",),
        parameter_types={"flags": "dict[str, bool]"},
        candidate_values={"flags": candidate_seed},
    )

    assert result.keyword_argument_dict["flags"] == {"enabled": False}
    assert candidate_seed == {}


def test_generate_combines_collection_truthiness_and_membership() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "labels",
            "'ready' in labels",
            "return True",
            "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "True", None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("labels",),
        parameter_types={"labels": "list[str]"},
    )

    assert result.keyword_argument_dict["labels"] == ["ready"]


def test_generate_rejects_membership_literal_incompatible_with_element_type() -> None:
    path = create_execution_path(
        node_labels=["START", "42 in names", "return True", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="eleman türü"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("names",),
            parameter_types={"names": "list[str]"},
        )


def test_generate_rejects_unhashable_dict_membership_literal() -> None:
    path = create_execution_path(
        node_labels=["START", "[1] not in mapping", "return True", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="hashable"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("mapping",),
            parameter_types={"mapping": "dict[int, str]"},
        )


def test_generate_rejects_membership_for_fixed_heterogeneous_tuple() -> None:
    path = create_execution_path(
        node_labels=["START", "1 in pair", "return True", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="tuple"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("pair",),
            parameter_types={"pair": "tuple[int, str]"},
        )


def _create_typed_collection_constraint_path(
    *expressions: str,
) -> ExecutionPath:
    return create_execution_path(
        node_labels=["START", *expressions, "return True", "END"],
        node_types=["start", *("if" for _ in expressions), "return", "end"],
        edge_labels=[None, *("True" for _ in expressions), None],
    )


@pytest.mark.parametrize(
    ("parameter_type", "seed", "condition"),
    (
        ("list[str]", [1], "'ok' in values"),
        ("list[int]", [True], "2 in values"),
        ("tuple[str, ...]", ("ok", 1), "'ok' in values"),
        ("set[int]", {True}, "2 in values"),
        ("frozenset[int]", frozenset({True}), "2 in values"),
        ("dict[str, int]", {1: 2}, "'required' in values"),
        ("dict[str, int]", {"required": "invalid"}, "'required' in values"),
    ),
)
def test_generate_rejects_typed_collection_seed_with_incompatible_contents(
    parameter_type: str,
    seed: object,
    condition: str,
) -> None:
    path = _create_typed_collection_constraint_path(condition)
    original_seed = (
        seed.copy()
        if isinstance(seed, (dict, list, set))
        else seed
    )

    with pytest.raises(
        UnsupportedInputSynthesisError,
        match="seed|Seed|typed|schema|şema|tür",
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("values",),
            parameter_types={"values": parameter_type},
            candidate_values={"values": seed},
        )

    assert seed == original_seed


def test_generate_preserves_valid_typed_collection_seed_by_copy() -> None:
    seed = ["present"]
    path = _create_typed_collection_constraint_path("'present' in values")

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
        candidate_values={"values": seed},
    ).keyword_argument_dict["values"]

    assert result == seed
    assert result is not seed


def test_generate_derives_string_witness_after_legacy_sentinels_are_forbidden() -> None:
    forbidden = ("__generated_member__", "__generated_member_2__")
    path = _create_typed_collection_constraint_path(
        "values",
        *(f"{member!r} not in values" for member in forbidden),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
    ).keyword_argument_dict["values"]

    assert result
    assert all(member not in result for member in forbidden)


def test_generate_derives_integer_witness_after_initial_candidates_are_forbidden() -> None:
    forbidden = (0, 1, -1, 2, -2)
    path = _create_typed_collection_constraint_path(
        "values",
        *(f"{member!r} not in values" for member in forbidden),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[int]"},
    ).keyword_argument_dict["values"]

    assert result
    assert all(member not in result for member in forbidden)


@pytest.mark.parametrize("parameter_type", ("list[bool]", "set[bool]"))
def test_generate_proves_exhaustion_only_for_closed_bool_domain(
    parameter_type: str,
) -> None:
    path = _create_typed_collection_constraint_path(
        "values",
        "False not in values",
        "True not in values",
    )

    with pytest.raises(UnreachablePathError, match="sentinel|domain|üye"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("values",),
            parameter_types={"values": parameter_type},
        )


def test_generate_classifies_unknown_membership_domain_as_unsupported() -> None:
    path = _create_typed_collection_constraint_path(
        "values",
        "b'forbidden' not in values",
    )

    with pytest.raises(UnsupportedInputSynthesisError):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("values",),
            parameter_types={"values": "list[bytes]"},
        )


def test_generate_membership_witness_is_deterministic_with_bounded_search() -> None:
    forbidden = tuple(f"blocked-{index}" for index in range(32))
    path = _create_typed_collection_constraint_path(
        "values",
        *(f"{member!r} not in values" for member in forbidden),
    )
    arguments = {
        "path": path,
        "parameter_names": ("values",),
        "parameter_types": {"values": "list[str]"},
    }

    first = PathInputGenerator().generate(**arguments).keyword_argument_dict["values"]
    second = PathInputGenerator().generate(**arguments).keyword_argument_dict["values"]

    assert first == second
    assert first
    assert all(member not in first for member in forbidden)


@pytest.mark.parametrize(
    "expression",
    (
        "'x' in options.keys()",
        "'x' not in options.keys()",
        "'x' in options.view",
        "'x' in options[0]",
        "'x' in make_options()",
        "'x' in (options + [])",
        "'x' in [item for item in options]",
        "member in options.keys()",
    ),
)
def test_generate_normalizes_unsupported_membership_rhs(
    expression: str,
) -> None:
    path = _create_typed_collection_constraint_path(expression)

    with pytest.raises(UnsupportedInputSynthesisError) as error_info:
        PathInputGenerator().generate(
            path=path,
            parameter_names=("options",),
            parameter_types={"options": "list[str]"},
        )

    assert expression not in str(error_info.value)


def _create_single_iteration_membership_path(expression: str) -> ExecutionPath:
    return ExecutionPath(
        node_ids=[1, 2, 3, 2, 4, 5],
        edge_labels=[None, "Iterate", "True", "Complete", None],
        node_labels=[
            "START",
            "entry in values",
            expression,
            "entry in values",
            "return True",
            "END",
        ],
        node_types=["start", "for", "if", "for", "return", "end"],
        line_numbers=[1, 2, 3, 2, 4, 5],
    )


@pytest.mark.parametrize(
    ("parameter_type", "seed"),
    (
        ("list[str]", ["present"]),
        ("tuple[str, ...]", ("present",)),
        ("set[str]", {"present"}),
        ("frozenset[str]", frozenset({"present"})),
        ("dict[str, int]", {"present": 1}),
    ),
)
def test_generate_preserves_single_iteration_seed_that_already_has_member(
    parameter_type: str,
    seed: object,
) -> None:
    path = _create_single_iteration_membership_path("'present' in values")

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": parameter_type},
        candidate_values={"values": seed},
    ).keyword_argument_dict["values"]

    assert result == seed
    assert len(result) == 1
    if isinstance(seed, (dict, list, set)):
        assert result is not seed


def test_generate_preserves_single_iteration_seed_that_already_excludes_member() -> None:
    seed = ["other"]
    path = _create_single_iteration_membership_path("'absent' not in values")

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
        candidate_values={"values": seed},
    ).keyword_argument_dict["values"]

    assert result == seed
    assert len(result) == 1
    assert result is not seed


def test_generate_removes_forbidden_member_copy_on_write_when_cardinality_is_free() -> None:
    seed = ["forbidden", "retained"]
    path = _create_typed_collection_constraint_path("'forbidden' not in values")

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
        candidate_values={"values": seed},
    ).keyword_argument_dict["values"]

    assert result == ["retained"]
    assert seed == ["forbidden", "retained"]


def test_generate_rejects_loop_membership_mutation_that_changes_cardinality() -> None:
    seed = ["other"]
    path = _create_single_iteration_membership_path("'required' in values")

    with pytest.raises(UnsupportedInputSynthesisError, match="loop|döngü"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("values",),
            parameter_types={"values": "list[str]"},
            candidate_values={"values": seed},
        )

    assert seed == ["other"]


def test_generate_allows_loop_membership_replacement_at_same_cardinality() -> None:
    seed = ["forbidden"]
    path = _create_single_iteration_membership_path(
        "'forbidden' not in values and 'required' in values"
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
        candidate_values={"values": seed},
    ).keyword_argument_dict["values"]

    assert result == ["required"]
    assert seed == ["forbidden"]


def _create_zero_iteration_membership_path(expression: str) -> ExecutionPath:
    return ExecutionPath(
        node_ids=[1, 2, 3, 4, 5],
        edge_labels=[None, "Complete", "True", None],
        node_labels=[
            "START",
            "entry in values",
            expression,
            "return True",
            "END",
        ],
        node_types=["start", "for", "if", "return", "end"],
        line_numbers=[1, 2, 3, 4, 5],
    )


def test_generate_rejects_membership_that_would_change_zero_iteration_loop() -> None:
    path = _create_zero_iteration_membership_path("'required' in values")

    with pytest.raises(UnsupportedInputSynthesisError, match="loop|döngü"):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("values",),
            parameter_types={"values": "list[str]"},
        )


def test_generate_preserves_zero_iteration_for_satisfied_not_in_constraint() -> None:
    path = _create_zero_iteration_membership_path("'absent' not in values")

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[str]"},
    ).keyword_argument_dict["values"]

    assert result == []


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


def create_derived_local_while_path(
    *,
    assignment: str,
    condition: str,
    updates: tuple[str, ...] = (),
    break_after_updates: bool = False,
) -> ExecutionPath:
    """Gerçek CFG ziyaret kimliği semantiğiyle local while yolu kurar."""
    node_ids = [1, 2, 3]
    edge_labels: list[str | None] = [None, None]
    node_labels = ["START", assignment, condition]
    node_types = ["start", "Assign", "while"]

    for update_index, update in enumerate(updates, start=1):
        node_ids.extend((3 + update_index, 3))
        edge_labels.extend(("True", "Loop"))
        node_labels.extend((update, condition))
        node_types.extend(("AugAssign", "while"))

    if break_after_updates:
        node_ids.append(20)
        edge_labels.extend(("True", "Break"))
        node_labels.append("break")
        node_types.append("break")
    else:
        edge_labels.append("False")

    node_ids.extend((21, 22))
    edge_labels.append(None)
    node_labels.extend(("return source", "END"))
    node_types.extend(("return", "end"))

    return ExecutionPath(
        node_ids=node_ids,
        edge_labels=edge_labels,
        node_labels=node_labels,
        node_types=node_types,
        line_numbers=list(range(1, len(node_ids) + 1)),
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


@pytest.mark.parametrize(
    ("return_expression", "parameter_names", "candidate_values", "expected"),
    [
        ("round(value)", ("value",), {"value": 12.5}, 12),
        ("round(value, 2)", ("value",), {"value": 12.345}, 12.35),
        ("round(value, 2)", ("value",), {"value": -12.345}, -12.35),
        ("round(value, -1)", ("value",), {"value": 125}, 120),
        ("round(value, None)", ("value",), {"value": 12.5}, 12),
        (
            "round(current / limit * 100, 2)",
            ("current", "limit"),
            {"current": 1, "limit": 3},
            33.33,
        ),
        (
            "round((value + 1) * (value - 1) / 3, 2)",
            ("value",),
            {"value": 4},
            5.0,
        ),
    ],
)
def test_generate_replays_allowlisted_round_calls(
    return_expression: str,
    parameter_names: tuple[str, ...],
    candidate_values: dict[str, object],
    expected: object,
) -> None:
    path = create_execution_path(
        node_labels=["START", f"return {return_expression}", "END"],
        node_types=["start", "return", "end"],
        edge_labels=[None, None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=parameter_names,
        candidate_values=candidate_values,
    )

    assert result.expected_result == expected


def test_generate_replays_round_from_local_arithmetic_assignment() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "percentage = current / limit * 100",
            "return round(percentage, 2)",
            "END",
        ],
        node_types=["start", "Assign", "return", "end"],
        edge_labels=[None, None, None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("current", "limit"),
        candidate_values={"current": 2, "limit": 3},
    )

    assert result.expected_result == 66.67


@pytest.mark.parametrize(
    "return_expression",
    [
        "round()",
        "round(value, 2, 3)",
        "unknown(value)",
        "value.round()",
        "round(value, ndigits=2)",
        "round(*values)",
        "round(unknown(value), 2)",
        "round(True)",
        "round(value, True)",
    ],
)
def test_generate_rejects_non_allowlisted_call_shapes(
    return_expression: str,
) -> None:
    path = create_execution_path(
        node_labels=["START", f"return {return_expression}", "END"],
        node_types=["start", "return", "end"],
        edge_labels=[None, None],
    )

    with pytest.raises(
        ValueError,
        match="Dinamik return ifadesi güvenli biçimde hesaplanamadı",
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("value", "values"),
            candidate_values={"value": 12.345, "values": [12.345]},
        )


def test_unsupported_expected_result_uses_specific_domain_error() -> None:
    path = create_execution_path(
        node_labels=["START", "return unknown_transform(value)", "END"],
        node_types=["start", "return", "end"],
        edge_labels=[None, None],
    )

    with pytest.raises(ValueError) as captured:
        PathInputGenerator().generate(
            path=path,
            parameter_names=("value",),
            candidate_values={"value": 3},
        )

    assert type(captured.value).__name__ == "UnsupportedExpectedResultError"
    assert "return unknown_transform(value)" in str(captured.value)


def test_generate_rejects_round_shadowed_by_parameter() -> None:
    path = create_execution_path(
        node_labels=["START", "return round(value)", "END"],
        node_types=["start", "return", "end"],
        edge_labels=[None, None],
    )

    with pytest.raises(
        ValueError,
        match="Dinamik return ifadesi güvenli biçimde hesaplanamadı",
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("round", "value"),
            candidate_values={"round": 1, "value": 12.5},
        )


@pytest.mark.parametrize(
    ("shadow_node_label", "shadow_node_type"),
    [
        ("round = 1", "Assign"),
        ("from helpers import round", "ImportFrom"),
    ],
)
def test_generate_rejects_round_shadowed_in_function_path(
    shadow_node_label: str,
    shadow_node_type: str,
) -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            shadow_node_label,
            "return round(value)",
            "END",
        ],
        node_types=["start", shadow_node_type, "return", "end"],
        edge_labels=[None, None, None],
    )

    with pytest.raises(
        ValueError,
        match="Dinamik return ifadesi güvenli biçimde hesaplanamadı",
    ):
        PathInputGenerator().generate(
            path=path,
            parameter_names=("value",),
            candidate_values={"value": 12.5},
        )


@pytest.mark.parametrize(
    ("assignment", "expected_source"),
    [
        ("counter = source", 1),
        ("counter = source + 1", 0),
        ("counter = 1 + source", 0),
        ("counter = source - 1", 2),
        ("counter = 10 - source", 9),
        ("counter = -source", -1),
    ],
)
def test_generate_back_propagates_affine_local_while_initialization(
    assignment: str,
    expected_source: int,
) -> None:
    path = create_derived_local_while_path(
        assignment=assignment,
        condition="counter > 0",
        updates=("counter -= 1",),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
        parameter_types={"source": "int"},
    )

    assert result.keyword_argument_dict == {"source": expected_source}
    assert isinstance(result.keyword_argument_dict["source"], int)


def test_generate_back_propagates_affine_local_while_false_edge() -> None:
    path = create_derived_local_while_path(
        assignment="counter = source + 1",
        condition="counter > 0",
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
    )

    assert result.keyword_argument_dict == {"source": -1}


def test_generate_replays_multiple_affine_local_while_updates() -> None:
    path = create_derived_local_while_path(
        assignment="counter = source + 1",
        condition="counter > 0",
        updates=("counter -= 1", "counter -= 1"),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
    )

    assert result.keyword_argument_dict == {"source": 1}


def test_generate_replays_incrementing_affine_local_while() -> None:
    path = create_derived_local_while_path(
        assignment="counter = source",
        condition="counter < 2",
        updates=("counter += 1",),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
    )

    assert result.keyword_argument_dict == {"source": 1}


def test_generate_affine_local_while_break_needs_no_back_edge_update() -> None:
    path = create_derived_local_while_path(
        assignment="counter = source + 1",
        condition="counter > 0",
        break_after_updates=True,
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
    )

    assert set(result.keyword_argument_dict) == {"source"}
    assert result.keyword_argument_dict["source"] + 1 > 0
    assert path.loop_iteration_count == 0
    assert path.is_zero_iteration_loop_path is False


@pytest.mark.parametrize(
    ("condition", "desired_result"),
    [
        ("counter > 2", True),
        ("counter >= 2", True),
        ("counter < 2", True),
        ("counter <= 2", True),
        ("counter == 2", True),
        ("counter != 2", True),
        ("counter > 2", False),
        ("counter >= 2", False),
        ("counter < 2", False),
        ("counter <= 2", False),
        ("counter == 2", False),
        ("counter != 2", False),
    ],
)
def test_generate_inverts_all_affine_while_comparisons(
    condition: str,
    desired_result: bool,
) -> None:
    path = create_derived_local_while_path(
        assignment="counter = source + 1",
        condition=condition,
        break_after_updates=desired_result,
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("source",),
    )
    actual = eval(
        condition,
        {"__builtins__": {}},
        {"counter": result.keyword_argument_dict["source"] + 1},
    )

    assert actual is desired_result


def test_generate_replays_update_before_affine_while_break() -> None:
    path = create_derived_local_while_path(
        assignment="counter = source + 1",
        condition="counter > 0",
        updates=("counter -= 1",),
        break_after_updates=True,
    )

    result = PathInputGenerator().generate(path, ("source",))

    assert set(result.keyword_argument_dict) == {"source"}
    assert result.keyword_argument_dict["source"] + 1 > 0


def test_generate_keeps_separate_affine_while_activations() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 5, 7, 8],
        edge_labels=[None, None, "False", None, "True", "Loop", "False", None],
        node_labels=[
            "START",
            "counter = first + 1",
            "counter > 0",
            "counter = second + 1",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "return first + second",
            "END",
        ],
        node_types=[
            "start", "Assign", "while", "Assign", "while",
            "AugAssign", "while", "return", "end",
        ],
        line_numbers=list(range(1, 10)),
    )

    result = PathInputGenerator().generate(path, ("first", "second"))

    assert result.keyword_argument_dict == {"first": -1, "second": 0}


def test_generate_supports_affine_while_inside_for() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 4, 2, 6, 7],
        edge_labels=[None, "Iterate", None, "True", "Loop", "False", "Complete", None],
        node_labels=[
            "START", "item in values", "counter = limit + 1",
            "counter > 0", "counter -= 1", "counter > 0",
            "item in values", "return limit", "END",
        ],
        node_types=[
            "start", "for", "Assign", "while", "AugAssign",
            "while", "for", "return", "end",
        ],
        line_numbers=list(range(1, 10)),
    )

    result = PathInputGenerator().generate(path, ("values", "limit"))

    assert result.keyword_argument_dict == {"values": [0], "limit": 0}


def test_generate_supports_nested_affine_while_loops() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 5, 7, 3, 8, 9],
        edge_labels=[
            None, None, "True", None, "True", "Loop", "False",
            "Loop", "False", None,
        ],
        node_labels=[
            "START", "outer = source + 1", "outer > 0",
            "inner = source + 1", "inner > 0", "inner -= 1",
            "inner > 0", "outer -= 1", "outer > 0", "return source", "END",
        ],
        node_types=[
            "start", "Assign", "while", "Assign", "while", "AugAssign",
            "while", "AugAssign", "while", "return", "end",
        ],
        line_numbers=list(range(1, 12)),
    )

    result = PathInputGenerator().generate(path, ("source",))

    assert result.keyword_argument_dict == {"source": 0}


@pytest.mark.parametrize(
    "assignment",
    [
        "counter = first + second",
        "counter = source * source",
    ],
)
def test_generate_classifies_unsupported_local_while_initialization(
    assignment: str,
) -> None:
    parameter_names = (
        ("first", "second")
        if "first" in assignment
        else ("source",)
    )
    path = create_derived_local_while_path(
        assignment=assignment,
        condition="counter > 0",
        break_after_updates=True,
    )

    with pytest.raises(
        UnsupportedInputSynthesisError,
        match="Derived while|desteklenmeyen provenance",
    ):
        PathInputGenerator().generate(path, parameter_names)


def test_generate_keeps_affine_while_constraint_conflicts_unreachable() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6],
        edge_labels=[None, "True", None, "False", None],
        node_labels=[
            "START", "source >= 0", "counter = source + 1",
            "counter > 0", "return source", "END",
        ],
        node_types=["start", "if", "Assign", "while", "return", "end"],
        line_numbers=list(range(1, 7)),
    )

    with pytest.raises(UnreachablePathError):
        PathInputGenerator().generate(path, ("source",))


def test_robustness_remaining_attempts_affine_initialization_is_accepted() -> None:
    path = create_derived_local_while_path(
        assignment="remaining_attempts = retry_count + 1",
        condition="remaining_attempts > 0",
        updates=("remaining_attempts -= 1",),
    )
    path.node_labels[-2] = "return retry_count"

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("retry_count",),
        parameter_types={"retry_count": "int"},
    )

    assert result.keyword_argument_dict == {"retry_count": 0}


@pytest.mark.parametrize(
    ("condition", "edge_label", "expected_dict"),
    [
        ("isinstance(value, dict)", "True", True),
        ("isinstance(value, dict)", "False", False),
        ("not isinstance(value, dict)", "True", False),
        ("not isinstance(value, dict)", "False", True),
    ],
)
def test_generate_applies_safe_isinstance_parameter_constraint(
    condition: str,
    edge_label: str,
    expected_dict: bool,
) -> None:
    path = create_execution_path(
        node_labels=["START", condition, "return 0", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, edge_label, None],
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("value",),
        parameter_types={"value": "dict[str, int]"},
    )

    assert isinstance(result.keyword_argument_dict["value"], dict) is expected_dict


def test_generate_applies_isinstance_to_loop_iteration_element() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 2, 5, 6],
        edge_labels=[None, "Iterate", "True", "Continue", "Complete", None],
        node_labels=[
            "START", "item in values", "not isinstance(item, dict)",
            "continue", "item in values", "return 0", "END",
        ],
        node_types=[
            "start", "for", "if", "continue", "for", "return", "end",
        ],
        line_numbers=list(range(1, 8)),
    )

    result = PathInputGenerator().generate(
        path=path,
        parameter_names=("values",),
        parameter_types={"values": "list[dict[str, int]]"},
    )

    assert len(result.keyword_argument_dict["values"]) == 1
    assert not isinstance(result.keyword_argument_dict["values"][0], dict)


@pytest.mark.parametrize(
    ("type_name", "expected_type"),
    [
        ("int", int),
        ("float", float),
        ("str", str),
        ("bool", bool),
        ("list", list),
        ("tuple", tuple),
        ("set", set),
        ("dict", dict),
    ],
)
def test_generate_supports_each_safe_isinstance_type(
    type_name: str,
    expected_type: type[object],
) -> None:
    path = create_execution_path(
        node_labels=[
            "START", f"isinstance(value, {type_name})", "return 0", "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    value = PathInputGenerator().generate(path, ("value",)).keyword_argument_dict[
        "value"
    ]

    assert type(value) is expected_type


@pytest.mark.parametrize("edge_label", ["True", "False"])
def test_generate_supports_isinstance_type_tuple(edge_label: str) -> None:
    path = create_execution_path(
        node_labels=[
            "START", "isinstance(value, (int, str))", "return 0", "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, edge_label, None],
    )

    value = PathInputGenerator().generate(path, ("value",)).keyword_argument_dict[
        "value"
    ]

    assert isinstance(value, (int, str)) is (edge_label == "True")


def test_generate_keeps_isinstance_constraints_iteration_scoped() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 2, 5, 6, 2, 7, 8],
        edge_labels=[
            None, "Iterate", "True", "Continue", "Iterate",
            "True", "Next", "Complete", None,
        ],
        node_labels=[
            "START", "item in values", "not isinstance(item, dict)",
            "continue", "item in values", "isinstance(item, dict)",
            "seen = 1", "item in values", "return 0", "END",
        ],
        node_types=[
            "start", "for", "if", "continue", "for", "if",
            "Assign", "for", "return", "end",
        ],
        line_numbers=list(range(1, 11)),
    )

    values = PathInputGenerator().generate(
        path,
        ("values",),
        {"values": "list[dict[str, int]]"},
    ).keyword_argument_dict["values"]

    assert len(values) == 2
    assert not isinstance(values[0], dict)
    assert isinstance(values[1], dict)


def test_generate_routes_isinstance_to_innermost_nested_loop_binding() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 3, 5, 2, 6, 7],
        edge_labels=[
            None, "Iterate", "Iterate", "True", "Complete",
            "True", "Complete", None,
        ],
        node_labels=[
            "START", "item in outer_values", "item in inner_values",
            "isinstance(item, str)", "item in inner_values",
            "isinstance(item, dict)", "item in outer_values", "return 0", "END",
        ],
        node_types=[
            "start", "for", "for", "if", "for", "if", "for", "return", "end",
        ],
        line_numbers=list(range(1, 10)),
    )

    values = PathInputGenerator().generate(
        path,
        ("outer_values", "inner_values"),
    ).keyword_argument_dict

    assert isinstance(values["outer_values"][0], dict)
    assert isinstance(values["inner_values"][0], str)


def test_generate_propagates_isinstance_through_direct_local_alias() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "alias = value", "isinstance(alias, tuple)",
            "return 0", "END",
        ],
        node_types=["start", "Assign", "if", "return", "end"],
        edge_labels=[None, None, "True", None],
    )

    value = PathInputGenerator().generate(path, ("value",)).keyword_argument_dict[
        "value"
    ]

    assert isinstance(value, tuple)


def test_generate_propagates_isinstance_through_static_subscript_alias() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "alias = values[0]", "isinstance(alias, set)",
            "return 0", "END",
        ],
        node_types=["start", "Assign", "if", "return", "end"],
        edge_labels=[None, None, "True", None],
    )

    values = PathInputGenerator().generate(
        path,
        ("values",),
        {"values": "list[int]"},
    ).keyword_argument_dict["values"]

    assert isinstance(values[0], set)


def test_generate_runtime_predicate_selects_union_compatible_alternative() -> None:
    path = create_execution_path(
        node_labels=["START", "isinstance(value, str)", "return 0", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    value = PathInputGenerator().generate(
        path,
        ("value",),
        {"value": "int | str"},
    ).keyword_argument_dict["value"]

    assert isinstance(value, str)


def test_generate_combines_isinstance_with_truthiness() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "isinstance(value, str)", "value", "return 0", "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "True", None],
    )

    value = PathInputGenerator().generate(path, ("value",)).keyword_argument_dict[
        "value"
    ]

    assert isinstance(value, str)
    assert value


def test_generate_combines_isinstance_with_numeric_range() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "isinstance(value, int)", "value > 5", "return 0", "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "True", None],
    )

    value = PathInputGenerator().generate(path, ("value",)).keyword_argument_dict[
        "value"
    ]

    assert isinstance(value, int)
    assert value > 5


def test_generate_rejects_contradictory_isinstance_constraints() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "isinstance(value, dict)", "isinstance(value, dict)",
            "return 0", "END",
        ],
        node_types=["start", "if", "if", "return", "end"],
        edge_labels=[None, "True", "False", None],
    )

    with pytest.raises(UnreachablePathError, match="type kısıtları"):
        PathInputGenerator().generate(path, ("value",))


@pytest.mark.parametrize(
    "condition",
    [
        "isinstance(value, CustomClass)",
        "isinstance(value, module.CustomClass)",
        "isinstance(value, dynamic_type)",
        "isinstance(value)",
        "isinstance(value, dict, str)",
        "isinstance(value=value, class_or_tuple=dict)",
        "isinstance(*values)",
        "obj.isinstance(value, dict)",
        "module.isinstance(value, dict)",
    ],
)
def test_generate_rejects_unsafe_isinstance_call_shapes(condition: str) -> None:
    path = create_execution_path(
        node_labels=["START", condition, "return 0", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError):
        PathInputGenerator().generate(path, ("value", "values"))


@pytest.mark.parametrize(
    ("prefix_labels", "prefix_types", "parameter_names"),
    [
        ((), (), ("isinstance", "value")),
        (("isinstance = predicate",), ("Assign",), ("value", "predicate")),
        (("from helpers import isinstance",), ("ImportFrom",), ("value",)),
        (("isinstance in predicates",), ("for",), ("value", "predicates")),
    ],
)
def test_generate_rejects_shadowed_isinstance(
    prefix_labels: tuple[str, ...],
    prefix_types: tuple[str, ...],
    parameter_names: tuple[str, ...],
) -> None:
    path = create_execution_path(
        node_labels=[
            "START", *prefix_labels, "isinstance(value, dict)", "return 0", "END",
        ],
        node_types=["start", *prefix_types, "if", "return", "end"],
        edge_labels=[None] * (len(prefix_labels) + 1) + ["True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="gölgeleniyor"):
        PathInputGenerator().generate(path, parameter_names)


def test_robustness_transaction_predicate_overrides_element_annotation() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 2, 5, 6],
        edge_labels=[None, "Iterate", "True", "Continue", "Complete", None],
        node_labels=[
            "START", "transaction in transactions",
            "not isinstance(transaction, dict)", "continue",
            "transaction in transactions", "return 0", "END",
        ],
        node_types=[
            "start", "for", "if", "continue", "for", "return", "end",
        ],
        line_numbers=list(range(1, 8)),
    )

    transactions = PathInputGenerator().generate(
        path,
        ("transactions",),
        {"transactions": "list[dict[str, int | str]]"},
    ).keyword_argument_dict["transactions"]

    assert len(transactions) == 1
    assert not isinstance(transactions[0], dict)


@pytest.mark.parametrize(
    ("condition", "edge_label", "expect_none"),
    [
        ("value is None", "True", True),
        ("value is None", "False", False),
        ("value is not None", "True", False),
        ("value is not None", "False", True),
        ("None is value", "True", True),
        ("None is value", "False", False),
        ("None is not value", "True", False),
        ("None is not value", "False", True),
    ],
)
def test_generate_supports_none_identity_edges(
    condition: str,
    edge_label: str,
    expect_none: bool,
) -> None:
    path = create_execution_path(
        node_labels=["START", condition, "return 0", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, edge_label, None],
    )

    value = PathInputGenerator().generate(
        path, ("value",), {"value": "int | None"}
    ).keyword_argument_dict["value"]

    assert (value is None) is expect_none


@pytest.mark.parametrize(
    "condition",
    [
        "left is right",
        "left is not right",
        "value is True",
        "value is False",
        "value is singleton.member",
    ],
)
def test_generate_rejects_non_none_identity(condition: str) -> None:
    path = create_execution_path(
        node_labels=["START", condition, "return 0", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="identity|None"):
        PathInputGenerator().generate(
            path, ("left", "right", "value", "singleton")
        )


def test_generate_replays_verified_empty_collection_local_state() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6],
        edge_labels=[None, None, "False", "True", None],
        node_labels=[
            "START",
            "__self_values = []",
            "key in __self_values",
            "not __self_values",
            "return 'empty'",
            "END",
        ],
        node_types=["start", "Assign", "if", "if", "return", "end"],
        line_numbers=[1, 2, 3, 4, 5, 6],
    )

    generated = PathInputGenerator().generate(
        path=path,
        parameter_names=("key",),
        parameter_types={"key": "str"},
    )

    assert generated.expected_result == "empty"
    assert set(generated.keyword_argument_dict) == {"key"}
    assert isinstance(generated.keyword_argument_dict["key"], str)


@pytest.mark.parametrize("literal", ("{}", "[]", "()"))
def test_generate_replays_constructor_proven_empty_literal_condition(
    literal: str,
) -> None:
    reachable = create_execution_path(
        node_labels=["START", f"not {literal}", "return 'empty'", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )
    contradictory = create_execution_path(
        node_labels=["START", f"not {literal}", "return 'other'", "END"],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "False", None],
    )

    generated = PathInputGenerator().generate(reachable, ())

    assert generated.keyword_arguments == ()
    assert generated.expected_result == "empty"
    with pytest.raises(UnreachablePathError, match="boş koleksiyon"):
        PathInputGenerator().generate(contradictory, ())


def test_generate_accepts_only_zero_iteration_for_fixed_empty_collection() -> None:
    complete_path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5],
        edge_labels=[None, None, "Complete", None],
        node_labels=[
            "START", "values = []", "value in values", "return 0", "END",
        ],
        node_types=["start", "Assign", "for", "return", "end"],
        line_numbers=[1, 2, 3, 4, 5],
    )
    iterate_path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5],
        edge_labels=[None, None, "Iterate", None],
        node_labels=[
            "START", "values = []", "value in values", "return 1", "END",
        ],
        node_types=["start", "Assign", "for", "return", "end"],
        line_numbers=[1, 2, 3, 4, 5],
    )

    generated = PathInputGenerator().generate(
        path=complete_path,
        parameter_names=(),
    )

    assert generated.expected_result == 0
    with pytest.raises(UnreachablePathError, match="boş koleksiyon"):
        PathInputGenerator().generate(
            path=iterate_path,
            parameter_names=(),
        )


def _create_dict_get_path(
    *,
    lookup: str = 'mapping.get("wanted")',
    condition: str = "lookup_result is None",
    edge_label: str = "True",
    prefix_labels: tuple[str, ...] = (),
    prefix_types: tuple[str, ...] = (),
    return_label: str = "return 0",
) -> ExecutionPath:
    return create_execution_path(
        node_labels=[
            "START", *prefix_labels,
            f"lookup_result = {lookup}", condition, return_label, "END",
        ],
        node_types=[
            "start", *prefix_types,
            "Assign", "if", "return", "end",
        ],
        edge_labels=[
            None, *(None for _ in prefix_labels), None, edge_label, None,
        ],
    )


def test_generate_dict_get_none_uses_absent_key() -> None:
    result = PathInputGenerator().generate(
        _create_dict_get_path(),
        ("mapping",),
        {"mapping": "dict[str, int]"},
    )

    assert "wanted" not in result.keyword_argument_dict["mapping"]


def test_generate_dict_get_non_none_creates_typed_value() -> None:
    result = PathInputGenerator().generate(
        _create_dict_get_path(condition="lookup_result is not None"),
        ("mapping",),
        {"mapping": "dict[str, int]"},
    )

    assert type(result.keyword_argument_dict["mapping"]["wanted"]) is int


def test_generate_truthy_dict_get_absence_uses_deterministic_sentinel() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "mapping", 'lookup_result = mapping.get("wanted")',
            "lookup_result is None", "return 0", "END",
        ],
        node_types=["start", "if", "Assign", "if", "return", "end"],
        edge_labels=[None, "True", None, "True", None],
    )

    first = PathInputGenerator().generate(
        path, ("mapping",), {"mapping": "dict[str, int]"}
    ).keyword_argument_dict["mapping"]
    second = PathInputGenerator().generate(
        path, ("mapping",), {"mapping": "dict[str, int]"}
    ).keyword_argument_dict["mapping"]

    assert first == second
    assert first
    assert "wanted" not in first
    assert all(type(value) is int for value in first.values())


def test_generate_dict_get_default_and_expected_result_replay() -> None:
    result = PathInputGenerator().generate(
        _create_dict_get_path(
            lookup='mapping.get("wanted", 7)',
            condition="lookup_result is not None",
            return_label="return lookup_result",
        ),
        ("mapping",),
        {"mapping": "dict[str, int]"},
    )

    assert result.expected_result == 7


@pytest.mark.parametrize(
    ("prefix_labels", "prefix_types", "lookup"),
    [
        ((), (), "mapping.get(key)"),
        (("key_alias = key",), ("Assign",), "mapping.get(key_alias)"),
        (("key_alias = keys[0]",), ("Assign",), "mapping.get(key_alias)"),
    ],
)
def test_generate_dict_get_preserves_dynamic_key_provenance(
    prefix_labels: tuple[str, ...],
    prefix_types: tuple[str, ...],
    lookup: str,
) -> None:
    parameter_names = (
        ("mapping", "key", "keys")
        if "keys[" in " ".join(prefix_labels)
        else ("mapping", "key")
    )
    candidate_values = {"key": "category", "keys": ["category"]}
    result = PathInputGenerator().generate(
        _create_dict_get_path(
            lookup=lookup,
            condition="lookup_result is not None",
            prefix_labels=prefix_labels,
            prefix_types=prefix_types,
        ),
        parameter_names,
        {"mapping": "dict[str, int]", **(
            {"key": "str"} if "key" in parameter_names else {}
        ), **({"keys": "list[str]"} if "keys" in parameter_names else {})},
        candidate_values,
    )
    arguments = result.keyword_argument_dict
    key = arguments.get("key", arguments.get("keys", [None])[0])

    assert key in arguments["mapping"]


def test_generate_synthesizes_loop_element_dictionary_lookup() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10, 11],
        edge_labels=[
            None, "Iterate", "False", None, None, "False", "False",
            None, "True", "Complete", None,
        ],
        node_labels=[
            "START", "transaction in transactions",
            "not isinstance(transaction, dict)",
            'category = transaction["category"]',
            'amount = transaction["amount"]',
            "not isinstance(category, str)",
            "not isinstance(amount, int)",
            "category_limit = limits.get(category)",
            "category_limit is not None",
            "transaction in transactions", "return 0", "END",
        ],
        node_types=[
            "start", "for", "if", "Assign", "Assign", "if", "if",
            "Assign", "if", "for", "return", "end",
        ],
        line_numbers=list(range(1, 13)),
    )

    arguments = PathInputGenerator().generate(
        path,
        ("transactions", "limits"),
        {
            "transactions": "list[dict[str, int | str]]",
            "limits": "dict[str, int]",
        },
    ).keyword_argument_dict
    transaction = arguments["transactions"][0]

    assert type(transaction) is dict
    assert type(transaction["category"]) is str
    assert type(transaction["amount"]) is int
    assert transaction["category"] in arguments["limits"]


def test_generate_keeps_structured_locals_separate_between_iterations() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 6, 7],
        edge_labels=[
            None, "Iterate", "True", None, "True", "Iterate",
            "True", None, "False", "Complete", None,
        ],
        node_labels=[
            "START", "item in items", "isinstance(item, dict)",
            'category = item["category"]', 'category == "A"',
            "item in items", "isinstance(item, dict)",
            'category = item["category"]', 'category == "A"',
            "item in items", "return 0", "END",
        ],
        node_types=[
            "start", "for", "if", "Assign", "if", "for", "if",
            "Assign", "if", "for", "return", "end",
        ],
        line_numbers=list(range(1, 13)),
    )

    items = PathInputGenerator().generate(
        path,
        ("items",),
        {"items": "list[dict[str, str]]"},
    ).keyword_argument_dict["items"]

    assert items[0]["category"] == "A"
    assert items[1]["category"] != "A"


def test_generate_nested_loop_dictionary_key_error_input() -> None:
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5, 6, 7],
        edge_labels=[None, "Iterate", "True", "Exception", None, None],
        node_labels=[
            "START", "item in items", "isinstance(item, dict)",
            'value = item["required"]', "except KeyError",
            'return "missing"', "END",
        ],
        node_types=[
            "start", "for", "if", "Assign", "except", "return", "end",
        ],
        line_numbers=list(range(1, 8)),
    )

    result = PathInputGenerator().generate(
        path,
        ("items",),
        {"items": "list[dict[str, int]]"},
    )

    assert "required" not in result.keyword_argument_dict["items"][0]
    assert result.expected_result == "missing"


def test_generate_rejects_dictionary_key_present_absent_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START", 'first = mapping["key"]',
            'second = mapping["key"]', "except KeyError", "return 0", "END",
        ],
        node_types=[
            "start", "Assign", "Assign", "except", "return", "end",
        ],
        edge_labels=[None, None, "Exception", None, None],
    )

    with pytest.raises(UnreachablePathError, match="present ve absent"):
        PathInputGenerator().generate(
            path,
            ("mapping",),
            {"mapping": "dict[str, int]"},
        )


@pytest.mark.parametrize(
    "lookup",
    [
        'receiver.get("key")',
        'holder.mapping.get("key")',
        'mapping.get(key="key")',
        'mapping.get(*keys)',
        'mapping.get("key", **options)',
        'mapping.other("key")',
    ],
)
def test_generate_rejects_unsafe_dict_get_shapes(lookup: str) -> None:
    path = _create_dict_get_path(
        lookup=lookup,
        condition="lookup_result is not None",
    )

    with pytest.raises(UnsupportedInputSynthesisError, match="dict|get|lookup"):
        PathInputGenerator().generate(
            path,
            ("mapping", "receiver", "holder", "keys", "options"),
            {"mapping": "dict[str, int]"},
        )


def test_generate_classifies_unsupported_affine_comparison_as_domain_error() -> None:
    path = create_execution_path(
        node_labels=[
            "START", "category_total > category_limit * 2", "return 0", "END",
        ],
        node_types=["start", "if", "return", "end"],
        edge_labels=[None, "True", None],
    )

    with pytest.raises(
        UnsupportedInputSynthesisError,
        match="category_total > category_limit \\* 2",
    ):
        PathInputGenerator().generate(
            path,
            ("category_total", "category_limit"),
        )
