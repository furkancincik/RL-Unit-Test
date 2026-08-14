import pytest

from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathConstraint,
    PathFeasibilityResult,
    RelationalConstraint,
)
from generator.input_candidate_generator import (
    InputCandidateGenerator,
)


def create_feasible_result(
    constraints: tuple[PathConstraint, ...] = (),
    relational_constraints: tuple[
        RelationalConstraint,
        ...
    ] = (),
) -> PathFeasibilityResult:
    return PathFeasibilityResult(
        status=FeasibilityStatus.FEASIBLE,
        constraints=constraints,
        conflicts=(),
        unsupported_conditions=(),
        alternative_groups=(),
        relational_constraints=relational_constraints,
    )


def test_exact_numeric_constraint_generates_candidate() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator="==",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict == {
        "amount": 100,
    }

    assert candidate.values[0].source == "literal_exact"


def test_exact_string_constraint_generates_candidate() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="customer_type",
                operator="==",
                value="VIP",
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict == {
        "customer_type": "VIP",
    }


def test_greater_equal_uses_boundary_value() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator=">=",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 100


def test_greater_than_uses_value_above_boundary() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator=">",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 101


def test_less_equal_uses_boundary_value() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator="<=",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 100


def test_less_than_uses_value_below_boundary() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator="<",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 99


def test_numeric_not_equal_generates_different_value() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator="!=",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] != 100


def test_string_not_equal_generates_different_value() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="customer_type",
                operator="!=",
                value="VIP",
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["customer_type"] != "VIP"


def test_membership_uses_first_allowed_value() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=(
                    "NONE",
                    "",
                ),
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["coupon"] == "NONE"


def test_not_in_generates_value_outside_set() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="coupon",
                operator="not in",
                value=(
                    "NONE",
                    "",
                ),
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["coupon"] not in {
        "NONE",
        "",
    }


def test_truthy_constraint_generates_true() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="items",
                operator="truthy",
                value=True,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["items"] is True


def test_falsy_constraint_generates_false() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="items",
                operator="falsy",
                value=True,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["items"] is False


def test_relational_witness_is_applied() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="stock",
                operator=">=",
                value=0,
            ),
            PathConstraint(
                variable_name="valid_item_count",
                operator=">=",
                value=0,
            ),
        ),
        relational_constraints=(
            RelationalConstraint(
                left_variable="stock",
                operator="<",
                right_variable="valid_item_count",
            ),
        ),
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result,
        relational_witness={
            "stock": 1.0,
            "valid_item_count": 2.0,
        },
    )

    assert candidate.value_dict == {
        "stock": 1.0,
        "valid_item_count": 2.0,
    }

    sources = {
        item.variable_name: item.source
        for item in candidate.values
    }

    assert sources == {
        "stock": "relational_witness",
        "valid_item_count": "relational_witness",
    }


def test_relational_witness_overrides_literal_candidate() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="stock",
                operator=">=",
                value=0,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result,
        relational_witness={
            "stock": 5.0,
        },
    )

    assert candidate.value_dict["stock"] == 5.0


def test_exact_value_has_priority_during_literal_merge() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator=">=",
                value=0,
            ),
            PathConstraint(
                variable_name="amount",
                operator="==",
                value=25,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 25


def test_multiple_numeric_lower_bounds_use_stronger_candidate() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="amount",
                operator=">=",
                value=10,
            ),
            PathConstraint(
                variable_name="amount",
                operator=">=",
                value=100,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["amount"] == 100


def test_multiple_numeric_upper_bounds_use_stronger_candidate() -> None:
    """
    Regression:
    score < 85 ve score < 50 birlikteyken aday 84 değil,
    iki üst sınırı da sağlayan 49 olmalıdır.
    """
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="score",
                operator="<",
                value=85,
            ),
            PathConstraint(
                variable_name="score",
                operator="<",
                value=50,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict["score"] == 49


def test_mixed_numeric_bounds_generate_value_inside_range() -> None:
    """
    50 <= score < 85 aralığında üretilen değer iki constraint'i
    birden sağlamalıdır.
    """
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="score",
                operator="<",
                value=85,
            ),
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=50,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    value = candidate.value_dict["score"]

    assert value >= 50
    assert value < 85
    assert value == 84


def test_mixed_numeric_bounds_are_order_independent() -> None:
    """
    Constraint sırası değişse de aynı geçerli aralık adayı
    üretilebilmelidir.
    """
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=50,
            ),
            PathConstraint(
                variable_name="score",
                operator="<",
                value=85,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    value = candidate.value_dict["score"]

    assert value >= 50
    assert value < 85


def test_non_feasible_result_is_rejected() -> None:
    result = PathFeasibilityResult(
        status=FeasibilityStatus.INFEASIBLE,
        constraints=(),
        conflicts=(
            "conflict",
        ),
        unsupported_conditions=(),
        alternative_groups=(),
        relational_constraints=(),
    )

    with pytest.raises(
        ValueError,
        match="Yalnızca FEASIBLE path",
    ):
        InputCandidateGenerator().generate(
            feasibility_result=result
        )


def test_invalid_feasibility_result_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="PathFeasibilityResult",
    ):
        InputCandidateGenerator().generate(
            feasibility_result="invalid",  # type: ignore[arg-type]
        )


# ============================================================
# PathInputGenerator'a aktarılacak somut aday değer testleri
# ============================================================


def test_path_input_values_exclude_truthy_placeholder() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="values",
                operator="truthy",
                value=True,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict == {
        "values": True,
    }
    assert candidate.path_input_value_dict == {}


def test_path_input_values_exclude_falsy_placeholder() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="values",
                operator="falsy",
                value=True,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.value_dict == {
        "values": False,
    }
    assert candidate.path_input_value_dict == {}


def test_path_input_values_preserve_exact_boolean() -> None:
    result = create_feasible_result(
        constraints=(
            PathConstraint(
                variable_name="enabled",
                operator="==",
                value=False,
            ),
        )
    )

    candidate = InputCandidateGenerator().generate(
        feasibility_result=result
    )

    assert candidate.path_input_value_dict == {
        "enabled": False,
    }

    assert candidate.values[0].source == (
        "literal_exact"
    )
