import pytest

from cfg.path_analyzer import ExecutionPath
from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathConstraint,
    PathFeasibilityAnalyzer,
)


def create_execution_path(
    *,
    node_labels: list[str],
    node_types: list[str],
    edge_labels: list[str | None],
) -> ExecutionPath:
    """
    PathFeasibilityAnalyzer testlerinde kullanılmak üzere
    metadata içeren ExecutionPath oluşturur.
    """
    node_count = len(node_labels)

    return ExecutionPath(
        node_ids=list(
            range(
                1,
                node_count + 1,
            )
        ),
        edge_labels=edge_labels,
        node_labels=node_labels,
        node_types=node_types,
        line_numbers=list(
            range(
                10,
                10 + node_count,
            )
        ),
    )


def test_path_constraint_can_be_created() -> None:
    constraint = PathConstraint(
        variable_name="score",
        operator=">=",
        value=50,
    )

    assert constraint.variable_name == "score"
    assert constraint.operator == ">="
    assert constraint.value == 50


def test_path_constraint_rejects_empty_variable_name() -> None:
    with pytest.raises(
        ValueError,
        match="variable_name boş olamaz.",
    ):
        PathConstraint(
            variable_name="",
            operator=">=",
            value=50,
        )


def test_path_constraint_rejects_unsupported_operator() -> None:
    with pytest.raises(
        ValueError,
        match="Desteklenmeyen constraint operatorü",
    ):
        PathConstraint(
            variable_name="score",
            operator="===",
            value=50,
        )


def test_feasible_range_is_detected() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=50,
            ),
            PathConstraint(
                variable_name="score",
                operator="<=",
                value=100,
            ),
        )
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.is_infeasible is False
    assert result.conflicts == ()


def test_exact_value_inside_range_is_feasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=40,
            ),
            PathConstraint(
                variable_name="score",
                operator="<=",
                value=80,
            ),
            PathConstraint(
                variable_name="score",
                operator="==",
                value=60,
            ),
        )
    )

    assert result.is_feasible is True


def test_exact_value_below_lower_bound_is_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="attendance",
                operator=">=",
                value=40,
            ),
            PathConstraint(
                variable_name="attendance",
                operator="==",
                value=0,
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert result.conflicts


def test_exact_value_above_upper_bound_is_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator="<=",
                value=100,
            ),
            PathConstraint(
                variable_name="score",
                operator="==",
                value=120,
            ),
        )
    )

    assert result.is_infeasible is True


def test_exact_value_cannot_also_be_excluded() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator="==",
                value=50,
            ),
            PathConstraint(
                variable_name="score",
                operator="!=",
                value=50,
            ),
        )
    )

    assert result.is_infeasible is True


def test_different_exact_values_are_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator="==",
                value=50,
            ),
            PathConstraint(
                variable_name="score",
                operator="==",
                value=60,
            ),
        )
    )

    assert result.is_infeasible is True


def test_lower_bound_greater_than_upper_bound_is_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=80,
            ),
            PathConstraint(
                variable_name="score",
                operator="<=",
                value=60,
            ),
        )
    )

    assert result.is_infeasible is True


def test_strict_equal_bounds_are_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">",
                value=50,
            ),
            PathConstraint(
                variable_name="score",
                operator="<=",
                value=50,
            ),
        )
    )

    assert result.is_infeasible is True


def test_multiple_variables_can_be_feasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=70,
            ),
            PathConstraint(
                variable_name="attendance",
                operator=">=",
                value=80,
            ),
            PathConstraint(
                variable_name="project_score",
                operator=">=",
                value=60,
            ),
        )
    )

    assert result.is_feasible is True


def test_conflict_in_one_variable_makes_path_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="score",
                operator=">=",
                value=70,
            ),
            PathConstraint(
                variable_name="attendance",
                operator=">=",
                value=40,
            ),
            PathConstraint(
                variable_name="attendance",
                operator="==",
                value=0,
            ),
        )
    )

    assert result.is_infeasible is True

    assert any(
        "attendance" in conflict
        for conflict in result.conflicts
    )


def test_empty_constraint_collection_is_feasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        ()
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.constraints == ()
    assert result.conflicts == ()


def test_analyzer_rejects_non_tuple_constraints() -> None:
    analyzer = PathFeasibilityAnalyzer()

    with pytest.raises(
        TypeError,
        match="constraints bir PathConstraint tuple'ı olmalıdır.",
    ):
        analyzer.analyze_constraints(
            []  # type: ignore[arg-type]
        )


def test_analyzer_rejects_invalid_constraint_element() -> None:
    analyzer = PathFeasibilityAnalyzer()

    with pytest.raises(
        TypeError,
        match="constraints yalnızca PathConstraint",
    ):
        analyzer.analyze_constraints(
            (
                "invalid",
            )  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "operator",
    (
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
    ),
)
def test_all_supported_operators_are_accepted(
    operator: str,
) -> None:
    constraint = PathConstraint(
        variable_name="value",
        operator=operator,
        value=10,
    )

    assert constraint.operator == operator


def test_extract_constraints_uses_true_edge_condition() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(
        path
    )

    assert result.constraints == (
        PathConstraint(
            variable_name="score",
            operator=">=",
            value=50.0,
        ),
    )

    assert result.unsupported_conditions == ()


def test_extract_constraints_negates_false_edge_condition() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "attendance < 40",
            "score >= 50",
        ],
        node_types=[
            "start",
            "if",
            "if",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(
        path
    )

    assert result.constraints == (
        PathConstraint(
            variable_name="attendance",
            operator=">=",
            value=40.0,
        ),
    )


def test_reversed_numeric_comparison_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "40 <= attendance",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(
        path
    )

    assert result.constraints == (
        PathConstraint(
            variable_name="attendance",
            operator=">=",
            value=40.0,
        ),
    )


def test_negative_numeric_constant_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "score >= -10",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(
        path
    )

    assert result.constraints == (
        PathConstraint(
            variable_name="score",
            operator=">=",
            value=-10.0,
        ),
    )


def test_unsupported_condition_results_in_unknown() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "custom_check(score)",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.is_unknown is True
    assert result.unsupported_conditions


def test_path_without_node_metadata_results_in_unknown() -> None:
    path = ExecutionPath(
        node_ids=[
            1,
            2,
        ],
        edge_labels=[
            None,
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.is_unknown is True
    assert result.unsupported_conditions


def test_path_32_style_conflict_is_detected_as_infeasible() -> None:
    """
    Gerçek multicomplex datasetinde bulduğumuz Path 32
    davranışının sentetik karşılığıdır.

    attendance < 40 koşulundan False çıkmak:
        attendance >= 40

    Daha sonra attendance == 0 koşulundan True çıkmak:
        attendance == 0

    Bu iki constraint birlikte mümkün değildir.
    """
    path = create_execution_path(
        node_labels=[
            "START",
            "attendance < 40",
            "score == 0",
            "attendance == 0",
            "return 'Başvuruya katılmadı'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True

    assert PathConstraint(
        variable_name="attendance",
        operator=">=",
        value=40.0,
    ) in result.constraints

    assert PathConstraint(
        variable_name="attendance",
        operator="==",
        value=0.0,
    ) in result.constraints

    assert result.conflicts


def test_unsupported_condition_does_not_hide_proven_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "custom_check(score)",
            "attendance < 40",
            "attendance == 0",
            "return 'END'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "False",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert result.conflicts
    assert result.unsupported_conditions


def test_analyze_paths_preserves_input_order() -> None:
    feasible_path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    infeasible_path = create_execution_path(
        node_labels=[
            "START",
            "score >= 50",
            "score < 40",
            "return 'NO'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "True",
        ],
    )

    analyzer = PathFeasibilityAnalyzer()

    results = analyzer.analyze_paths(
        (
            feasible_path,
            infeasible_path,
        )
    )

    assert len(results) == 2

    assert (
        results[0].status
        == FeasibilityStatus.FEASIBLE
    )

    assert (
        results[1].status
        == FeasibilityStatus.INFEASIBLE
    )


def test_analyze_path_rejects_invalid_path() -> None:
    analyzer = PathFeasibilityAnalyzer()

    with pytest.raises(
        TypeError,
        match="path bir ExecutionPath örneği olmalıdır.",
    ):
        analyzer.analyze_path(
            "invalid"  # type: ignore[arg-type]
        )


def test_analyze_paths_rejects_non_tuple_collection() -> None:
    analyzer = PathFeasibilityAnalyzer()

    with pytest.raises(
        TypeError,
        match="paths bir ExecutionPath tuple'ı olmalıdır.",
    ):
        analyzer.analyze_paths(
            []  # type: ignore[arg-type]
        )


def test_analyze_paths_rejects_invalid_path_element() -> None:
    analyzer = PathFeasibilityAnalyzer()

    with pytest.raises(
        TypeError,
        match="paths yalnızca ExecutionPath",
    ):
        analyzer.analyze_paths(
            (
                "invalid",
            )  # type: ignore[arg-type]
        )


# ============================================================
# PathFeasibilityAnalyzer v2 tests
# ============================================================


def test_string_equality_constraint_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "customer_type == 'VIP'",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.unsupported_conditions == ()

    assert PathConstraint(
        variable_name="customer_type",
        operator="==",
        value="VIP",
    ) in result.constraints


def test_false_string_equality_becomes_inequality() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "customer_type == 'VIP'",
            "return 'OTHER'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="customer_type",
            operator="!=",
            value="VIP",
        ),
    )


def test_conflicting_string_equalities_are_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="customer_type",
                operator="==",
                value="VIP",
            ),
            PathConstraint(
                variable_name="customer_type",
                operator="==",
                value="MEMBER",
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_string_equality_and_inequality_conflict_is_detected() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="coupon",
                operator="==",
                value="SAVE10",
            ),
            PathConstraint(
                variable_name="coupon",
                operator="!=",
                value="SAVE10",
            ),
        )
    )

    assert result.is_infeasible is True


def test_not_name_true_edge_becomes_falsy_constraint() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "return 'EMPTY'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="items",
            operator="falsy",
            value=True,
        ),
    )

    assert result.unsupported_conditions == ()


def test_not_name_false_edge_becomes_truthy_constraint() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "not items",
            "return 'CONTINUE'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="items",
            operator="truthy",
            value=True,
        ),
    )


def test_truthy_and_falsy_constraints_conflict() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="items",
                operator="truthy",
                value=True,
            ),
            PathConstraint(
                variable_name="items",
                operator="falsy",
                value=True,
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_true_and_condition_expands_all_operands() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "amount >= 100 and stock > 0",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.unsupported_conditions == ()

    assert PathConstraint(
        variable_name="amount",
        operator=">=",
        value=100,
    ) in result.constraints

    assert PathConstraint(
        variable_name="stock",
        operator=">",
        value=0,
    ) in result.constraints


def test_false_and_condition_remains_unknown() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "amount >= 100 and stock > 0",
            "return 'OTHER'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.unsupported_conditions


def test_false_or_condition_expands_negated_operands() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon == 'SAVE5' or amount >= 500",
            "return 'OTHER'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.unsupported_conditions == ()

    assert PathConstraint(
        variable_name="coupon",
        operator="!=",
        value="SAVE5",
    ) in result.constraints

    assert PathConstraint(
        variable_name="amount",
        operator="<",
        value=500,
    ) in result.constraints


def test_true_or_condition_remains_unknown() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon == 'SAVE5' or amount >= 500",
            "return 'DISCOUNT'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.unsupported_conditions


def test_true_membership_constraint_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon in ('NONE', '')",
            "return 'OK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="coupon",
            operator="in",
            value=("NONE", ""),
        ),
    )


def test_true_not_in_membership_constraint_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon not in ('NONE', '')",
            "return 'INVALID'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="coupon",
            operator="not in",
            value=("NONE", ""),
        ),
    )


def test_false_not_in_becomes_in_constraint() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon not in ('NONE', '')",
            "return 'CONTINUE'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="coupon",
            operator="in",
            value=("NONE", ""),
        ),
    )


def test_membership_and_exact_value_conflict_is_detected() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=("NONE", ""),
            ),
            PathConstraint(
                variable_name="coupon",
                operator="==",
                value="SAVE10",
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_not_in_and_exact_value_conflict_is_detected() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="coupon",
                operator="not in",
                value=("NONE", ""),
            ),
            PathConstraint(
                variable_name="coupon",
                operator="==",
                value="NONE",
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_intersecting_membership_constraints_are_feasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=("NONE", "", "SAVE5"),
            ),
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=("NONE", "SAVE10"),
            ),
        )
    )

    assert result.status == FeasibilityStatus.FEASIBLE


def test_disjoint_membership_constraints_are_infeasible() -> None:
    analyzer = PathFeasibilityAnalyzer()

    result = analyzer.analyze_constraints(
        (
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=("NONE", ""),
            ),
            PathConstraint(
                variable_name="coupon",
                operator="in",
                value=("SAVE5", "SAVE10"),
            ),
        )
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_complex_and_with_string_and_numeric_constraints_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            (
                "customer_type == 'VIP' "
                "and discount >= 25 "
                "and average_item_value >= 50"
            ),
            "return 'VIP'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.unsupported_conditions == ()

    assert PathConstraint(
        variable_name="customer_type",
        operator="==",
        value="VIP",
    ) in result.constraints

    assert PathConstraint(
        variable_name="discount",
        operator=">=",
        value=25,
    ) in result.constraints

    assert PathConstraint(
        variable_name="average_item_value",
        operator=">=",
        value=50,
    ) in result.constraints


def test_variable_to_variable_comparison_remains_unknown() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
        ],
        node_types=[
            "start",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.unsupported_conditions
