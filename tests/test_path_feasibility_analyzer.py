import pytest

from cfg.data_flow_analyzer import (
    DataFlowAnalyzer,
    DataFlowAnalysisResult,
    InferredNumericRange,
)

from cfg.path_analyzer import ExecutionPath
from cfg.path_state_analyzer import (
    PathSymbolicState,
    SymbolicVariableState,
)
from cfg.path_feasibility_analyzer import (
    ConstraintAlternativeGroup,
    FeasibilityStatus,
    PathConstraint,
    PathFeasibilityAnalyzer,
    RelationalConstraint,
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


def test_false_and_condition_creates_alternative_group() -> None:
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

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == ()
    assert result.unsupported_conditions == ()
    assert len(result.alternative_groups) == 1

    group = result.alternative_groups[0]

    assert group == ConstraintAlternativeGroup(
        alternatives=(
            (
                PathConstraint(
                    variable_name="amount",
                    operator="<",
                    value=100,
                ),
            ),
            (
                PathConstraint(
                    variable_name="stock",
                    operator="<=",
                    value=0,
                ),
            ),
        )
    )


def test_false_and_condition_can_be_feasible() -> None:
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

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.unsupported_conditions == ()
    assert result.alternative_groups


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


def test_true_or_condition_creates_alternative_group() -> None:
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

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == ()
    assert result.unsupported_conditions == ()
    assert len(result.alternative_groups) == 1

    group = result.alternative_groups[0]

    assert group == ConstraintAlternativeGroup(
        alternatives=(
            (
                PathConstraint(
                    variable_name="coupon",
                    operator="==",
                    value="SAVE5",
                ),
            ),
            (
                PathConstraint(
                    variable_name="amount",
                    operator=">=",
                    value=500,
                ),
            ),
        )
    )


def test_true_or_condition_can_be_feasible() -> None:
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

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.unsupported_conditions == ()
    assert result.alternative_groups


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


def test_variable_to_variable_comparison_is_feasible_with_concrete_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count >= 0",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.unsupported_conditions == ()

    assert result.relational_constraints == (
        RelationalConstraint(
            left_variable="stock",
            operator="<",
            right_variable="valid_item_count",
        ),
    )


# ============================================================
# PathFeasibilityAnalyzer v3 tests
# ============================================================


def test_alternative_group_rejects_empty_alternatives() -> None:
    with pytest.raises(
        ValueError,
        match="alternatives boş olamaz.",
    ):
        ConstraintAlternativeGroup(
            alternatives=(),
        )


def test_alternative_group_rejects_invalid_constraint_element() -> None:
    with pytest.raises(
        TypeError,
        match="alternatives yalnızca PathConstraint",
    ):
        ConstraintAlternativeGroup(
            alternatives=(
                (
                    "invalid",
                ),  # type: ignore[arg-type]
            )
        )


def test_false_and_becomes_infeasible_when_all_alternatives_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "amount >= 100",
            "stock > 0",
            "amount >= 100 and stock > 0",
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
            "True",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert result.conflicts
    assert result.alternative_groups


def test_true_or_becomes_infeasible_when_all_alternatives_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon != 'SAVE5'",
            "amount < 500",
            "coupon == 'SAVE5' or amount >= 500",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert result.conflicts
    assert result.alternative_groups


def test_true_or_uses_feasible_alternative_when_one_branch_conflicts() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon != 'SAVE5'",
            "coupon == 'SAVE5' or amount >= 500",
            "return 'END'",
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

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.alternative_groups


def test_multiple_alternative_groups_are_combined_across_path() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "coupon == 'SAVE5' or amount >= 500",
            "customer_type == 'VIP' or stock > 0",
            "return 'END'",
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

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert len(result.alternative_groups) == 2
    assert result.unsupported_conditions == ()


def test_nested_boolean_expression_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            (
                "(customer_type == 'VIP' and amount >= 100) "
                "or coupon == 'SAVE10'"
            ),
            "return 'END'",
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
    assert result.alternative_groups


def test_nested_boolean_false_branch_is_supported() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            (
                "(customer_type == 'VIP' and amount >= 100) "
                "or coupon == 'SAVE10'"
            ),
            "return 'END'",
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

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.unsupported_conditions == ()


def test_relational_condition_with_alternative_group_is_feasible_with_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count >= 0",
            "stock < valid_item_count",
            "coupon == 'SAVE5' or amount >= 500",
            "return 'END'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "True",
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.unsupported_conditions == ()
    assert result.alternative_groups
    assert result.relational_constraints


# ============================================================
# PathFeasibilityAnalyzer v4.1 tests
# ============================================================


def test_relational_constraint_can_be_created() -> None:
    constraint = RelationalConstraint(
        left_variable="stock",
        operator="<",
        right_variable="valid_item_count",
    )

    assert constraint.left_variable == "stock"
    assert constraint.operator == "<"
    assert constraint.right_variable == "valid_item_count"


@pytest.mark.parametrize(
    "field_name, kwargs",
    (
        (
            "left_variable",
            {
                "left_variable": "",
                "operator": "<",
                "right_variable": "y",
            },
        ),
        (
            "right_variable",
            {
                "left_variable": "x",
                "operator": "<",
                "right_variable": "",
            },
        ),
    ),
)
def test_relational_constraint_rejects_empty_variable_name(
    field_name: str,
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} boş olamaz.",
    ):
        RelationalConstraint(**kwargs)


def test_relational_constraint_rejects_unsupported_operator() -> None:
    with pytest.raises(
        ValueError,
        match="Desteklenmeyen relational operator",
    ):
        RelationalConstraint(
            left_variable="x",
            operator="===",
            right_variable="y",
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
def test_all_relational_operators_are_supported(
    operator: str,
) -> None:
    constraint = RelationalConstraint(
        left_variable="x",
        operator=operator,
        right_variable="y",
    )

    assert constraint.operator == operator


def test_extract_constraints_recognizes_relational_comparison() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count >= 0",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.constraints == (
        PathConstraint(
            variable_name="stock",
            operator=">=",
            value=0.0,
        ),
        PathConstraint(
            variable_name="valid_item_count",
            operator=">=",
            value=0.0,
        ),
    )
    assert result.alternative_groups == ()
    assert result.unsupported_conditions == ()

    assert result.relational_constraints == (
        RelationalConstraint(
            left_variable="stock",
            operator="<",
            right_variable="valid_item_count",
        ),
    )


def test_false_relational_edge_is_negated() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock < valid_item_count",
            "return 'ENOUGH_STOCK'",
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

    assert result.relational_constraints == (
        RelationalConstraint(
            left_variable="stock",
            operator=">=",
            right_variable="valid_item_count",
        ),
    )


def test_relational_exact_values_can_prove_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 10",
            "valid_item_count == 5",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert result.conflicts
    assert result.relational_constraints


def test_relational_exact_values_can_prove_feasible_result() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 2",
            "valid_item_count == 5",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    # v4.5: exact numeric değerler relation'ı kesin biçimde
    # doğruluyorsa relational kısım PROVEN_TRUE kabul edilir.
    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()
    assert result.unsupported_conditions == ()


def test_relational_bounds_can_prove_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 10",
            "valid_item_count <= 5",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_strict_relational_equal_bounds_are_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 5",
            "valid_item_count <= 5",
            "stock < valid_item_count",
            "return 'LOW_STOCK'",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_greater_than_relational_conflict_is_detected_by_reversal() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 10",
            "x > y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_equal_relational_exact_values_can_be_consistent() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x == 7",
            "y == 7",
            "x == y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()
    assert result.relational_constraints


def test_equal_relational_exact_values_can_prove_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x == 7",
            "y == 8",
            "x == y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_duplicate_relational_constraints_are_deduplicated() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x < y",
            "x < y",
            "return 'END'",
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

    result = PathFeasibilityAnalyzer().extract_constraints(path)

    assert result.relational_constraints == (
        RelationalConstraint(
            left_variable="x",
            operator="<",
            right_variable="y",
        ),
    )


# ============================================================
# PathFeasibilityAnalyzer v4.2 data-flow integration tests
# ============================================================


def test_data_flow_lower_bound_can_prove_relational_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock <= -1",
            "stock >= valid_item_count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="valid_item_count",
                lower_bound=0.0,
                upper_bound=None,
            ),
        ),
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts
    assert result.relational_constraints


def test_data_flow_upper_bound_can_prove_relational_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 10",
            "stock < valid_item_count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="valid_item_count",
                lower_bound=None,
                upper_bound=5.0,
            ),
        ),
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_data_flow_range_with_relational_witness_is_feasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "stock < valid_item_count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="valid_item_count",
                lower_bound=0.0,
                upper_bound=None,
            ),
        ),
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()
    assert result.unsupported_conditions == ()


def test_data_flow_analyzer_result_can_be_used_directly() -> None:
    source = """
def sample(items):
    valid_item_count = 0

    for item in items:
        valid_item_count += 1
"""

    data_flow_result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    path = create_execution_path(
        node_labels=[
            "START",
            "stock <= -1",
            "stock >= valid_item_count",
            "return 'END'",
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

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_analyze_paths_propagates_data_flow_result() -> None:
    first_path = create_execution_path(
        node_labels=[
            "START",
            "stock <= -1",
            "stock >= valid_item_count",
            "return 'END'",
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

    second_path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "stock < valid_item_count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="valid_item_count",
                lower_bound=0.0,
                upper_bound=None,
            ),
        ),
    )

    results = PathFeasibilityAnalyzer().analyze_paths(
        (
            first_path,
            second_path,
        ),
        data_flow_result=data_flow_result,
    )

    assert len(results) == 2
    assert results[0].status == FeasibilityStatus.INFEASIBLE
    assert results[1].status == FeasibilityStatus.FEASIBLE


def test_missing_data_flow_result_can_still_use_relational_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count >= 0",
            "stock < valid_item_count",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.unsupported_conditions == ()
    assert result.relational_constraints


def test_empty_data_flow_result_allows_relational_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=DataFlowAnalysisResult(
            variables=(),
            inferred_numeric_ranges=(),
        ),
    )

    assert result.status == FeasibilityStatus.FEASIBLE


def test_data_flow_exact_range_can_participate_in_conflict_reasoning() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 5",
            "stock < valid_item_count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="valid_item_count",
                lower_bound=3.0,
                upper_bound=3.0,
            ),
        ),
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_data_flow_range_for_unrelated_variable_does_not_block_relational_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count >= 0",
            "stock < valid_item_count",
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
            "True",
            "True",
        ],
    )

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="other",
                lower_bound=100.0,
                upper_bound=100.0,
            ),
        ),
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()


# ============================================================
# PathFeasibilityAnalyzer v4.3 path-state integration tests
# ============================================================


def test_path_state_exact_value_can_prove_relational_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 0",
            "stock >= count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="count",
                exact_value=1.0,
                lower_bound=1.0,
                upper_bound=1.0,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_path_state_exact_value_can_prove_relational_feasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 5",
            "stock >= count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="count",
                exact_value=1.0,
                lower_bound=1.0,
                upper_bound=1.0,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.is_feasible is True
    assert result.conflicts == ()


def test_path_state_lower_bound_can_prove_relational_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock <= -1",
            "stock >= count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="count",
                exact_value=None,
                lower_bound=0.0,
                upper_bound=None,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_path_state_upper_bound_can_prove_relational_conflict() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 10",
            "stock < count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="count",
                exact_value=None,
                lower_bound=None,
                upper_bound=5.0,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_path_state_and_data_flow_can_be_used_together() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock == 0",
            "stock >= count",
            "return 'END'",
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

    data_flow_result = DataFlowAnalysisResult(
        variables=(),
        inferred_numeric_ranges=(
            InferredNumericRange(
                variable_name="count",
                lower_bound=0.0,
                upper_bound=None,
            ),
        ),
    )

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="count",
                exact_value=1.0,
                lower_bound=1.0,
                upper_bound=1.0,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        data_flow_result=data_flow_result,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_missing_path_state_can_still_use_relational_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "count >= 0",
            "stock < count",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.relational_constraints


def test_analyze_paths_applies_matching_path_states() -> None:
    first_path = create_execution_path(
        node_labels=[
            "START",
            "stock == 0",
            "stock >= count",
            "return 'END'",
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

    second_path = create_execution_path(
        node_labels=[
            "START",
            "stock == 5",
            "stock >= count",
            "return 'END'",
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

    path_states = (
        PathSymbolicState(
            variables=(
                SymbolicVariableState(
                    variable_name="count",
                    exact_value=1.0,
                    lower_bound=1.0,
                    upper_bound=1.0,
                ),
            )
        ),
        PathSymbolicState(
            variables=(
                SymbolicVariableState(
                    variable_name="count",
                    exact_value=1.0,
                    lower_bound=1.0,
                    upper_bound=1.0,
                ),
            )
        ),
    )

    results = PathFeasibilityAnalyzer().analyze_paths(
        (
            first_path,
            second_path,
        ),
        path_states=path_states,
    )

    assert results[0].status == FeasibilityStatus.INFEASIBLE
    assert results[1].status == FeasibilityStatus.FEASIBLE


def test_analyze_paths_rejects_path_state_length_mismatch() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "x < y",
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
            "True",
            "True",
        ],
    )

    with pytest.raises(
        ValueError,
        match="path_states ve paths aynı uzunlukta olmalıdır.",
    ):
        PathFeasibilityAnalyzer().analyze_paths(
            (
                path,
            ),
            path_states=(),
        )


def test_analyze_paths_rejects_non_tuple_path_states() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "x < y",
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
            "True",
            "True",
        ],
    )

    with pytest.raises(
        TypeError,
        match="path_states bir PathSymbolicState tuple'ı olmalıdır.",
    ):
        PathFeasibilityAnalyzer().analyze_paths(
            (
                path,
            ),
            path_states=[],  # type: ignore[arg-type]
        )


def test_analyze_paths_rejects_invalid_path_state_element() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "x < y",
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
            "True",
            "True",
        ],
    )

    with pytest.raises(
        TypeError,
        match="path_states yalnızca PathSymbolicState",
    ):
        PathFeasibilityAnalyzer().analyze_paths(
            (
                path,
            ),
            path_states=(
                "invalid",
            ),  # type: ignore[arg-type]
        )


# ============================================================
# PathFeasibilityAnalyzer v4.5 relational truth proving tests
# ============================================================


def test_greater_equal_relation_is_proven_true_from_bounds() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "valid_item_count == 0",
            "stock >= valid_item_count",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()
    assert result.relational_constraints


def test_less_equal_relation_is_proven_true_from_separated_bounds() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 10",
            "x <= y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()


def test_strict_less_relation_is_proven_true_from_separated_bounds() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 4",
            "y >= 5",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE


def test_strict_less_equal_boundary_is_proven_true_when_equality_is_excluded() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x < 5",
            "y >= 5",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE


def test_strict_less_equal_boundary_is_feasible_when_witness_exists() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 5",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()


def test_less_equal_equal_boundary_is_proven_true() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 5",
            "x <= y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE


def test_less_relation_is_proven_false_from_bounds() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 10",
            "y <= 5",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_greater_relation_is_proven_false_by_reversal() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 10",
            "x > y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE


def test_disjoint_domains_prove_relational_inequality_true() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 10",
            "x != y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE


def test_disjoint_domains_prove_relational_equality_false() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x <= 5",
            "y >= 10",
            "x == y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_overlapping_domains_are_feasible_when_witness_exists() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 1",
            "x >= y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()


def test_multiple_relations_can_be_feasible_with_single_common_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "a == 1",
            "b == 2",
            "c >= 0",
            "a < b",
            "b >= c",
            "return 'END'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "True",
            "True",
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    # a < b proven true, fakat b >= c mevcut domain ile kesin değil.
    assert result.status == FeasibilityStatus.FEASIBLE


def test_any_proven_false_relation_makes_path_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "a == 1",
            "b == 2",
            "c == 10",
            "a < b",
            "b >= c",
            "return 'END'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "True",
            "True",
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_path_state_exact_zero_can_prove_stock_greater_equal_count() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "stock >= valid_item_count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="valid_item_count",
                exact_value=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.FEASIBLE


def test_path_state_open_range_is_feasible_with_relational_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "stock >= 0",
            "stock >= valid_item_count",
            "return 'END'",
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

    path_state = PathSymbolicState(
        variables=(
            SymbolicVariableState(
                variable_name="valid_item_count",
                exact_value=None,
                lower_bound=1.0,
                upper_bound=None,
            ),
        )
    )

    result = PathFeasibilityAnalyzer().analyze_path(
        path,
        path_state=path_state,
    )

    assert result.status == FeasibilityStatus.FEASIBLE


# ============================================================
# PathFeasibilityAnalyzer v4.6 concrete witness tests
# ============================================================


def test_v4_6_literal_domains_allow_less_relation_concrete_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "x < y",
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
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()
    assert result.unsupported_conditions == ()


def test_v4_6_multiple_relations_require_one_common_witness() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x >= 0",
            "y >= 0",
            "z >= 2",
            "x < y",
            "y < z",
            "return 'END'",
        ],
        node_types=[
            "start",
            "if",
            "if",
            "if",
            "if",
            "if",
            "return",
        ],
        edge_labels=[
            None,
            "True",
            "True",
            "True",
            "True",
            "True",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert len(result.relational_constraints) == 2


def test_v4_6_exact_relational_contradiction_stays_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "x == 1",
            "y == 1",
            "x < y",
            "return 'END'",
        ],
        node_types=["start", "if", "if", "if", "return"],
        edge_labels=[None, "True", "True", "True"],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.conflicts


def test_v4_6_witness_search_cap_preserves_unknown_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    analyzer = PathFeasibilityAnalyzer()
    monkeypatch.setattr(analyzer, "_MAX_RELATIONAL_WITNESS_COMBINATIONS", 1)

    path = create_execution_path(
        node_labels=["START", "x < y", "return 'END'"],
        node_types=["start", "if", "return"],
        edge_labels=[None, "True"],
    )

    result = analyzer.analyze_path(path)

    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.conflicts == ()


# ============================================================
# PathFeasibilityAnalyzer deterministic local while tests
# ============================================================


def test_local_constant_while_exact_iteration_count_is_feasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "counter = 2",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "return 'END'",
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
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.conflicts == ()


def test_local_constant_while_too_few_iterations_is_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "counter = 2",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "return 'END'",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "return",
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True
    assert any(
        "while" in conflict.lower()
        and "iterasyon" in conflict.lower()
        for conflict in result.conflicts
    )


def test_local_constant_while_too_many_iterations_is_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "counter = 2",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "return 'END'",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "return",
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.is_infeasible is True


def test_local_constant_incrementing_while_is_checked_generically() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "index = 0",
            "index < 3",
            "index += 1",
            "index < 3",
            "index += 1",
            "index < 3",
            "index += 1",
            "index < 3",
            "return 'END'",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "AugAssign",
            "while",
            "return",
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "True",
            "Loop",
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.FEASIBLE


def test_non_literal_local_while_initial_value_is_not_marked_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "counter = initial_count",
            "counter > 0",
            "counter -= 1",
            "counter > 0",
            "return 'END'",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "return",
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    # Başlangıç değeri kesin bilinmediği için analyzer bu path'i
    # sırf local while nedeniyle INFEASIBLE olarak işaretlememelidir.
    assert result.status != FeasibilityStatus.INFEASIBLE


def test_local_while_with_unsupported_update_is_not_proven_infeasible() -> None:
    path = create_execution_path(
        node_labels=[
            "START",
            "counter = 2",
            "counter > 0",
            "counter *= 2",
            "counter > 0",
            "return 'END'",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "AugAssign",
            "while",
            "return",
        ],
        edge_labels=[
            None,
            None,
            "True",
            "Loop",
            "False",
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    # Desteklenmeyen update için güvenlik ilkesi:
    # kesin çelişki kanıtlanamadığı sürece INFEASIBLE denmemeli.
    assert result.status != FeasibilityStatus.INFEASIBLE


def test_local_constant_while_zero_iteration_is_infeasible() -> None:
    """
    Sabit pozitif bir local değişken ile başlayan while döngüsünün
    sıfır iterasyonlu CFG yolu erişilemez olmalıdır.

    Path üzerinde döngü gövdesindeki AugAssign bulunmasa bile
    başlangıç ataması ile ilk while koşulu birlikte değerlendirilmelidir.
    """
    path = create_execution_path(
        node_labels=[
            "START",
            "remaining_checks = 2",
            "remaining_checks > 0",
            "return 'done'",
            "END",
        ],
        node_types=[
            "start",
            "Assign",
            "while",
            "return",
            "end",
        ],
        edge_labels=[
            None,
            None,
            "False",
            None,
        ],
    )

    result = PathFeasibilityAnalyzer().analyze_path(path)

    assert result.status == FeasibilityStatus.INFEASIBLE