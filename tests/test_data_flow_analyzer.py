from pathlib import Path

import pytest

from cfg.data_flow_analyzer import (
    DataFlowAnalyzer,
    DataFlowOperationType,
    InferredNumericRange,
)


def test_simple_assignment_is_detected() -> None:
    source = """
def sample():
    count = 0
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "count"
    )

    assert variable is not None
    assert len(variable.updates) == 1

    update = variable.updates[0]

    assert (
        update.operation_type
        == DataFlowOperationType.ASSIGNMENT
    )

    assert update.expression == "0"
    assert update.dependencies == ()
    assert update.inside_loop is False


def test_augmented_assignment_is_detected() -> None:
    source = """
def sample():
    count = 0
    count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "count"
    )

    assert variable is not None
    assert len(variable.updates) == 2
    assert variable.is_updated is True

    update = variable.updates[1]

    assert (
        update.operation_type
        == DataFlowOperationType.AUGMENTED_ASSIGNMENT
    )

    assert update.expression == "count += 1"


def test_loop_update_is_detected() -> None:
    source = """
def sample(items):
    count = 0

    for item in items:
        count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "count"
    )

    assert variable is not None
    assert variable.is_updated is True
    assert variable.is_updated_inside_loop is True

    assert variable.updates[0].inside_loop is False
    assert variable.updates[1].inside_loop is True


def test_derived_assignment_dependencies_are_detected() -> None:
    source = """
def sample(total, count):
    average = total / count
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "average"
    )

    assert variable is not None

    update = variable.updates[0]

    assert (
        update.operation_type
        == DataFlowOperationType.DERIVED_ASSIGNMENT
    )

    assert update.expression == "total / count"

    assert set(
        update.dependencies
    ) == {
        "total",
        "count",
    }


def test_multiple_dependencies_are_detected() -> None:
    source = """
def sample(price, quantity, tax):
    total = price * quantity + tax
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "total"
    )

    assert variable is not None

    assert set(
        variable.dependencies
    ) == {
        "price",
        "quantity",
        "tax",
    }


def test_assignment_from_variable_is_detected() -> None:
    source = """
def sample(value):
    result = value
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "result"
    )

    assert variable is not None

    assert variable.dependencies == (
        "value",
    )


def test_if_body_assignment_is_detected() -> None:
    source = """
def sample(value):
    result = 0

    if value > 10:
        result = value
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "result"
    )

    assert variable is not None
    assert len(variable.updates) == 2

    assert variable.updates[1].expression == "value"


def test_try_body_assignment_is_detected() -> None:
    source = """
def sample(values):
    result = 0

    try:
        result = values[0]
    except IndexError:
        result = -1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "result"
    )

    assert variable is not None
    assert len(variable.updates) == 3


def test_while_loop_update_is_detected() -> None:
    source = """
def sample():
    counter = 2

    while counter > 0:
        counter -= 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "counter"
    )

    assert variable is not None
    assert variable.is_updated_inside_loop is True


def test_multiple_variables_are_collected() -> None:
    source = """
def sample():
    first = 1
    second = 2
    third = first + second
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert result.get_variable(
        "first"
    ) is not None

    assert result.get_variable(
        "second"
    ) is not None

    assert result.get_variable(
        "third"
    ) is not None


def test_unknown_variable_returns_none() -> None:
    source = """
def sample():
    value = 10
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_variable(
            "missing"
        )
        is None
    )


def test_annotated_assignment_is_detected() -> None:
    source = """
def sample():
    count: int = 0
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "count"
    )

    assert variable is not None
    assert variable.updates[0].expression == "0"


def test_function_not_found_raises_error() -> None:
    source = """
def existing():
    return 1
"""

    with pytest.raises(
        ValueError,
        match="Fonksiyon bulunamadı",
    ):
        DataFlowAnalyzer().analyze_source(
            source=source,
            function_name="missing",
        )


def test_empty_function_name_is_rejected() -> None:
    source = """
def sample():
    pass
"""

    with pytest.raises(
        ValueError,
        match="function_name boş olamaz.",
    ):
        DataFlowAnalyzer().analyze_source(
            source=source,
            function_name="",
        )


def test_non_string_source_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="source string olmalıdır.",
    ):
        DataFlowAnalyzer().analyze_source(
            source=123,  # type: ignore[arg-type]
            function_name="sample",
        )


def test_valid_item_count_pattern_is_detected() -> None:
    """
    Ultracomplex'te karşılaştığımız yapının genel
    data-flow karşılığı.
    """

    source = """
def sample(items):
    valid_item_count = 0

    for item in items:
        if item < 0:
            continue

        valid_item_count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    variable = result.get_variable(
        "valid_item_count"
    )

    assert variable is not None
    assert len(variable.updates) == 2

    initial_update = variable.updates[0]
    loop_update = variable.updates[1]

    assert initial_update.expression == "0"
    assert initial_update.inside_loop is False

    assert loop_update.expression == (
        "valid_item_count += 1"
    )

    assert loop_update.inside_loop is True

    assert variable.is_updated is True
    assert variable.is_updated_inside_loop is True


# ============================================================
# DataFlowAnalyzer v4.2 numeric range inference tests
# ============================================================


def test_simple_numeric_assignment_creates_exact_range() -> None:
    source = """
def sample():
    value = 10
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "value"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="value",
        lower_bound=10.0,
        upper_bound=10.0,
    )


def test_positive_increment_preserves_initial_lower_bound() -> None:
    source = """
def sample():
    count = 0
    count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "count"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="count",
        lower_bound=0.0,
        upper_bound=None,
    )


def test_positive_loop_increment_infers_non_negative_lower_bound() -> None:
    source = """
def sample(items):
    count = 0

    for item in items:
        count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "count"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="count",
        lower_bound=0.0,
        upper_bound=None,
    )


def test_positive_decrement_preserves_initial_upper_bound() -> None:
    source = """
def sample():
    remaining = 10
    remaining -= 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "remaining"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="remaining",
        lower_bound=None,
        upper_bound=10.0,
    )


def test_negative_initial_value_is_supported() -> None:
    source = """
def sample():
    counter = -5
    counter += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "counter"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="counter",
        lower_bound=-5.0,
        upper_bound=None,
    )


def test_derived_initial_assignment_does_not_create_numeric_range() -> None:
    source = """
def sample(start):
    count = start
    count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_later_normal_assignment_invalidates_numeric_range() -> None:
    source = """
def sample(value):
    count = 0
    count += 1
    count = value
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_later_derived_assignment_invalidates_numeric_range() -> None:
    source = """
def sample(value):
    count = 0
    count += 1
    count = value + 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_negative_increment_delta_is_not_inferred() -> None:
    source = """
def sample():
    count = 0
    count += -1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_negative_decrement_delta_is_not_inferred() -> None:
    source = """
def sample():
    count = 0
    count -= -1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_non_numeric_augmented_assignment_is_not_inferred() -> None:
    source = """
def sample(step):
    count = 0
    count += step
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_multiplication_augmented_assignment_is_not_inferred() -> None:
    source = """
def sample():
    count = 2
    count *= 2
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "count"
        )
        is None
    )


def test_unknown_numeric_range_returns_none() -> None:
    source = """
def sample():
    value = 10
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    assert (
        result.get_numeric_range(
            "missing"
        )
        is None
    )


def test_valid_item_count_pattern_infers_non_negative_range() -> None:
    source = """
def sample(items):
    valid_item_count = 0

    for item in items:
        if item < 0:
            continue

        valid_item_count += 1
"""

    result = DataFlowAnalyzer().analyze_source(
        source=source,
        function_name="sample",
    )

    numeric_range = result.get_numeric_range(
        "valid_item_count"
    )

    assert numeric_range == InferredNumericRange(
        variable_name="valid_item_count",
        lower_bound=0.0,
        upper_bound=None,
    )
