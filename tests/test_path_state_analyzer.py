from __future__ import annotations

import pytest

from cfg.path_analyzer import ExecutionPath
from cfg.path_state_analyzer import (
    PathStateAnalyzer,
)


def create_execution_path(
    *,
    line_numbers: list[int | None],
) -> ExecutionPath:
    node_count = len(line_numbers)

    return ExecutionPath(
        node_ids=list(
            range(
                1,
                node_count + 1,
            )
        ),
        edge_labels=[
            None
            for _ in range(
                max(
                    node_count - 1,
                    0,
                )
            )
        ],
        node_labels=[
            "NODE"
            for _ in range(
                node_count
            )
        ],
        node_types=[
            "stmt"
            for _ in range(
                node_count
            )
        ],
        line_numbers=line_numbers,
    )


def test_simple_assignment_creates_exact_state() -> None:
    source = """
def sample():
    count = 0
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 0.0
    assert count.lower_bound == 0.0
    assert count.upper_bound == 0.0


def test_augmented_assignment_updates_exact_state() -> None:
    source = """
def sample():
    count = 0
    count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            4,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 1.0
    assert count.lower_bound == 1.0
    assert count.upper_bound == 1.0


def test_subtraction_updates_exact_state() -> None:
    source = """
def sample():
    remaining = 5
    remaining -= 2
"""

    path = create_execution_path(
        line_numbers=[
            3,
            4,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    remaining = result.get_variable(
        "remaining"
    )

    assert remaining is not None
    assert remaining.exact_value == 3.0
    assert remaining.lower_bound == 3.0
    assert remaining.upper_bound == 3.0


def test_assignment_not_on_path_is_ignored() -> None:
    source = """
def sample():
    count = 0
    count = 10
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 0.0


def test_augmented_assignment_not_on_path_is_ignored() -> None:
    source = """
def sample():
    count = 0
    count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 0.0


def test_negative_literal_assignment_is_supported() -> None:
    source = """
def sample():
    value = -5
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    value = result.get_variable(
        "value"
    )

    assert value is not None
    assert value.exact_value == -5.0


def test_positive_unary_literal_assignment_is_supported() -> None:
    source = """
def sample():
    value = +5
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    value = result.get_variable(
        "value"
    )

    assert value is not None
    assert value.exact_value == 5.0


def test_annotated_assignment_is_supported() -> None:
    source = """
def sample():
    count: int = 3
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 3.0


def test_unknown_assignment_removes_known_state() -> None:
    source = """
def sample(value):
    count = 0
    count = value
"""

    path = create_execution_path(
        line_numbers=[
            3,
            4,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    assert (
        result.get_variable(
            "count"
        )
        is None
    )


def test_unknown_augmented_assignment_removes_known_state() -> None:
    source = """
def sample(step):
    count = 0
    count += step
"""

    path = create_execution_path(
        line_numbers=[
            3,
            4,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    assert (
        result.get_variable(
            "count"
        )
        is None
    )


def test_unsupported_augmented_operator_removes_state() -> None:
    source = """
def sample():
    count = 2
    count *= 3
"""

    path = create_execution_path(
        line_numbers=[
            3,
            4,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    assert (
        result.get_variable(
            "count"
        )
        is None
    )


def test_if_body_assignment_is_applied_only_when_line_is_on_path() -> None:
    source = """
def sample():
    count = 0

    if True:
        count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 1.0


def test_if_body_assignment_is_skipped_when_not_on_path() -> None:
    source = """
def sample():
    count = 0

    if True:
        count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value == 0.0


def test_loop_body_update_is_applied_conservatively_when_line_is_on_path() -> None:
    source = """
def sample(items):
    count = 0

    for item in items:
        count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None
    assert count.exact_value is None
    assert count.lower_bound == 1.0
    assert count.upper_bound is None


def test_loop_body_update_does_not_claim_exact_iteration_count() -> None:
    source = """
def sample(items):
    count = 0

    for item in items:
        count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    count = result.get_variable(
        "count"
    )

    assert count is not None

    # v4.4: loop kaç kez döndü bilinmediği için exact değer verilmez.
    # Update satırı path üzerinde bulunduğundan en az bir artışın
    # gerçekleştiği güvenli biçimde söylenebilir.
    assert count.exact_value is None
    assert count.lower_bound == 1.0
    assert count.upper_bound is None


def test_try_body_assignment_is_supported_when_on_path() -> None:
    source = """
def sample():
    result = 0

    try:
        result = 5
    except ValueError:
        result = -1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    variable = result.get_variable(
        "result"
    )

    assert variable is not None
    assert variable.exact_value == 5.0


def test_unknown_variable_returns_none() -> None:
    source = """
def sample():
    value = 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    result = (
        PathStateAnalyzer()
        .analyze_source(
            source=source,
            function_name="sample",
            path=path,
        )
    )

    assert (
        result.get_variable(
            "missing"
        )
        is None
    )


def test_function_not_found_raises_error() -> None:
    source = """
def existing():
    return 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Fonksiyon bulunamadı",
    ):
        PathStateAnalyzer().analyze_source(
            source=source,
            function_name="missing",
            path=path,
        )


def test_empty_function_name_is_rejected() -> None:
    source = """
def sample():
    pass
"""

    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    with pytest.raises(
        ValueError,
        match="function_name boş olamaz.",
    ):
        PathStateAnalyzer().analyze_source(
            source=source,
            function_name="",
            path=path,
        )


def test_non_string_source_is_rejected() -> None:
    path = create_execution_path(
        line_numbers=[
            3,
        ]
    )

    with pytest.raises(
        TypeError,
        match="source string olmalıdır.",
    ):
        PathStateAnalyzer().analyze_source(
            source=123,  # type: ignore[arg-type]
            function_name="sample",
            path=path,
        )


def test_invalid_path_is_rejected() -> None:
    source = """
def sample():
    pass
"""

    with pytest.raises(
        TypeError,
        match="path bir ExecutionPath örneği olmalıdır.",
    ):
        PathStateAnalyzer().analyze_source(
            source=source,
            function_name="sample",
            path="invalid",  # type: ignore[arg-type]
        )


# ============================================================
# PathStateAnalyzer v4.4 loop-aware conservative tests
# ============================================================


def test_loop_decrement_creates_safe_upper_bound() -> None:
    source = """
def sample(items):
    count = 10

    for item in items:
        count -= 2
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = PathStateAnalyzer().analyze_source(
        source=source,
        function_name="sample",
        path=path,
    )

    count = result.get_variable("count")

    assert count is not None
    assert count.exact_value is None
    assert count.lower_bound is None
    assert count.upper_bound == 8.0


def test_while_loop_increment_is_widened_conservatively() -> None:
    source = """
def sample():
    count = 2

    while True:
        count += 3
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = PathStateAnalyzer().analyze_source(
        source=source,
        function_name="sample",
        path=path,
    )

    count = result.get_variable("count")

    assert count is not None
    assert count.exact_value is None
    assert count.lower_bound == 5.0
    assert count.upper_bound is None


def test_zero_delta_inside_loop_preserves_exact_state() -> None:
    source = """
def sample(items):
    count = 5

    for item in items:
        count += 0
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = PathStateAnalyzer().analyze_source(
        source=source,
        function_name="sample",
        path=path,
    )

    count = result.get_variable("count")

    assert count is not None
    assert count.exact_value == 5.0
    assert count.lower_bound == 5.0
    assert count.upper_bound == 5.0


def test_loop_update_nested_inside_if_is_still_conservative() -> None:
    source = """
def sample(items):
    count = 0

    for item in items:
        if item > 0:
            count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            7,
        ]
    )

    result = PathStateAnalyzer().analyze_source(
        source=source,
        function_name="sample",
        path=path,
    )

    count = result.get_variable("count")

    assert count is not None
    assert count.exact_value is None
    assert count.lower_bound == 1.0
    assert count.upper_bound is None


def test_non_loop_if_update_remains_exact() -> None:
    source = """
def sample():
    count = 0

    if True:
        count += 1
"""

    path = create_execution_path(
        line_numbers=[
            3,
            6,
        ]
    )

    result = PathStateAnalyzer().analyze_source(
        source=source,
        function_name="sample",
        path=path,
    )

    count = result.get_variable("count")

    assert count is not None
    assert count.exact_value == 1.0
    assert count.lower_bound == 1.0
    assert count.upper_bound == 1.0


def test_ordered_step_state_applies_repeated_updates_per_visit() -> None:
    source = """
def sample(items):
    count = 0
    for item in items:
        count += 1
    if count == 2:
        return True
    return False
"""
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 3, 4, 3, 5, 6, 7],
        edge_labels=[None, None, "Iterate", "Next", "Iterate", "Next",
                     "Complete", "True", None],
        node_labels=[
            "START", "count = 0", "item in items", "count += 1",
            "item in items", "count += 1", "item in items", "count == 2",
            "return True", "END",
        ],
        node_types=[
            "start", "Assign", "for", "AugAssign", "for", "AugAssign",
            "for", "if", "return", "end",
        ],
        line_numbers=[None, 3, 4, 5, 4, 5, 4, 6, 7, None],
    )

    result = PathStateAnalyzer().analyze_source(
        source=source, function_name="sample", path=path
    )
    condition_state = result.state_before_step(7)

    assert condition_state is not None
    count = condition_state.get_variable("count")
    assert count is not None
    assert count.exact_value == 2.0


def test_ordered_step_state_marks_unsupported_tracked_update() -> None:
    source = """
def sample(items, delta):
    count = 0
    for item in items:
        count += delta
    if count == 0:
        return True
    return False
"""
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 3, 5, 6, 7],
        edge_labels=[None, None, "Iterate", "Next", "Complete", "True", None],
        node_labels=[
            "START", "count = 0", "item in items", "count += delta",
            "item in items", "count == 0", "return True", "END",
        ],
        node_types=[
            "start", "Assign", "for", "AugAssign", "for", "if", "return", "end",
        ],
        line_numbers=[None, 3, 4, 5, 4, 6, 7, None],
    )

    result = PathStateAnalyzer().analyze_source(
        source=source, function_name="sample", path=path
    )
    condition_state = result.state_before_step(5)

    assert condition_state is not None
    assert condition_state.get_variable("count") is None
    assert condition_state.unsupported_variables == ("count",)
