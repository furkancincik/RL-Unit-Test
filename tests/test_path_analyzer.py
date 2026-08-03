from pathlib import Path

import pytest

from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer


def test_find_paths_for_score_function() -> None:
    builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()

    graphs = builder.build_from_file(
        "datasets/sample_code.py"
    )

    graph = graphs[0]
    paths = path_analyzer.find_paths(graph)

    assert len(paths) == 3

    node_paths = [
        path.node_ids
        for path in paths
    ]

    assert [1, 3, 4, 2] in node_paths
    assert [1, 3, 5, 6, 2] in node_paths
    assert [1, 3, 5, 7, 2] in node_paths


def test_paths_contain_expected_edge_labels() -> None:
    builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()

    graph = builder.build_from_file(
        "datasets/sample_code.py"
    )[0]

    paths = path_analyzer.find_paths(graph)

    edge_label_paths = [
        path.edge_labels
        for path in paths
    ]

    assert [None, "True", None] in edge_label_paths
    assert [None, "False", "True", None] in edge_label_paths
    assert [None, "False", "False", None] in edge_label_paths


def create_while_source(
    tmp_path: Path,
) -> Path:
    """While döngüsü içeren geçici kaynak dosya oluşturur."""
    source_file = tmp_path / "while_example.py"

    source_file.write_text(
        """
def countdown(value: int) -> int:
    while value > 0:
        value -= 1

    return value
""".strip(),
        encoding="utf-8",
    )

    return source_file


def create_for_source(
    tmp_path: Path,
) -> Path:
    """For döngüsü içeren geçici kaynak dosya oluşturur."""
    source_file = tmp_path / "for_example.py"

    source_file.write_text(
        """
def calculate_total(values: list[int]) -> int:
    total = 0

    for value in values:
        total += value

    return total
""".strip(),
        encoding="utf-8",
    )

    return source_file


def test_find_paths_limits_while_loop_to_zero_and_one_iteration(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_while_source(tmp_path)
    )[0]

    paths = CFGPathAnalyzer().find_paths(graph)

    assert len(paths) == 2

    assert {
        path.loop_iteration_count
        for path in paths
    } == {
        0,
        1,
    }

    assert all(
        path.contains_loop
        for path in paths
    )

    assert sum(
        path.is_zero_iteration_loop_path
        for path in paths
    ) == 1


def test_find_paths_limits_for_loop_to_zero_and_one_iteration(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_for_source(tmp_path)
    )[0]

    paths = CFGPathAnalyzer().find_paths(graph)

    assert len(paths) == 2

    assert {
        path.loop_iteration_count
        for path in paths
    } == {
        0,
        1,
    }

    assert all(
        path.contains_loop
        for path in paths
    )


def test_find_paths_supports_controlled_multiple_iterations(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_while_source(tmp_path)
    )[0]

    paths = CFGPathAnalyzer().find_paths(
        graph,
        max_visits_per_node=3,
    )

    assert len(paths) == 3

    assert {
        path.loop_iteration_count
        for path in paths
    } == {
        0,
        1,
        2,
    }


def test_loop_steps_preserve_loop_node_metadata(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_while_source(tmp_path)
    )[0]

    paths = CFGPathAnalyzer().find_paths(graph)

    one_iteration_path = next(
        path
        for path in paths
        if path.loop_iteration_count == 1
    )

    assert len(one_iteration_path.loop_steps) == 2

    assert all(
        step.node_type == "while"
        for step in one_iteration_path.loop_steps
    )

    assert all(
        step.node_label == "value > 0"
        for step in one_iteration_path.loop_steps
    )


def test_find_paths_rejects_boolean_max_visits(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_while_source(tmp_path)
    )[0]

    with pytest.raises(
        TypeError,
        match=(
            "max_visits_per_node bir integer "
            "olmalıdır"
        ),
    ):
        CFGPathAnalyzer().find_paths(
            graph,
            max_visits_per_node=True,
        )


def test_find_paths_rejects_zero_max_visits(
    tmp_path: Path,
) -> None:
    graph = ControlFlowGraphBuilder().build_from_file(
        create_while_source(tmp_path)
    )[0]

    with pytest.raises(
        ValueError,
        match=(
            "max_visits_per_node en az 1 "
            "olmalıdır"
        ),
    ):
        CFGPathAnalyzer().find_paths(
            graph,
            max_visits_per_node=0,
        )