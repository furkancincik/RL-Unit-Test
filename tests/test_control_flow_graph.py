from pathlib import Path

import pytest

from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer


SOURCE_FILE = "datasets/sample_code.py"


def test_build_from_file_creates_expected_graph() -> None:
    """Örnek kaynak dosya için beklenen CFG yapısının üretildiğini doğrular."""
    builder = ControlFlowGraphBuilder()

    graphs = builder.build_from_file(
        Path(SOURCE_FILE)
    )

    assert len(graphs) == 1

    graph = graphs[0]

    assert graph.function_name == "calculate_score"
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 8


def test_graph_contains_expected_nodes() -> None:
    """CFG düğüm türlerinin ve etiketlerinin doğru olduğunu doğrular."""
    builder = ControlFlowGraphBuilder()

    graphs = builder.build_from_file(SOURCE_FILE)
    graph = graphs[0]

    node_types = [
        node.node_type
        for node in graph.nodes
    ]
    node_labels = [
        node.label
        for node in graph.nodes
    ]

    assert node_types.count("start") == 1
    assert node_types.count("end") == 1
    assert node_types.count("if") == 2
    assert node_types.count("return") == 3

    assert "START" in node_labels
    assert "END" in node_labels
    assert "score >= 85" in node_labels
    assert "score >= 50" in node_labels
    assert "return 'Başarılı'" in node_labels
    assert "return 'Orta'" in node_labels
    assert "return 'Başarısız'" in node_labels


def test_graph_contains_expected_edges() -> None:
    """Örnek fonksiyonun CFG bağlantılarının doğru olduğunu doğrular."""
    builder = ControlFlowGraphBuilder()

    graph = builder.build_from_file(SOURCE_FILE)[0]

    edges = {
        (edge.source_id, edge.target_id, edge.label)
        for edge in graph.edges
    }

    assert edges == {
        (1, 3, None),
        (3, 4, "True"),
        (4, 2, None),
        (3, 5, "False"),
        (5, 6, "True"),
        (6, 2, None),
        (5, 7, "False"),
        (7, 2, None),
    }


def test_build_from_file_raises_error_when_file_is_missing() -> None:
    builder = ControlFlowGraphBuilder()

    with pytest.raises(FileNotFoundError):
        builder.build_from_file(
            "datasets/olmayan_dosya.py"
        )


def test_build_from_file_rejects_non_python_files(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "örnek içerik",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        builder.build_from_file(text_file)


def test_build_from_file_rejects_invalid_python_syntax(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text(
        "def broken_function(\n",
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError):
        builder.build_from_file(invalid_file)


def test_graph_supports_while_loop(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

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

    graph = builder.build_from_file(source_file)[0]

    assert graph.function_name == "countdown"
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 5


def test_graph_supports_for_loop(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

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

    graph = builder.build_from_file(source_file)[0]

    assert graph.function_name == "calculate_total"
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 6


def test_graph_supports_try_except(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

    source_file = tmp_path / "try_example.py"
    source_file.write_text(
        """
def safe_divide(a: int, b: int) -> float | None:
    try:
        result = a / b
    except ZeroDivisionError:
        return None

    return result
""".strip(),
        encoding="utf-8",
    )

    graph = builder.build_from_file(source_file)[0]

    assert graph.function_name == "safe_divide"
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 7

    node_by_label = {
        node.label: node
        for node in graph.nodes
    }

    edges = {
        (edge.source_id, edge.target_id, edge.label)
        for edge in graph.edges
    }

    try_node = node_by_label["try"]
    assignment_node = node_by_label["result = a / b"]
    except_node = node_by_label[
        "except ZeroDivisionError"
    ]

    assert (
        try_node.node_id,
        assignment_node.node_id,
        "Success",
    ) in edges

    assert (
        assignment_node.node_id,
        except_node.node_id,
        "Exception",
    ) in edges

    assert (
        try_node.node_id,
        except_node.node_id,
        "Exception",
    ) not in edges


def test_exception_path_preserves_raising_expression(
    tmp_path: Path,
) -> None:
    builder = ControlFlowGraphBuilder()

    source_file = tmp_path / "index_example.py"
    source_file.write_text(
        """
def first_item(values: list[int]) -> int | None:
    try:
        result = values[0]
    except IndexError:
        return None

    return result
""".strip(),
        encoding="utf-8",
    )

    graph = builder.build_from_file(source_file)[0]

    assignment_node = next(
        node
        for node in graph.nodes
        if node.label == "result = values[0]"
    )
    except_node = next(
        node
        for node in graph.nodes
        if node.label == "except IndexError"
    )

    assert any(
        edge.source_id == assignment_node.node_id
        and edge.target_id == except_node.node_id
        and edge.label == "Exception"
        for edge in graph.edges
    )


def test_continue_returns_directly_to_for_node(
    tmp_path: Path,
) -> None:
    """Continue düğümünün doğrudan ilgili for düğümüne döndüğünü doğrular."""
    source_file = tmp_path / "continue_example.py"
    source_file.write_text(
        """
def count_valid(items: list[int]) -> int:
    count = 0

    for item in items:
        if item < 0:
            continue

        count += 1

    return count
""".strip(),
        encoding="utf-8",
    )

    graph = ControlFlowGraphBuilder().build_from_file(
        source_file
    )[0]

    for_node = next(
        node
        for node in graph.nodes
        if node.node_type == "for"
    )
    continue_node = next(
        node
        for node in graph.nodes
        if node.node_type == "continue"
    )
    increment_node = next(
        node
        for node in graph.nodes
        if node.node_type == "AugAssign"
    )

    assert any(
        edge.source_id == continue_node.node_id
        and edge.target_id == for_node.node_id
        and edge.label == "Continue"
        for edge in graph.edges
    )

    assert not any(
        edge.source_id == continue_node.node_id
        and edge.target_id == increment_node.node_id
        for edge in graph.edges
    )


def test_paths_do_not_continue_after_continue_statement(
    tmp_path: Path,
) -> None:
    """
    Continue sonrasında aynı iterasyondaki return ifadesine
    doğrudan geçilmediğini doğrular.
    """
    source_file = tmp_path / "continue_path_example.py"
    source_file.write_text(
        """
def classify(items: list[int]) -> str:
    for item in items:
        if item < 0:
            continue

        return "non-negative"

    return "finished"
""".strip(),
        encoding="utf-8",
    )

    graph = ControlFlowGraphBuilder().build_from_file(
        source_file
    )[0]
    paths = CFGPathAnalyzer().find_paths(graph)

    for path in paths:
        steps = path.steps

        for index, step in enumerate(steps):
            if (
                step.node_type != "continue"
                or index + 1 >= len(steps)
            ):
                continue

            next_step = steps[index + 1]

            assert next_step.node_type == "for"
            assert (
                step.outgoing_edge_label
                == "Continue"
            )

def test_nested_if_continue_returns_to_loop(
    tmp_path: Path,
) -> None:
    """İç içe if içindeki continue ifadesinin doğru döngüye döndüğünü doğrular."""
    source_file = tmp_path / "nested_continue.py"
    source_file.write_text(
        """
def process(items: list[int]) -> int:
    total = 0

    for item in items:
        if item < 0:
            if item < -10:
                continue

        total += item

    return total
""".strip(),
        encoding="utf-8",
    )

    graph = ControlFlowGraphBuilder().build_from_file(
        source_file
    )[0]

    continue_node = next(
        node
        for node in graph.nodes
        if node.node_type == "continue"
    )
    for_node = next(
        node
        for node in graph.nodes
        if node.node_type == "for"
    )

    assert any(
        edge.source_id == continue_node.node_id
        and edge.target_id == for_node.node_id
        and edge.label == "Continue"
        for edge in graph.edges
    )


def _build_inline_graph(
    tmp_path: Path,
    source: str,
    function_name: str = "process",
):
    source_file = tmp_path / f"{function_name}.py"
    source_file.write_text(source.strip(), encoding="utf-8")

    return next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(source_file)
        if graph.function_name == function_name
    )


@pytest.mark.parametrize(
    ("loop_source", "loop_type", "back_edge_label"),
    [
        (
            "for value in values:",
            "for",
            "Next",
        ),
        (
            "while values:",
            "while",
            "Loop",
        ),
    ],
)
def test_break_skips_remaining_loop_body_and_reaches_post_loop_statement(
    tmp_path: Path,
    loop_source: str,
    loop_type: str,
    back_edge_label: str,
) -> None:
    graph = _build_inline_graph(
        tmp_path,
        f"""
def process(values):
    {loop_source}
        break
        values.append(1)

    return "finished"
""",
    )

    break_node = next(node for node in graph.nodes if node.node_type == "break")
    loop_node = next(node for node in graph.nodes if node.node_type == loop_type)
    return_node = next(node for node in graph.nodes if node.node_type == "return")

    assert break_node.label == "break"
    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and (
            edge.target_id == loop_node.node_id
            or edge.label == back_edge_label
        )
        for edge in graph.edges
    )
    assert all(node.label != "values.append(1)" for node in graph.nodes)


def test_break_inside_if_uses_enclosing_loop_exit(tmp_path: Path) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(values):
    for value in values:
        if value == 0:
            break

        value += 1

    return "finished"
""",
    )
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    update_node = next(node for node in graph.nodes if node.node_type == "AugAssign")
    return_node = next(node for node in graph.nodes if node.node_type == "return")

    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.target_id == update_node.node_id
        for edge in graph.edges
    )


def test_inner_break_exits_only_inner_loop_and_outer_loop_can_continue(
    tmp_path: Path,
) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(rows):
    for row in rows:
        for value in row:
            if value == 0:
                break

        row.append(1)

    return rows
""",
    )
    loops = {node.label: node for node in graph.nodes if node.node_type == "for"}
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    after_inner = next(node for node in graph.nodes if node.label == "row.append(1)")

    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == after_inner.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert any(
        edge.source_id == after_inner.node_id
        and edge.target_id == loops["row in rows"].node_id
        and edge.label == "Next"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.target_id == loops["row in rows"].node_id
        for edge in graph.edges
    )


def test_outer_break_exits_outer_loop(tmp_path: Path) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(rows):
    for row in rows:
        for value in row:
            value += 1

        if not row:
            break

        row.append(1)

    return rows
""",
    )
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    return_node = next(node for node in graph.nodes if node.node_type == "return")
    after_break = next(node for node in graph.nodes if node.label == "row.append(1)")

    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.target_id == after_break.node_id
        for edge in graph.edges
    )


def test_sequential_loops_keep_break_exits_separate(tmp_path: Path) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(first, second):
    for value in first:
        break

    first_done = True

    for value in second:
        break

    return first_done
""",
    )
    break_nodes = sorted(
        (node for node in graph.nodes if node.node_type == "break"),
        key=lambda node: node.line_number or 0,
    )
    first_done = next(node for node in graph.nodes if node.label == "first_done = True")
    return_node = next(node for node in graph.nodes if node.node_type == "return")

    assert any(
        edge.source_id == break_nodes[0].node_id
        and edge.target_id == first_done.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert any(
        edge.source_id == break_nodes[1].node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )


def test_continue_and_break_keep_distinct_loop_edges(tmp_path: Path) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(values):
    for value in values:
        if value < 0:
            continue
        if value == 0:
            break

    return "finished"
""",
    )
    loop_node = next(node for node in graph.nodes if node.node_type == "for")
    continue_node = next(node for node in graph.nodes if node.node_type == "continue")
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    return_node = next(node for node in graph.nodes if node.node_type == "return")

    assert any(
        edge.source_id == continue_node.node_id
        and edge.target_id == loop_node.node_id
        and edge.label == "Continue"
        for edge in graph.edges
    )
    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )


@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_break_skips_loop_else_but_normal_completion_enters_it(
    tmp_path: Path,
    loop_kind: str,
) -> None:
    loop_header = "for value in values:" if loop_kind == "for" else "while values:"
    graph = _build_inline_graph(
        tmp_path,
        f"""
def process(values):
    {loop_header}
        break
    else:
        values.append("else")

    return "finished"
""",
    )
    loop_node = next(node for node in graph.nodes if node.node_type == loop_kind)
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    else_node = next(node for node in graph.nodes if node.label == "values.append('else')")
    return_node = next(node for node in graph.nodes if node.node_type == "return")
    completion_label = "Complete" if loop_kind == "for" else "False"

    assert any(
        edge.source_id == loop_node.node_id
        and edge.target_id == else_node.node_id
        and edge.label == completion_label
        for edge in graph.edges
    )
    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.target_id == else_node.node_id
        for edge in graph.edges
    )


def test_break_inside_try_uses_enclosing_loop_exit(tmp_path: Path) -> None:
    graph = _build_inline_graph(
        tmp_path,
        """
def process(values):
    for value in values:
        try:
            if value == 0:
                break
        except ValueError:
            value += 1

        value += 2

    return "finished"
""",
    )
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    return_node = next(node for node in graph.nodes if node.node_type == "return")

    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == return_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.label == "Exception"
        for edge in graph.edges
    )


def test_robustness_break_has_structural_loop_exit() -> None:
    graph = next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(
            Path("datasets/sample_robustness_code.py")
        )
        if graph.function_name == "analyze_transactions"
    )
    break_node = next(node for node in graph.nodes if node.node_type == "break")
    post_loop_node = next(
        node for node in graph.nodes if node.label == "amount > category_limit"
    )
    decrement_nodes = {
        node.node_id
        for node in graph.nodes
        if node.label in {"amount -= 10", "remaining_attempts -= 1"}
    }

    assert any(
        edge.source_id == break_node.node_id
        and edge.target_id == post_loop_node.node_id
        and edge.label == "Break"
        for edge in graph.edges
    )
    assert not any(
        edge.source_id == break_node.node_id
        and edge.target_id in decrement_nodes
        for edge in graph.edges
    )


def test_graph_discovery_preserves_lexical_source_order(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "ordered_functions.py"
    source_file.write_text(
        "def first():\n    return 1\n\n"
        "def outer():\n"
        "    def nested():\n        return 2\n"
        "    return nested()\n\n"
        "class Handler:\n"
        "    def method(self):\n        return 3\n\n"
        "def last():\n    return 4\n",
        encoding="utf-8",
    )

    graphs = ControlFlowGraphBuilder().build_from_file(source_file)

    assert [graph.function_name for graph in graphs] == [
        "first",
        "outer",
        "nested",
        "Handler.method",
        "last",
    ]


def test_empty_collection_method_normalization_preserves_safe_zero_iteration(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "empty_collection_method.py"
    source_file.write_text(
        "class Catalog:\n"
        "    def __init__(self):\n"
        "        self.mapping = {}\n"
        "        self.entries = []\n\n"
        "    def inspect(self, key: str) -> int:\n"
        "        count = 0\n"
        "        for name, value in self.mapping.items():\n"
        "            count += 1\n"
        "        for entry in self.entries:\n"
        "            count += 1\n"
        "        if key in self.mapping:\n"
        "            count += 1\n"
        "        return count\n",
        encoding="utf-8",
    )

    graph = next(
        item
        for item in ControlFlowGraphBuilder().build_from_file(source_file)
        if item.function_name == "Catalog.inspect"
    )
    labels = [node.label for node in graph.nodes]

    assert "__self_mapping = {}" in labels
    assert "__self_entries = []" in labels
    assert "(name, value) in ()" in labels
    assert "entry in __self_entries" in labels
    assert "key in __self_mapping" in labels
    assert all("self." not in label for label in labels)
