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