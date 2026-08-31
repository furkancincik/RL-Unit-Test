from types import SimpleNamespace

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer, ExecutionPath
from evaluator.dqm import DQMScore, DecisionQualityMatrix


def test_dqm_scores_all_execution_paths() -> None:
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()

    analysis_result = analyzer.analyze_file(
        "datasets/sample_code.py"
    )

    function = analysis_result.functions[0]

    graph = cfg_builder.build_from_file(
        "datasets/sample_code.py"
    )[0]

    paths = path_analyzer.find_paths(graph)

    scores = dqm.evaluate_paths(
        function=function,
        paths=paths,
    )

    assert len(scores) == 3

    assert all(
        0 <= score.normalized_score <= 100
        for score in scores
    )

    assert scores[0].normalized_score == 100.0


def test_dqm_sorts_paths_by_priority() -> None:
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()

    function = analyzer.analyze_file(
        "datasets/sample_code.py"
    ).functions[0]

    graph = cfg_builder.build_from_file(
        "datasets/sample_code.py"
    )[0]

    paths = path_analyzer.find_paths(graph)

    scores = dqm.evaluate_paths(
        function=function,
        paths=paths,
    )

    normalized_scores = [
        score.normalized_score
        for score in scores
    ]

    assert normalized_scores == sorted(
        normalized_scores,
        reverse=True,
    )


def test_dqm_returns_empty_list_for_empty_paths() -> None:
    analyzer = PythonAnalyzer()
    dqm = DecisionQualityMatrix()

    function = analyzer.analyze_file(
        "datasets/sample_code.py"
    ).functions[0]

    scores = dqm.evaluate_paths(
        function=function,
        paths=[],
    )

    assert scores == []


def test_dqm_uses_explicit_score_then_path_index_total_order() -> None:
    function = SimpleNamespace(
        cyclomatic_complexity=1,
        risk_level="Low",
    )
    paths = [
        ExecutionPath(node_ids=[1, 2], edge_labels=[None]),
        ExecutionPath(
            node_ids=[1, 2, 3],
            edge_labels=["True", None],
        ),
        ExecutionPath(
            node_ids=[4, 5, 6],
            edge_labels=["False", None],
        ),
    ]

    scores = DecisionQualityMatrix().evaluate_paths(function, paths)

    assert [score.path_index for score in scores] == [2, 3, 1]
    assert scores[0].normalized_score == scores[1].normalized_score


def test_dqm_tie_order_is_independent_of_input_sort_stability() -> None:
    tied = [
        DQMScore(
            path_index=path_index,
            path_length=2,
            decision_edge_count=1,
            contains_loop=False,
            contains_exception=False,
            raw_score=10.0,
            normalized_score=100.0,
            priority_level="High",
        )
        for path_index in (3, 1, 2)
    ]

    ordered = DecisionQualityMatrix._sort_scores(tied)

    assert [score.path_index for score in ordered] == [1, 2, 3]

