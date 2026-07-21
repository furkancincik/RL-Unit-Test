from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix


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