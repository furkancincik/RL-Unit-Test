import json
from pathlib import Path

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from reports.json_reporter import JSONReportWriter


def test_write_dqm_report_creates_json_file(
    tmp_path: Path,
) -> None:
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    reporter = JSONReportWriter()

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

    output_file = tmp_path / "dqm_report.json"

    result_path = reporter.write_dqm_report(
        function=function,
        paths=paths,
        scores=scores,
        output_path=output_file,
        source_file="datasets/sample_code.py",
    )

    assert result_path.exists()
    assert result_path.suffix == ".json"

    report_data = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    assert report_data["metadata"]["report_type"] == "dqm"
    assert report_data["metadata"]["schema_version"] == "1.0"
    assert (
        report_data["metadata"]["source_file"]
        == "datasets/sample_code.py"
    )
    assert report_data["metadata"]["generated_at"]

    assert (
        report_data["summary"]["function_name"]
        == "calculate_score"
    )
    assert report_data["summary"]["execution_path_count"] == 3
    assert report_data["summary"]["scored_path_count"] == 3
    assert (
        report_data["summary"]["highest_normalized_score"]
        == 100.0
    )
    assert (
        report_data["summary"]["high_priority_path_count"]
        == 2
    )
    assert (
        report_data["summary"]["medium_priority_path_count"]
        == 1
    )
    assert (
        report_data["summary"]["low_priority_path_count"]
        == 0
    )

    assert report_data["function"]["name"] == "calculate_score"
    assert len(report_data["execution_paths"]) == 3
    assert len(report_data["dqm_scores"]) == 3
    assert (
        report_data["dqm_scores"][0]["normalized_score"]
        == 100.0
    )