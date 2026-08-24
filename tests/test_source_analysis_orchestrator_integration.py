from pathlib import Path

import pytest

from models.project_analysis_result import (
    FunctionRunStatus,
    ProjectRunStatus,
)
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator


def test_real_multi_function_pipeline_isolates_controlled_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "multi_target.py"
    source_file.write_text(
        "def first(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def unsupported(value: int) -> int:\n"
        "    return abs(value)\n\n"
        "def last(value: int) -> int:\n"
        "    return value - 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="multi_target",
        function_name=None,
        all_functions=True,
        output_root=tmp_path / "output",
        max_visits_per_node=3,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        timeout_seconds=30.0,
        per_function_timeout_seconds=20.0,
    )

    statuses = {
        item.target.name: item.status for item in result.function_results
    }
    assert statuses == {
        "first": FunctionRunStatus.COMPLETED,
        "unsupported": FunctionRunStatus.PARTIAL,
        "last": FunctionRunStatus.COMPLETED,
    }
    assert result.status is ProjectRunStatus.PARTIAL
    assert result.completed_count == 2
    assert result.partial_count == 1
    assert result.report_path.is_file()
    assert len({item.output_directory for item in result.function_results}) == 3


def test_safe_instance_method_runs_full_coverage_greedy_and_rl_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "counter_fixture.py"
    source_file.write_text(
        "class Counter:\n"
        "    def __init__(self, value: int = 0):\n"
        "        self.value = value\n\n"
        "    def classify(self, delta: int) -> str:\n"
        "        if self.value + delta >= 10:\n"
        "            return 'high'\n"
        "        return 'low'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="counter_fixture",
        function_name="Counter.classify",
        all_functions=False,
        output_root=tmp_path / "output",
        max_visits_per_node=3,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        timeout_seconds=30.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.target.qualified_name == "Counter.classify"
    assert function_result.status is FunctionRunStatus.COMPLETED
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.line_coverage_percent == 100.0
    assert function_result.scenario_pool_coverage.branch_coverage_percent == 100.0
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert function_result.diagnostic is not None
    assert function_result.diagnostic.funnel.rl_executed_test_count is not None
    assert result.coverage_candidates
    assert all(
        "self" not in candidate.scenario.keyword_argument_dict
        for candidate in result.coverage_candidates
    )
    assert all(
        dict(candidate.scenario.constructor_arguments).keys() == {"value"}
        for candidate in result.coverage_candidates
    )
    report = result.report_path.read_text(encoding="utf-8")
    assert '"qualified_name": "Counter.classify"' in report
    assert "keyword_arguments" not in report
    generated_tests = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from counter_fixture import Counter" in generated_tests
    assert "target = Counter(" in generated_tests
    assert "target.classify(" in generated_tests
    assert "self=" not in generated_tests
