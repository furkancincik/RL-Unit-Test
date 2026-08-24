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


def test_inferred_top_level_parameter_reaches_the_real_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "renamed_numeric_module.py"
    source_file.write_text(
        "def categorize(signal):\n"
        "    if signal < -7:\n"
        "        return 'below-private-marker'\n"
        "    return 'other-private-marker'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="renamed_numeric_module",
        function_name="categorize",
        all_functions=False,
        output_root=tmp_path / "numeric_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        timeout_seconds=30.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status is FunctionRunStatus.COMPLETED
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.line_coverage_percent == 100.0
    assert function_result.scenario_pool_coverage.branch_coverage_percent == 100.0
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert result.coverage_candidates
    assert all(
        type(candidate.scenario.keyword_argument_dict["signal"]) is int
        for candidate in result.coverage_candidates
    )
    public_report = result.report_path.read_text(encoding="utf-8")
    assert "def categorize" not in public_report
    assert "private-marker" not in public_report
    assert "keyword_arguments" not in public_report


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


def test_inferred_instance_parameters_run_pytest_coverage_greedy_and_rl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "wallet_fixture.py"
    source_file.write_text(
        "class Wallet:\n"
        "    def __init__(self, balance=0):\n"
        "        self.balance = balance\n\n"
        "    def withdraw(self, amount):\n"
        "        if amount <= 0:\n"
        "            return 'invalid'\n"
        "        if amount > self.balance:\n"
        "            return 'insufficient'\n"
        "        self.balance -= amount\n"
        "        return 'ok'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="wallet_fixture",
        function_name="Wallet.withdraw",
        all_functions=False,
        output_root=tmp_path / "wallet_output",
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
    assert function_result.status is FunctionRunStatus.COMPLETED
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.line_coverage_percent > 0.0
    assert function_result.scenario_pool_coverage.branch_coverage_percent > 0.0
    assert function_result.scenario_pool_coverage.covered_lines
    assert function_result.scenario_pool_coverage.covered_branches
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
        set(dict(candidate.scenario.constructor_arguments)) == {"balance"}
        for candidate in result.coverage_candidates
    )
    generated_tests = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from wallet_fixture import Wallet" in generated_tests
    assert "target = Wallet(" in generated_tests
    assert "target.withdraw(" in generated_tests
    assert "assert result ==" in generated_tests
    assert "self=" not in generated_tests
    report = result.report_path.read_text(encoding="utf-8")
    assert "def withdraw" not in report
    assert "keyword_arguments" not in report
    assert "constructor_arguments" not in report


def test_empty_collection_method_runs_real_pytest_coverage_and_greedy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "empty_collection_fixture.py"
    source_file.write_text(
        "class Repository:\n"
        "    def __init__(self):\n"
        "        self.entries = []\n\n"
        "    def classify(self, token: str) -> str:\n"
        "        if token in self.entries:\n"
        "            return 'present'\n"
        "        if not self.entries:\n"
        "            return 'empty'\n"
        "        return 'other'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="empty_collection_fixture",
        function_name="Repository.classify",
        all_functions=False,
        output_root=tmp_path / "collection_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status is FunctionRunStatus.COMPLETED, (
        (
            function_result.diagnostic.error_category,
            function_result.diagnostic.error_message,
            function_result.diagnostic.stopped_stage,
            function_result.diagnostic.scenario_rejection_counts,
        )
        if function_result.diagnostic is not None
        else None
    )
    assert function_result.scenario_count > 0
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.covered_lines
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert result.coverage_candidates
    assert all(
        not candidate.scenario.constructor_arguments
        for candidate in result.coverage_candidates
    )
    assert all(
        "self" not in candidate.scenario.keyword_argument_dict
        for candidate in result.coverage_candidates
    )
    generated_tests = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from empty_collection_fixture import Repository" in generated_tests
    assert "target = Repository()" in generated_tests
    assert "target.classify(" in generated_tests
    assert "assert result ==" in generated_tests
    report = result.report_path.read_text(encoding="utf-8")
    assert "__self_" not in report
    assert "constructor_arguments" not in report
