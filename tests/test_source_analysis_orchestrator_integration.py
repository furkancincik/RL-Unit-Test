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
