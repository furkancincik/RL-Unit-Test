from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from rl.coverage_state import CoverageState
from rl.scenario_coverage_transition import (
    ScenarioCoverageTransition,
)
from services.scenario_coverage_service import (
    ScenarioCoverageResult,
    ScenarioCoverageService,
)


def create_source_file(
    tmp_path: Path,
) -> Path:
    """Geçiş testlerinde kullanılacak kaynak dosyayı oluşturur."""
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 85:\n"
        "        return 'Başarılı'\n"
        "    return 'Başarısız'\n",
        encoding="utf-8",
    )

    return source_file


def create_scenario() -> Scenario:
    """Testlerde kullanılacak örnek Scenario nesnesini oluşturur."""
    return Scenario(
        scenario_id="calculate_score_scenario_001",
        name="calculate_score yürütme yolu 1",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 3, 4, 2),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=False,
        description=(
            "calculate_score fonksiyonunun başarılı yolu."
        ),
        keyword_arguments=(("score", 85),),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_current_state() -> CoverageState:
    """Testlerde kullanılacak mevcut RL durumunu oluşturur."""
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4),
        uncovered_branches=2,
    )


def create_coverage_result(
    *,
    line_coverage_percent: float = 75.0,
    branch_coverage_percent: float = 50.0,
    covered_line_count: int = 3,
    missing_line_count: int = 1,
    total_line_count: int = 4,
    covered_branch_count: int = 1,
    missing_branch_count: int = 1,
    total_branch_count: int = 2,
    test_exit_code: int = 0,
) -> CoverageResult:
    """Kontrollü CoverageResult nesnesi oluşturur."""
    return CoverageResult(
        source_file=Path("sample_code.py"),
        test_file=Path("test_calculate_score.py"),
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=test_exit_code,
        duration_seconds=0.25,
    )


def create_mock_service(
    scenario: Scenario,
    coverage_result: CoverageResult,
) -> Mock:
    """Mock ScenarioCoverageService bağımlılığı oluşturur."""
    service = Mock(
        spec=ScenarioCoverageService,
    )

    service.measure_scenario.return_value = (
        ScenarioCoverageResult(
            scenario=scenario,
            test_file=Path(
                "output/generated_tests/"
                "test_calculate_score_scenario_001.py"
            ),
            coverage=coverage_result,
        )
    )

    return service


def test_transition_returns_coverage_state(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()
    service = create_mock_service(
        scenario=scenario,
        coverage_result=create_coverage_result(),
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    result = transition(
        create_current_state(),
        scenario,
    )

    assert isinstance(result, CoverageState)


def test_transition_calls_scenario_coverage_service(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()
    service = create_mock_service(
        scenario=scenario,
        coverage_result=create_coverage_result(),
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
        overwrite=False,
        timeout_seconds=15.0,
    )

    transition(
        create_current_state(),
        scenario,
    )

    service.measure_scenario.assert_called_once_with(
        source_file=source_file.resolve(),
        module_path="sample_code",
        function_name="calculate_score",
        scenario=scenario,
        output_directory=tmp_path,
        overwrite=False,
        timeout_seconds=15.0,
    )


def test_transition_maps_line_coverage_to_next_state(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()

    coverage_result = create_coverage_result(
        line_coverage_percent=75.0,
    )

    service = create_mock_service(
        scenario=scenario,
        coverage_result=coverage_result,
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    result = transition(
        create_current_state(),
        scenario,
    )

    assert result.coverage_percentage == 75.0


def test_transition_increments_executed_test_count(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()
    service = create_mock_service(
        scenario=scenario,
        coverage_result=create_coverage_result(),
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    current_state = CoverageState(
        coverage_percentage=25.0,
        executed_tests=3,
        missing_lines=(1, 2, 3),
        uncovered_branches=2,
    )

    result = transition(
        current_state,
        scenario,
    )

    assert result.executed_tests == 4


def test_transition_preserves_missing_line_count(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()

    coverage_result = create_coverage_result(
        covered_line_count=2,
        missing_line_count=2,
        total_line_count=4,
    )

    service = create_mock_service(
        scenario=scenario,
        coverage_result=coverage_result,
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    result = transition(
        create_current_state(),
        scenario,
    )

    assert result.missing_line_count == 2
    assert result.missing_lines == (1, 2)


def test_transition_preserves_missing_branch_count(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()

    coverage_result = create_coverage_result(
        covered_branch_count=0,
        missing_branch_count=2,
        total_branch_count=2,
    )

    service = create_mock_service(
        scenario=scenario,
        coverage_result=coverage_result,
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    result = transition(
        create_current_state(),
        scenario,
    )

    assert result.uncovered_branches == 2


def test_transition_creates_full_coverage_state(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()

    coverage_result = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=4,
        missing_line_count=0,
        total_line_count=4,
        covered_branch_count=2,
        missing_branch_count=0,
        total_branch_count=2,
    )

    service = create_mock_service(
        scenario=scenario,
        coverage_result=coverage_result,
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    result = transition(
        create_current_state(),
        scenario,
    )

    assert result.coverage_percentage == 100.0
    assert result.missing_lines == ()
    assert result.uncovered_branches == 0
    assert result.is_fully_covered is True


def test_transition_rejects_failed_coverage_execution(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenario = create_scenario()

    coverage_result = create_coverage_result(
        test_exit_code=1,
    )

    service = create_mock_service(
        scenario=scenario,
        coverage_result=coverage_result,
    )

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        scenario_coverage_service=service,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Seçilen senaryonun coverage işlemi "
            "başarısız oldu"
        ),
    ):
        transition(
            create_current_state(),
            scenario,
        )


def test_transition_rejects_invalid_state(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="state bir CoverageState örneği olmalıdır",
    ):
        transition(
            "invalid",  # type: ignore[arg-type]
            create_scenario(),
        )


def test_transition_rejects_invalid_scenario(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    transition = ScenarioCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="scenario bir Scenario örneği olmalıdır",
    ):
        transition(
            create_current_state(),
            "invalid",  # type: ignore[arg-type]
        )


def test_transition_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        ScenarioCoverageTransition(
            source_file=tmp_path / "missing.py",
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_transition_rejects_invalid_module_path(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        ScenarioCoverageTransition(
            source_file=source_file,
            module_path="sample/code",
            function_name="calculate_score",
            output_directory=tmp_path,
        )


def test_transition_rejects_invalid_function_name(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    with pytest.raises(
        ValueError,
        match="Geçersiz Python fonksiyon adı",
    ):
        ScenarioCoverageTransition(
            source_file=source_file,
            module_path="sample_code",
            function_name="calculate-score",
            output_directory=tmp_path,
        )


def test_transition_rejects_invalid_service(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    with pytest.raises(
        TypeError,
        match=(
            "scenario_coverage_service bir "
            "ScenarioCoverageService örneği olmalıdır"
        ),
    ):
        ScenarioCoverageTransition(
            source_file=source_file,
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            scenario_coverage_service="invalid",  # type: ignore[arg-type]
        )


def test_transition_rejects_invalid_timeout(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)

    with pytest.raises(
        ValueError,
        match="timeout_seconds sıfırdan büyük olmalıdır",
    ):
        ScenarioCoverageTransition(
            source_file=source_file,
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            timeout_seconds=0.0,
        )