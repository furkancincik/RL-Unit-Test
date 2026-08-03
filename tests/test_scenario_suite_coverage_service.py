from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from generator.file_writer import GeneratedTestFileWriter
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from services.coverage_service import CoverageService
from services.scenario_suite_coverage_service import (
    ScenarioSuiteCoverageResult,
    ScenarioSuiteCoverageService,
)


def create_scenario(
    *,
    scenario_id: str,
    path_index: int,
    priority_rank: int,
    score: int,
    expected_result: str,
) -> Scenario:
    """Testlerde kullanılacak örnek Scenario nesnesini oluşturur."""
    return Scenario(
        scenario_id=scenario_id,
        name=f"calculate_score yürütme yolu {path_index}",
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 3, 4, 2),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=False,
        description="Scenario suite coverage testi.",
        keyword_arguments=(("score", score),),
        expected_result=expected_result,
        expected_exception=None,
    )


def create_scenarios() -> tuple[Scenario, ...]:
    """Birlikte çalıştırılacak örnek senaryoları oluşturur."""
    return (
        create_scenario(
            scenario_id="calculate_score_scenario_001",
            path_index=1,
            priority_rank=1,
            score=85,
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="calculate_score_scenario_002",
            path_index=2,
            priority_rank=2,
            score=50,
            expected_result="Orta",
        ),
    )


def create_coverage_result(
    *,
    line_coverage_percent: float = 75.0,
    branch_coverage_percent: float = 75.0,
    covered_line_count: int = 6,
    missing_line_count: int = 2,
    total_line_count: int = 8,
    covered_branch_count: int = 3,
    missing_branch_count: int = 1,
    total_branch_count: int = 4,
    test_exit_code: int = 0,
) -> CoverageResult:
    """Kontrollü CoverageResult nesnesi oluşturur."""
    return CoverageResult(
        source_file=Path("datasets/sample_code.py"),
        test_file=Path(
            "output/generated_tests/"
            "test_calculate_score_scenario_suite.py"
        ),
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


def create_dependencies(
    tmp_path: Path,
) -> tuple[
    Mock,
    Mock,
    Mock,
    CoverageResult,
]:
    """Servis için kontrollü mock bağımlılıkları oluşturur."""
    pytest_generator = Mock(
        spec=PytestGenerator,
    )
    file_writer = Mock(
        spec=GeneratedTestFileWriter,
    )
    coverage_service = Mock(
        spec=CoverageService,
    )

    generated_code = (
        '"""Otomatik oluşturulmuş test paketi."""\n'
        "\n"
        "def test_example() -> None:\n"
        "    assert True\n"
    )

    written_file = (
        tmp_path
        / "test_calculate_score_scenario_suite.py"
    )

    coverage_result = create_coverage_result()

    pytest_generator.generate.return_value = generated_code
    file_writer.write.return_value = written_file
    coverage_service.measure.return_value = coverage_result

    return (
        pytest_generator,
        file_writer,
        coverage_service,
        coverage_result,
    )


def create_source_file(
    tmp_path: Path,
) -> Path:
    """Testlerde kullanılacak Python kaynak dosyasını oluşturur."""
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    return source_file


def test_measure_scenarios_returns_result_model(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenarios = create_scenarios()

    (
        pytest_generator,
        file_writer,
        coverage_service,
        coverage_result,
    ) = create_dependencies(tmp_path)

    service = ScenarioSuiteCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    result = service.measure_scenarios(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
        output_directory=tmp_path,
    )

    assert isinstance(
        result,
        ScenarioSuiteCoverageResult,
    )
    assert result.scenarios == scenarios
    assert result.coverage is coverage_result


def test_measure_scenarios_passes_all_scenarios_to_generator(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenarios = create_scenarios()

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_dependencies(tmp_path)

    service = ScenarioSuiteCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenarios(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
        output_directory=tmp_path,
    )

    pytest_generator.generate.assert_called_once_with(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
    )


def test_measure_scenarios_writes_generated_suite(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenarios = create_scenarios()

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_dependencies(tmp_path)

    generated_code = (
        pytest_generator.generate.return_value
    )

    service = ScenarioSuiteCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenarios(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
        output_directory=tmp_path,
        overwrite=False,
    )

    file_writer.write.assert_called_once_with(
        source_code=generated_code,
        output_path=(
            tmp_path
            / "test_calculate_score_scenario_suite.py"
        ),
        overwrite=False,
    )


def test_measure_scenarios_runs_coverage_for_suite_file(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    scenarios = create_scenarios()

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_dependencies(tmp_path)

    written_file = file_writer.write.return_value

    service = ScenarioSuiteCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenarios(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
        output_directory=tmp_path,
        timeout_seconds=15.0,
    )

    coverage_service.measure.assert_called_once_with(
        source_file=source_file.resolve(),
        test_file=written_file,
        timeout_seconds=15.0,
    )


def test_result_returns_scenario_count() -> None:
    scenarios = create_scenarios()

    result = ScenarioSuiteCoverageResult(
        scenarios=scenarios,
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
        coverage=create_coverage_result(),
    )

    assert result.scenario_count == 2


def test_result_success_uses_coverage_result() -> None:
    result = ScenarioSuiteCoverageResult(
        scenarios=create_scenarios(),
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
        coverage=create_coverage_result(
            test_exit_code=0,
        ),
    )

    assert result.success is True


def test_result_full_coverage_uses_coverage_result() -> None:
    coverage_result = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=8,
        missing_line_count=0,
        total_line_count=8,
        covered_branch_count=4,
        missing_branch_count=0,
        total_branch_count=4,
    )

    result = ScenarioSuiteCoverageResult(
        scenarios=create_scenarios(),
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
        coverage=coverage_result,
    )

    assert result.has_full_coverage is True


def test_measure_scenarios_rejects_empty_scenario_tuple(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        ValueError,
        match="En az bir Scenario gereklidir",
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=(),
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_non_tuple_scenarios(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        TypeError,
        match="scenarios bir Scenario tuple'ı olmalıdır",
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=list(create_scenarios()),  # type: ignore[arg-type]
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_invalid_scenario_item(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        TypeError,
        match=(
            "scenarios yalnızca Scenario "
            "nesneleri içermelidir"
        ),
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=(
                create_scenarios()[0],
                "invalid",  # type: ignore[arg-type]
            ),
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_duplicate_scenarios(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    scenario = create_scenarios()[0]

    with pytest.raises(
        ValueError,
        match=(
            "scenarios tekrar eden "
            "Scenario içeremez"
        ),
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=(
                scenario,
                scenario,
            ),
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        service.measure_scenarios(
            source_file=tmp_path / "missing.py",
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=create_scenarios(),
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_invalid_module_path(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets/sample_code",
            function_name="calculate_score",
            scenarios=create_scenarios(),
            output_directory=tmp_path,
        )


def test_measure_scenarios_rejects_invalid_function_name(
    tmp_path: Path,
) -> None:
    source_file = create_source_file(tmp_path)
    service = ScenarioSuiteCoverageService()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python fonksiyon adı",
    ):
        service.measure_scenarios(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate-score",
            scenarios=create_scenarios(),
            output_directory=tmp_path,
        )