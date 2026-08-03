from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from generator.file_writer import GeneratedTestFileWriter
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from services.coverage_service import CoverageService
from services.scenario_coverage_service import (
    ScenarioCoverageResult,
    ScenarioCoverageService,
)


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
            "calculate_score fonksiyonunun birinci yolu."
        ),
        keyword_arguments=(("score", 85),),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_service_dependencies(
    tmp_path: Path,
) -> tuple[
    Mock,
    Mock,
    Mock,
    Mock,
]:
    """
    ScenarioCoverageService için kontrollü bağımlılıklar oluşturur.
    """
    pytest_generator = Mock(
        spec=PytestGenerator
    )
    file_writer = Mock(
        spec=GeneratedTestFileWriter
    )
    coverage_service = Mock(
        spec=CoverageService
    )
    coverage_result = Mock(
        spec=CoverageResult
    )

    pytest_generator.generate.return_value = (
        '"""Otomatik oluşturulmuş test."""\n'
        "\n"
        "def test_example() -> None:\n"
        "    assert True\n"
    )

    generated_file = (
        tmp_path
        / "test_calculate_score_scenario_001.py"
    )

    file_writer.write.return_value = generated_file

    coverage_result.success = True
    coverage_result.has_full_coverage = False

    coverage_service.measure.return_value = (
        coverage_result
    )

    return (
        pytest_generator,
        file_writer,
        coverage_service,
        coverage_result,
    )


def test_measure_scenario_returns_result_model(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    (
        pytest_generator,
        file_writer,
        coverage_service,
        coverage_result,
    ) = create_service_dependencies(tmp_path)

    service = ScenarioCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    scenario = create_scenario()

    result = service.measure_scenario(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenario=scenario,
        output_directory=tmp_path,
    )

    assert isinstance(
        result,
        ScenarioCoverageResult,
    )
    assert result.scenario is scenario
    assert result.coverage is coverage_result


def test_measure_scenario_generates_only_selected_scenario(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_service_dependencies(tmp_path)

    service = ScenarioCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    scenario = create_scenario()

    service.measure_scenario(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenario=scenario,
        output_directory=tmp_path,
    )

    pytest_generator.generate.assert_called_once_with(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=(scenario,),
    )


def test_measure_scenario_writes_generated_test_code(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_service_dependencies(tmp_path)

    generated_code = (
        pytest_generator.generate.return_value
    )

    service = ScenarioCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenario(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenario=create_scenario(),
        output_directory=tmp_path,
        overwrite=True,
    )

    written_call = file_writer.write.call_args

    assert (
        written_call.kwargs["source_code"]
        == generated_code
    )
    assert written_call.kwargs["overwrite"] is True


def test_measure_scenario_uses_stable_test_file_name(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_service_dependencies(tmp_path)

    service = ScenarioCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenario(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenario=create_scenario(),
        output_directory=tmp_path,
    )

    output_path = (
        file_writer.write
        .call_args.kwargs["output_path"]
    )

    assert output_path == (
        tmp_path
        / "test_calculate_score_scenario_001.py"
    )


def test_measure_scenario_runs_coverage_for_written_file(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    (
        pytest_generator,
        file_writer,
        coverage_service,
        _,
    ) = create_service_dependencies(tmp_path)

    written_file = file_writer.write.return_value

    service = ScenarioCoverageService(
        pytest_generator=pytest_generator,
        file_writer=file_writer,
        coverage_service=coverage_service,
    )

    service.measure_scenario(
        source_file=source_file,
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenario=create_scenario(),
        output_directory=tmp_path,
        timeout_seconds=15.0,
    )

    coverage_service.measure.assert_called_once_with(
        source_file=source_file.resolve(),
        test_file=written_file,
        timeout_seconds=15.0,
    )


def test_result_success_uses_coverage_result() -> None:
    coverage_result = Mock(
        spec=CoverageResult
    )
    coverage_result.success = True
    coverage_result.has_full_coverage = False

    result = ScenarioCoverageResult(
        scenario=create_scenario(),
        test_file=Path("test_example.py"),
        coverage=coverage_result,
    )

    assert result.success is True


def test_result_full_coverage_uses_coverage_result() -> None:
    coverage_result = Mock(
        spec=CoverageResult
    )
    coverage_result.success = True
    coverage_result.has_full_coverage = True

    result = ScenarioCoverageResult(
        scenario=create_scenario(),
        test_file=Path("test_example.py"),
        coverage=coverage_result,
    )

    assert result.has_full_coverage is True


def test_measure_scenario_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    service = ScenarioCoverageService()

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        service.measure_scenario(
            source_file=tmp_path / "missing.py",
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenario=create_scenario(),
            output_directory=tmp_path,
        )


def test_measure_scenario_rejects_non_python_source(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.txt"
    source_file.write_text(
        "invalid",
        encoding="utf-8",
    )

    service = ScenarioCoverageService()

    with pytest.raises(
        ValueError,
        match="source_file uzantısı .py olmalıdır",
    ):
        service.measure_scenario(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenario=create_scenario(),
            output_directory=tmp_path,
        )


def test_measure_scenario_rejects_invalid_module_path(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    service = ScenarioCoverageService()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        service.measure_scenario(
            source_file=source_file,
            module_path="datasets/sample_code",
            function_name="calculate_score",
            scenario=create_scenario(),
            output_directory=tmp_path,
        )


def test_measure_scenario_rejects_invalid_function_name(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    service = ScenarioCoverageService()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python fonksiyon adı",
    ):
        service.measure_scenario(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate-score",
            scenario=create_scenario(),
            output_directory=tmp_path,
        )


def test_measure_scenario_rejects_invalid_scenario(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "sample_code.py"
    source_file.write_text(
        "def calculate_score(score):\n"
        "    return score\n",
        encoding="utf-8",
    )

    service = ScenarioCoverageService()

    with pytest.raises(
        TypeError,
        match="scenario bir Scenario örneği olmalıdır",
    ):
        service.measure_scenario(
            source_file=source_file,
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenario="invalid",  # type: ignore[arg-type]
            output_directory=tmp_path,
        )