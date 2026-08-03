from __future__ import annotations

from pathlib import Path

from generator.scenario_generator import Scenario
from services.scenario_suite_coverage_service import (
    ScenarioSuiteCoverageResult,
    ScenarioSuiteCoverageService,
)


SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


def create_success_scenario() -> Scenario:
    """score >= 85 yürütme yolunu temsil eder."""
    return Scenario(
        scenario_id="calculate_score_scenario_001",
        name="calculate_score başarılı yürütme yolu",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 3, 4, 2),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=False,
        description=(
            "score değeri 85 veya daha büyük olduğunda "
            "Başarılı sonucu beklenir."
        ),
        keyword_arguments=(("score", 85),),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_medium_scenario() -> Scenario:
    """50 <= score < 85 yürütme yolunu temsil eder."""
    return Scenario(
        scenario_id="calculate_score_scenario_002",
        name="calculate_score orta yürütme yolu",
        path_index=2,
        priority_rank=2,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 3, 5, 6, 2),
        edge_labels=(None, "False", "True", None),
        contains_loop=False,
        contains_exception=False,
        description=(
            "score değeri 50 ile 85 arasında olduğunda "
            "Orta sonucu beklenir."
        ),
        keyword_arguments=(("score", 50),),
        expected_result="Orta",
        expected_exception=None,
    )


def create_failure_scenario() -> Scenario:
    """score < 50 yürütme yolunu temsil eder."""
    return Scenario(
        scenario_id="calculate_score_scenario_003",
        name="calculate_score başarısız yürütme yolu",
        path_index=3,
        priority_rank=3,
        priority_level="Medium",
        dqm_score=76.47,
        node_ids=(1, 3, 5, 7, 2),
        edge_labels=(None, "False", "False", None),
        contains_loop=False,
        contains_exception=False,
        description=(
            "score değeri 50'nin altında olduğunda "
            "Başarısız sonucu beklenir."
        ),
        keyword_arguments=(("score", 49),),
        expected_result="Başarısız",
        expected_exception=None,
    )


def test_real_scenario_suite_coverage_is_cumulative(
    tmp_path: Path,
) -> None:
    """
    Seçilen senaryo sayısı arttıkça coverage oranının kümülatif
    biçimde arttığını ve bütün yollar eklendiğinde tam coverage
    sağlandığını doğrular.
    """
    service = ScenarioSuiteCoverageService()

    success_scenario = create_success_scenario()
    medium_scenario = create_medium_scenario()
    failure_scenario = create_failure_scenario()

    first_result = service.measure_scenarios(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=(
            success_scenario,
        ),
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    second_result = service.measure_scenarios(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=(
            success_scenario,
            medium_scenario,
        ),
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    third_result = service.measure_scenarios(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=(
            success_scenario,
            medium_scenario,
            failure_scenario,
        ),
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    assert isinstance(
        first_result,
        ScenarioSuiteCoverageResult,
    )
    assert isinstance(
        second_result,
        ScenarioSuiteCoverageResult,
    )
    assert isinstance(
        third_result,
        ScenarioSuiteCoverageResult,
    )

    assert first_result.success is True
    assert second_result.success is True
    assert third_result.success is True

    assert first_result.scenario_count == 1
    assert second_result.scenario_count == 2
    assert third_result.scenario_count == 3

    assert (
        second_result.coverage.line_coverage_percent
        > first_result.coverage.line_coverage_percent
    )

    assert (
        third_result.coverage.line_coverage_percent
        >= second_result.coverage.line_coverage_percent
    )

    assert (
        second_result.coverage.branch_coverage_percent
        > first_result.coverage.branch_coverage_percent
    )

    assert (
        third_result.coverage.branch_coverage_percent
        > second_result.coverage.branch_coverage_percent
    )

    assert (
        third_result.coverage.line_coverage_percent
        == 100.0
    )
    assert (
        third_result.coverage.branch_coverage_percent
        == 100.0
    )

    assert (
        third_result.coverage.missing_line_count
        == 0
    )
    assert (
        third_result.coverage.missing_branch_count
        == 0
    )
    assert third_result.has_full_coverage is True


def test_real_scenario_suite_generates_all_selected_tests(
    tmp_path: Path,
) -> None:
    """Üretilen dosyada seçilen üç senaryonun da bulunduğunu doğrular."""
    scenarios = (
        create_success_scenario(),
        create_medium_scenario(),
        create_failure_scenario(),
    )

    result = ScenarioSuiteCoverageService().measure_scenarios(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=scenarios,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    assert result.test_file.exists()

    generated_source = result.test_file.read_text(
        encoding="utf-8",
    )

    assert generated_source.count(
        "def test_calculate_score_"
    ) == 3

    assert (
        "result = calculate_score(score=85)"
        in generated_source
    )
    assert (
        "result = calculate_score(score=50)"
        in generated_source
    )
    assert (
        "result = calculate_score(score=49)"
        in generated_source
    )

    assert (
        "assert result == 'Başarılı'"
        in generated_source
    )
    assert (
        "assert result == 'Orta'"
        in generated_source
    )
    assert (
        "assert result == 'Başarısız'"
        in generated_source
    )

    assert "assert callable" not in generated_source
    assert "TODO" not in generated_source