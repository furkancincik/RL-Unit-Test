from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from generator.scenario_generator import Scenario
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from rl.coverage_state import CoverageState
from rl.scenario_suite_coverage_transition import (
    ScenarioSuiteCoverageTransition,
)
from services.scenario_suite_coverage_service import (
    ScenarioSuiteCoverageResult,
    ScenarioSuiteCoverageService,
)


def create_source_file(
    tmp_path: Path,
) -> Path:
    """Testlerde kullanılacak örnek Python kaynak dosyasını oluşturur."""
    source_file = tmp_path / "sample_code.py"

    source_file.write_text(
        "def calculate_score(score):\n"
        "    if score >= 85:\n"
        "        return 'Başarılı'\n"
        "    if score >= 50:\n"
        "        return 'Orta'\n"
        "    return 'Başarısız'\n",
        encoding="utf-8",
    )

    return source_file


def create_scenario(
    *,
    scenario_id: str,
    path_index: int,
    priority_rank: int,
    score: int,
    expected_result: str,
) -> Scenario:
    """Geçiş testlerinde kullanılacak Scenario nesnesini oluşturur."""
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
        description="Kümülatif coverage geçiş testi.",
        keyword_arguments=(("score", score),),
        expected_result=expected_result,
        expected_exception=None,
    )


def create_scenarios() -> tuple[Scenario, ...]:
    """Episode içerisinde seçilebilecek örnek senaryoları oluşturur."""
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
        create_scenario(
            scenario_id="calculate_score_scenario_003",
            path_index=3,
            priority_rank=3,
            score=49,
            expected_result="Başarısız",
        ),
    )


def create_state(
    *,
    coverage_percentage: float = 0.0,
    executed_tests: int = 0,
    missing_lines: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    uncovered_branches: int = 4,
) -> CoverageState:
    """Kontrollü CoverageState oluşturur."""
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=missing_lines,
        uncovered_branches=uncovered_branches,
    )


def create_coverage_result(
    *,
    line_coverage_percent: float,
    branch_coverage_percent: float,
    covered_line_count: int,
    missing_line_count: int,
    total_line_count: int,
    covered_branch_count: int,
    missing_branch_count: int,
    total_branch_count: int,
    test_exit_code: int = 0,
) -> CoverageResult:
    """Kontrollü CoverageResult oluşturur."""
    return CoverageResult(
        source_file=Path("sample_code.py"),
        test_file=Path(
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
        duration_seconds=0.10,
    )


def create_function_coverage_result(
    *,
    function_line_coverage_percent: float = 100.0,
    function_branch_coverage_percent: float = 100.0,
    file_line_coverage_percent: float = 32.0,
    file_branch_coverage_percent: float = 25.0,
    test_exit_code: int = 0,
) -> FunctionCoverageResult:
    """Kontrollü FunctionCoverageResult oluşturur."""
    source_file = Path("sample_code.py")
    test_file = Path(
        "test_calculate_score_scenario_suite.py"
    )

    file_coverage = CoverageResult(
        source_file=source_file,
        test_file=test_file,
        line_coverage_percent=file_line_coverage_percent,
        branch_coverage_percent=file_branch_coverage_percent,
        covered_line_count=8,
        missing_line_count=17,
        total_line_count=25,
        covered_branch_count=2,
        missing_branch_count=6,
        total_branch_count=8,
        test_exit_code=test_exit_code,
        duration_seconds=0.10,
    )

    return FunctionCoverageResult(
        source_file=source_file,
        test_file=test_file,
        function_name="calculate_score",
        start_line=1,
        end_line=6,
        line_coverage_percent=function_line_coverage_percent,
        branch_coverage_percent=function_branch_coverage_percent,
        covered_lines=(1, 2, 3, 4, 5, 6),
        missing_lines=(),
        covered_branch_count=4,
        missing_branch_count=0,
        test_exit_code=test_exit_code,
        duration_seconds=0.10,
        file_coverage=file_coverage,
    )


def create_suite_result(
    *,
    scenarios: tuple[Scenario, ...],
    coverage_result: CoverageResult | FunctionCoverageResult,
) -> ScenarioSuiteCoverageResult:
    """ScenarioSuiteCoverageResult oluşturur."""
    return ScenarioSuiteCoverageResult(
        scenarios=scenarios,
        test_file=Path(
            "test_calculate_score_scenario_suite.py"
        ),
        coverage=coverage_result,
    )


def create_mock_service(
    results: list[ScenarioSuiteCoverageResult],
) -> Mock:
    """Sıralı sonuçlar döndüren mock coverage servisi oluşturur."""
    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )

    service.measure_scenarios.side_effect = results

    return service


def test_transition_starts_with_empty_selected_scenarios(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    assert transition.selected_scenarios == ()
    assert transition.selected_scenario_count == 0


def test_transition_adds_selected_scenario(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    coverage_result = create_coverage_result(
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
        covered_branch_count=2,
        missing_branch_count=2,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=coverage_result,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(
        create_state(),
        scenario,
    )

    assert transition.selected_scenarios == (
        scenario,
    )
    assert transition.selected_scenario_count == 1


def test_transition_passes_cumulative_scenarios_to_service(
    tmp_path: Path,
) -> None:
    first_scenario, second_scenario, _ = create_scenarios()

    first_result = create_suite_result(
        scenarios=(first_scenario,),
        coverage_result=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
        ),
    )

    second_result = create_suite_result(
        scenarios=(
            first_scenario,
            second_scenario,
        ),
        coverage_result=create_coverage_result(
            line_coverage_percent=83.33,
            branch_coverage_percent=75.0,
            covered_line_count=5,
            missing_line_count=1,
            total_line_count=6,
            covered_branch_count=3,
            missing_branch_count=1,
            total_branch_count=4,
        ),
    )

    service = create_mock_service(
        [
            first_result,
            second_result,
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    first_state = transition(
        create_state(),
        first_scenario,
    )

    transition(
        first_state,
        second_scenario,
    )

    first_call = (
        service.measure_scenarios
        .call_args_list[0]
        .kwargs
    )

    second_call = (
        service.measure_scenarios
        .call_args_list[1]
        .kwargs
    )

    assert first_call["scenarios"] == (
        first_scenario,
    )

    assert second_call["scenarios"] == (
        first_scenario,
        second_scenario,
    )


def test_transition_preserves_scenario_selection_order(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenarios[2],),
                coverage_result=create_coverage_result(
                    line_coverage_percent=50.0,
                    branch_coverage_percent=50.0,
                    covered_line_count=3,
                    missing_line_count=3,
                    total_line_count=6,
                    covered_branch_count=2,
                    missing_branch_count=2,
                    total_branch_count=4,
                ),
            ),
            create_suite_result(
                scenarios=(
                    scenarios[2],
                    scenarios[0],
                ),
                coverage_result=create_coverage_result(
                    line_coverage_percent=83.33,
                    branch_coverage_percent=75.0,
                    covered_line_count=5,
                    missing_line_count=1,
                    total_line_count=6,
                    covered_branch_count=3,
                    missing_branch_count=1,
                    total_branch_count=4,
                ),
            ),
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    state = transition(
        create_state(),
        scenarios[2],
    )

    transition(
        state,
        scenarios[0],
    )

    assert transition.selected_scenarios == (
        scenarios[2],
        scenarios[0],
    )


def test_transition_returns_coverage_state(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=create_coverage_result(
                    line_coverage_percent=50.0,
                    branch_coverage_percent=50.0,
                    covered_line_count=3,
                    missing_line_count=3,
                    total_line_count=6,
                    covered_branch_count=2,
                    missing_branch_count=2,
                    total_branch_count=4,
                ),
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    result = transition(
        create_state(),
        scenario,
    )

    assert isinstance(result, CoverageState)
    assert result.coverage_percentage == 50.0
    assert result.executed_tests == 1
    assert result.missing_line_count == 3
    assert result.uncovered_branches == 2


def test_transition_increments_executed_test_count(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=create_coverage_result(
                    line_coverage_percent=50.0,
                    branch_coverage_percent=50.0,
                    covered_line_count=3,
                    missing_line_count=3,
                    total_line_count=6,
                    covered_branch_count=2,
                    missing_branch_count=2,
                    total_branch_count=4,
                ),
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    result = transition(
        create_state(
            executed_tests=4,
        ),
        scenario,
    )

    assert result.executed_tests == 5


def test_transition_rejects_duplicate_scenario(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=create_coverage_result(
                    line_coverage_percent=50.0,
                    branch_coverage_percent=50.0,
                    covered_line_count=3,
                    missing_line_count=3,
                    total_line_count=6,
                    covered_branch_count=2,
                    missing_branch_count=2,
                    total_branch_count=4,
                ),
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    next_state = transition(
        create_state(),
        scenario,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Scenario episode içerisinde "
            "daha önce seçildi"
        ),
    ):
        transition(
            next_state,
            scenario,
        )

    assert transition.selected_scenarios == (
        scenario,
    )


def test_transition_rolls_back_scenario_when_service_raises(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )

    service.measure_scenarios.side_effect = RuntimeError(
        "Coverage hatası"
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    with pytest.raises(
        RuntimeError,
        match="Coverage hatası",
    ):
        transition(
            create_state(),
            scenario,
        )

    assert transition.selected_scenarios == ()
    assert transition.selected_scenario_count == 0


def test_transition_rolls_back_failed_suite_result(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    failed_result = create_suite_result(
        scenarios=(scenario,),
        coverage_result=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
            test_exit_code=1,
        ),
    )

    service = create_mock_service(
        [
            failed_result,
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    with pytest.raises(
        RuntimeError,
    ) as error_info:
        transition(
            create_state(),
            scenario,
        )

    error_message = str(error_info.value)

    assert (
        "Seçilen senaryo paketinin coverage işlemi "
        "başarısız oldu."
        in error_message
    )
    assert "Pytest çıkış kodu: 1." in error_message
    assert (
        "test_calculate_score_scenario_suite.py"
        in error_message
    )
    assert scenario.scenario_id in error_message
    assert "pytest " in error_message
    assert "-x -vv" in error_message

    assert transition.selected_scenarios == ()


def test_transition_reset_clears_selected_scenarios(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=create_coverage_result(
                    line_coverage_percent=50.0,
                    branch_coverage_percent=50.0,
                    covered_line_count=3,
                    missing_line_count=3,
                    total_line_count=6,
                    covered_branch_count=2,
                    missing_branch_count=2,
                    total_branch_count=4,
                ),
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(
        create_state(),
        scenario,
    )

    transition.reset()

    assert transition.selected_scenarios == ()
    assert transition.selected_scenario_count == 0


def test_transition_allows_scenario_again_after_reset(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]

    suite_result = create_suite_result(
        scenarios=(scenario,),
        coverage_result=create_coverage_result(
            line_coverage_percent=50.0,
            branch_coverage_percent=50.0,
            covered_line_count=3,
            missing_line_count=3,
            total_line_count=6,
            covered_branch_count=2,
            missing_branch_count=2,
            total_branch_count=4,
        ),
    )

    service = create_mock_service(
        [
            suite_result,
            suite_result,
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(
        create_state(),
        scenario,
    )

    transition.reset()

    result = transition(
        create_state(),
        scenario,
    )

    assert result.coverage_percentage == 50.0
    assert transition.selected_scenarios == (
        scenario,
    )


def test_transition_rejects_invalid_state(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
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
            create_scenarios()[0],
        )


def test_transition_rejects_invalid_scenario(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="scenario bir Scenario örneği olmalıdır",
    ):
        transition(
            create_state(),
            "invalid",  # type: ignore[arg-type]
        )


def test_transition_rejects_invalid_coverage_service(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "coverage_service bir "
            "ScenarioSuiteCoverageService örneği olmalıdır"
        ),
    ):
        ScenarioSuiteCoverageTransition(
            source_file=create_source_file(tmp_path),
            module_path="sample_code",
            function_name="calculate_score",
            output_directory=tmp_path,
            coverage_service="invalid",  # type: ignore[arg-type]
        )


def test_transition_starts_without_last_coverage_result(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    assert transition.last_coverage_result is None
    assert transition.last_function_coverage is None
    assert transition.last_file_coverage is None


def test_transition_stores_last_file_coverage_result(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]
    coverage_result = create_coverage_result(
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
        covered_branch_count=2,
        missing_branch_count=2,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=coverage_result,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(create_state(), scenario)

    assert transition.last_coverage_result is coverage_result
    assert transition.last_function_coverage is None
    assert transition.last_file_coverage is coverage_result


def test_transition_exposes_function_and_file_coverage(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]
    function_coverage = create_function_coverage_result()

    service = create_mock_service(
        [
            ScenarioSuiteCoverageResult(
                scenarios=(scenario,),
                test_file=Path(
                    "test_calculate_score_scenario_suite.py"
                ),
                coverage=function_coverage,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        function_start_line=1,
        function_end_line=6,
        output_directory=tmp_path,
        coverage_service=service,
    )

    result = transition(create_state(), scenario)

    assert result.coverage_percentage == 100.0
    assert transition.last_coverage_result is function_coverage
    assert transition.last_function_coverage is function_coverage
    assert (
        transition.last_file_coverage
        is function_coverage.file_coverage
    )
    assert (
        transition.last_file_coverage.line_coverage_percent
        == 32.0
    )


def test_transition_preserves_previous_coverage_when_service_raises(
    tmp_path: Path,
) -> None:
    first_scenario, second_scenario, _ = create_scenarios()
    first_coverage = create_coverage_result(
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
        covered_branch_count=2,
        missing_branch_count=2,
        total_branch_count=4,
    )

    service = Mock(
        spec=ScenarioSuiteCoverageService,
    )
    service.measure_scenarios.side_effect = [
        create_suite_result(
            scenarios=(first_scenario,),
            coverage_result=first_coverage,
        ),
        RuntimeError("Coverage hatası"),
    ]

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    first_state = transition(
        create_state(),
        first_scenario,
    )

    with pytest.raises(
        RuntimeError,
        match="Coverage hatası",
    ):
        transition(
            first_state,
            second_scenario,
        )

    assert transition.last_coverage_result is first_coverage
    assert transition.selected_scenarios == (
        first_scenario,
    )


def test_transition_reset_preserves_last_coverage_result(
    tmp_path: Path,
) -> None:
    scenario = create_scenarios()[0]
    coverage_result = create_coverage_result(
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
        covered_branch_count=2,
        missing_branch_count=2,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(scenario,),
                coverage_result=coverage_result,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(create_state(), scenario)
    transition.reset()

    assert transition.selected_scenarios == ()
    assert transition.last_coverage_result is coverage_result


def test_measure_scenarios_returns_baseline_file_coverage_without_mutating_episode(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    baseline_coverage = create_coverage_result(
        line_coverage_percent=83.33,
        branch_coverage_percent=75.0,
        covered_line_count=5,
        missing_line_count=1,
        total_line_count=6,
        covered_branch_count=3,
        missing_branch_count=1,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=scenarios,
                coverage_result=baseline_coverage,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    result = transition.measure_scenarios(
        scenarios
    )

    assert result is baseline_coverage
    assert transition.selected_scenarios == ()
    assert transition.selected_scenario_count == 0
    assert transition.last_coverage_result is None
    assert transition.last_function_coverage is None
    assert transition.last_file_coverage is None


def test_measure_scenarios_passes_complete_baseline_package_to_service(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    baseline_coverage = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=6,
        missing_line_count=0,
        total_line_count=6,
        covered_branch_count=4,
        missing_branch_count=0,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=scenarios,
                coverage_result=baseline_coverage,
            )
        ]
    )

    source_file = create_source_file(
        tmp_path
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=source_file,
        module_path="sample_code",
        function_name="calculate_score",
        function_start_line=1,
        function_end_line=6,
        output_directory=tmp_path,
        coverage_service=service,
        overwrite=False,
        timeout_seconds=12.5,
    )

    transition.measure_scenarios(
        scenarios
    )

    service.measure_scenarios.assert_called_once_with(
        source_file=source_file.resolve(),
        module_path="sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
        output_directory=tmp_path,
        function_start_line=1,
        function_end_line=6,
        overwrite=False,
        timeout_seconds=12.5,
    )


def test_measure_scenarios_returns_function_coverage_without_mutating_last_result(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()
    function_coverage = (
        create_function_coverage_result(
            function_line_coverage_percent=83.33,
            function_branch_coverage_percent=75.0,
        )
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=scenarios,
                coverage_result=function_coverage,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        function_start_line=1,
        function_end_line=6,
        output_directory=tmp_path,
        coverage_service=service,
    )

    result = transition.measure_scenarios(
        scenarios
    )

    assert result is function_coverage
    assert result.line_coverage_percent == 83.33
    assert result.branch_coverage_percent == 75.0

    assert transition.selected_scenarios == ()
    assert transition.last_coverage_result is None
    assert transition.last_function_coverage is None
    assert transition.last_file_coverage is None


def test_measure_scenarios_does_not_replace_previous_episode_coverage(
    tmp_path: Path,
) -> None:
    first_scenario = create_scenarios()[0]
    all_scenarios = create_scenarios()

    previous_coverage = create_coverage_result(
        line_coverage_percent=50.0,
        branch_coverage_percent=50.0,
        covered_line_count=3,
        missing_line_count=3,
        total_line_count=6,
        covered_branch_count=2,
        missing_branch_count=2,
        total_branch_count=4,
    )

    baseline_coverage = create_coverage_result(
        line_coverage_percent=100.0,
        branch_coverage_percent=100.0,
        covered_line_count=6,
        missing_line_count=0,
        total_line_count=6,
        covered_branch_count=4,
        missing_branch_count=0,
        total_branch_count=4,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=(first_scenario,),
                coverage_result=previous_coverage,
            ),
            create_suite_result(
                scenarios=all_scenarios,
                coverage_result=baseline_coverage,
            ),
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    transition(
        create_state(),
        first_scenario,
    )

    baseline_result = (
        transition.measure_scenarios(
            all_scenarios
        )
    )

    assert baseline_result is baseline_coverage

    assert transition.selected_scenarios == (
        first_scenario,
    )
    assert transition.selected_scenario_count == 1
    assert (
        transition.last_coverage_result
        is previous_coverage
    )


def test_measure_scenarios_raises_for_failed_baseline_without_mutating_episode(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()

    failed_coverage = create_coverage_result(
        line_coverage_percent=83.33,
        branch_coverage_percent=75.0,
        covered_line_count=5,
        missing_line_count=1,
        total_line_count=6,
        covered_branch_count=3,
        missing_branch_count=1,
        total_branch_count=4,
        test_exit_code=1,
    )

    service = create_mock_service(
        [
            create_suite_result(
                scenarios=scenarios,
                coverage_result=failed_coverage,
            )
        ]
    )

    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
        coverage_service=service,
    )

    with pytest.raises(
        RuntimeError,
    ) as error_info:
        transition.measure_scenarios(
            scenarios
        )

    error_message = str(
        error_info.value
    )

    assert (
        "Baseline coverage ölçümü başarısız oldu."
        in error_message
    )
    assert (
        "Pytest çıkış kodu: 1."
        in error_message
    )
    assert (
        "test_calculate_score_scenario_suite.py"
        in error_message
    )

    for scenario in scenarios:
        assert (
            scenario.scenario_id
            in error_message
        )

    assert "pytest " in error_message
    assert "-x -vv" in error_message

    assert transition.selected_scenarios == ()
    assert transition.selected_scenario_count == 0
    assert transition.last_coverage_result is None


@pytest.mark.parametrize(
    "invalid_scenarios",
    (
        [],
        "invalid",
        42,
    ),
)
def test_measure_scenarios_rejects_non_tuple_input(
    tmp_path: Path,
    invalid_scenarios: object,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match=(
            "scenarios bir Scenario tuple'ı "
            "olmalıdır."
        ),
    ):
        transition.measure_scenarios(
            invalid_scenarios,  # type: ignore[arg-type]
        )


def test_measure_scenarios_rejects_empty_tuple(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="scenarios boş olamaz.",
    ):
        transition.measure_scenarios(
            ()
        )


def test_measure_scenarios_rejects_invalid_scenario_item(
    tmp_path: Path,
) -> None:
    transition = ScenarioSuiteCoverageTransition(
        source_file=create_source_file(tmp_path),
        module_path="sample_code",
        function_name="calculate_score",
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match=(
            "scenarios yalnızca Scenario "
            "nesneleri içermelidir."
        ),
    ):
        transition.measure_scenarios(
            (
                create_scenarios()[0],
                "invalid",  # type: ignore[arg-type]
            )
        )
