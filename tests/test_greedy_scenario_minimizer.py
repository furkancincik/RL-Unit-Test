from __future__ import annotations

from pathlib import Path

import pytest

from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from services.greedy_scenario_minimizer import (
    GreedyScenarioMinimizationResult,
    GreedyScenarioMinimizer,
    GreedyScenarioSelectionStep,
)


def create_scenario(
    scenario_id: str,
    path_index: int,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        name=f"{scenario_id} test scenario",
        path_index=path_index,
        priority_rank=path_index + 1,
        priority_level="Medium",
        dqm_score=50.0,
        node_ids=(1, 2, 3),
        edge_labels=(None, "True"),
        contains_loop=False,
        contains_exception=False,
        description="Greedy minimizer test scenario.",
        keyword_arguments=(("value", path_index),),
        expected_result=path_index,
        expected_exception=None,
    )


def create_coverage_result(
    coverage_percentage: float,
) -> CoverageResult:
    """
    Greedy minimizer testlerinde kullanılmak üzere
    kontrollü CoverageResult oluşturur.
    """
    total_line_count = 100

    covered_line_count = int(
        round(
            total_line_count
            * coverage_percentage
            / 100.0
        )
    )

    missing_line_count = (
        total_line_count
        - covered_line_count
    )

    total_branch_count = 100

    covered_branch_count = int(
        round(
            total_branch_count
            * coverage_percentage
            / 100.0
        )
    )

    missing_branch_count = (
        total_branch_count
        - covered_branch_count
    )

    return CoverageResult(
        source_file=Path("sample.py"),
        test_file=Path("test_sample.py"),
        line_coverage_percent=coverage_percentage,
        branch_coverage_percent=coverage_percentage,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=0,
        duration_seconds=0.01,
    )


def test_selection_step_can_be_created() -> None:
    scenario = create_scenario(
        "scenario_001",
        0,
    )

    step = GreedyScenarioSelectionStep(
        step_number=1,
        scenario=scenario,
        coverage_before=0.0,
        coverage_after=40.0,
        marginal_gain=40.0,
    )

    assert step.step_number == 1
    assert step.scenario == scenario
    assert step.coverage_before == 0.0
    assert step.coverage_after == 40.0
    assert step.marginal_gain == 40.0


def test_result_reports_selected_scenario_count() -> None:
    first = create_scenario(
        "scenario_001",
        0,
    )

    second = create_scenario(
        "scenario_002",
        1,
    )

    result = GreedyScenarioMinimizationResult(
        selected_scenarios=(
            first,
            second,
        ),
        steps=(),
        target_coverage_percentage=90.0,
        final_coverage_percentage=90.0,
    )

    assert result.selected_scenario_count == 2
    assert result.target_reached is True


def test_result_reports_target_not_reached() -> None:
    scenario = create_scenario(
        "scenario_001",
        0,
    )

    result = GreedyScenarioMinimizationResult(
        selected_scenarios=(
            scenario,
        ),
        steps=(),
        target_coverage_percentage=90.0,
        final_coverage_percentage=70.0,
    )

    assert result.target_reached is False


def test_minimizer_selects_highest_initial_coverage() -> None:
    first = create_scenario(
        "scenario_001",
        0,
    )

    second = create_scenario(
        "scenario_002",
        1,
    )

    third = create_scenario(
        "scenario_003",
        2,
    )

    coverage_by_suite = {
        ("scenario_001",): 20.0,
        ("scenario_002",): 40.0,
        ("scenario_003",): 30.0,
        ("scenario_002", "scenario_001"): 70.0,
        ("scenario_002", "scenario_003"): 60.0,
        (
            "scenario_002",
            "scenario_001",
            "scenario_003",
        ): 90.0,
    }

    def measure(
        scenarios: tuple[Scenario, ...],
    ) -> CoverageResult:
        key = tuple(
            scenario.scenario_id
            for scenario in scenarios
        )

        return create_coverage_result(
            coverage_by_suite[key]
        )

    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=measure,
    )

    result = minimizer.minimize(
        scenarios=(
            first,
            second,
            third,
        ),
        target_coverage_percentage=90.0,
    )

    assert (
        result.selected_scenarios[0]
        == second
    )


def test_minimizer_uses_marginal_coverage_gain() -> None:
    first = create_scenario(
        "scenario_001",
        0,
    )

    second = create_scenario(
        "scenario_002",
        1,
    )

    third = create_scenario(
        "scenario_003",
        2,
    )

    coverage_by_suite = {
        ("scenario_001",): 30.0,
        ("scenario_002",): 50.0,
        ("scenario_003",): 40.0,

        (
            "scenario_002",
            "scenario_001",
        ): 55.0,

        (
            "scenario_002",
            "scenario_003",
        ): 80.0,

        (
            "scenario_002",
            "scenario_003",
            "scenario_001",
        ): 90.0,
    }

    def measure(
        scenarios: tuple[Scenario, ...],
    ) -> CoverageResult:
        key = tuple(
            scenario.scenario_id
            for scenario in scenarios
        )

        return create_coverage_result(
            coverage_by_suite[key]
        )

    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=measure,
    )

    result = minimizer.minimize(
        scenarios=(
            first,
            second,
            third,
        ),
        target_coverage_percentage=80.0,
    )

    assert result.selected_scenarios == (
        second,
        third,
    )

    assert result.selected_scenario_count == 2
    assert result.final_coverage_percentage == 80.0
    assert result.target_reached is True

    assert (
        result.steps[0].marginal_gain
        == pytest.approx(50.0)
    )

    assert (
        result.steps[1].marginal_gain
        == pytest.approx(30.0)
    )


def test_minimizer_stops_when_target_is_reached() -> None:
    first = create_scenario(
        "scenario_001",
        0,
    )

    second = create_scenario(
        "scenario_002",
        1,
    )

    coverage_by_suite = {
        ("scenario_001",): 90.0,
        ("scenario_002",): 50.0,
    }

    def measure(
        scenarios: tuple[Scenario, ...],
    ) -> CoverageResult:
        key = tuple(
            scenario.scenario_id
            for scenario in scenarios
        )

        return create_coverage_result(
            coverage_by_suite[key]
        )

    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=measure,
    )

    result = minimizer.minimize(
        scenarios=(
            first,
            second,
        ),
        target_coverage_percentage=90.0,
    )

    assert result.selected_scenarios == (
        first,
    )

    assert result.selected_scenario_count == 1
    assert result.target_reached is True


def test_minimizer_stops_when_no_positive_gain_remains() -> None:
    first = create_scenario(
        "scenario_001",
        0,
    )

    second = create_scenario(
        "scenario_002",
        1,
    )

    coverage_by_suite = {
        ("scenario_001",): 60.0,
        ("scenario_002",): 40.0,
        (
            "scenario_001",
            "scenario_002",
        ): 60.0,
    }

    def measure(
        scenarios: tuple[Scenario, ...],
    ) -> CoverageResult:
        key = tuple(
            scenario.scenario_id
            for scenario in scenarios
        )

        return create_coverage_result(
            coverage_by_suite[key]
        )

    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=measure,
    )

    result = minimizer.minimize(
        scenarios=(
            first,
            second,
        ),
        target_coverage_percentage=90.0,
    )

    assert result.selected_scenarios == (
        first,
    )

    assert result.selected_scenario_count == 1
    assert result.final_coverage_percentage == 60.0
    assert result.target_reached is False


def test_minimizer_rejects_empty_scenarios() -> None:
    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=lambda scenarios: (
            create_coverage_result(0.0)
        ),
    )

    with pytest.raises(
        ValueError,
        match="scenarios boş olamaz.",
    ):
        minimizer.minimize(
            scenarios=(),
            target_coverage_percentage=90.0,
        )


@pytest.mark.parametrize(
    "target",
    (
        0.0,
        -1.0,
        100.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_minimizer_rejects_invalid_target(
    target: float,
) -> None:
    scenario = create_scenario(
        "scenario_001",
        0,
    )

    minimizer = GreedyScenarioMinimizer(
        coverage_measurement=lambda scenarios: (
            create_coverage_result(50.0)
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        minimizer.minimize(
            scenarios=(
                scenario,
            ),
            target_coverage_percentage=target,
        )