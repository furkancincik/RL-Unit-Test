from __future__ import annotations

from pathlib import Path

from generator.scenario_generator import Scenario
from services.real_rl_training_service import RealRLTrainingService


def create_scenario(
    *,
    scenario_id: str,
    keyword_arguments: tuple[tuple[str, object], ...],
    expected_result: object = None,
    expected_exception: str | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        name=scenario_id,
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 2),
        edge_labels=(None,),
        contains_loop=False,
        contains_exception=(
            expected_exception is not None
        ),
        description="Concrete validation testi.",
        keyword_arguments=keyword_arguments,
        expected_result=expected_result,
        expected_exception=expected_exception,
    )


def create_source_file(
    tmp_path: Path,
) -> Path:
    source_file = tmp_path / "validation_sample.py"
    source_file.write_text(
        """
def classify(value: int) -> str:
    if value < 0:
        return "negative"

    return "non-negative"


def fail(value: int) -> None:
    if value == 0:
        raise ValueError("zero")
""".strip(),
        encoding="utf-8",
    )
    return source_file


def test_filter_keeps_matching_scenario(
    tmp_path: Path,
) -> None:
    service = RealRLTrainingService()
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="matching",
        keyword_arguments=(("value", -1),),
        expected_result="negative",
    )

    result = service._filter_executable_scenarios(
        source_file=source_file,
        function_name="classify",
        scenarios=(scenario,),
    )

    assert result == (scenario,)


def test_filter_removes_mismatching_scenario(
    tmp_path: Path,
) -> None:
    service = RealRLTrainingService()
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="mismatching",
        keyword_arguments=(("value", 1),),
        expected_result="negative",
    )

    result = service._filter_executable_scenarios(
        source_file=source_file,
        function_name="classify",
        scenarios=(scenario,),
    )

    assert result == ()


def test_filter_keeps_expected_exception(
    tmp_path: Path,
) -> None:
    service = RealRLTrainingService()
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="expected-exception",
        keyword_arguments=(("value", 0),),
        expected_exception="ValueError",
    )

    result = service._filter_executable_scenarios(
        source_file=source_file,
        function_name="fail",
        scenarios=(scenario,),
    )

    assert result == (scenario,)


def test_filter_removes_unexpected_exception(
    tmp_path: Path,
) -> None:
    service = RealRLTrainingService()
    source_file = create_source_file(tmp_path)

    scenario = create_scenario(
        scenario_id="unexpected-exception",
        keyword_arguments=(("value", 0),),
        expected_result=None,
    )

    result = service._filter_executable_scenarios(
        source_file=source_file,
        function_name="fail",
        scenarios=(scenario,),
    )

    assert result == ()