from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

from analyzer.safe_custom_object import SafeObjectConstructionBlueprint
from analyzer.safe_method_summary import analyze_safe_method_summaries
from generator.safe_method_setup_plan import (
    SafeObjectSetupPlan,
    SafeObjectSlot,
    SafeSetupCall,
    SafeTargetBinding,
    SafeTypedArgumentBinding,
)
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


def create_setup_plan(
    module_identity: str,
    *,
    target_identity: str,
    receiver_slot_id: str | None = None,
) -> SafeObjectSetupPlan:
    summary = analyze_safe_method_summaries(
        ast.parse(
            """
class Counter:
    def __init__(self):
        self.value = 0

    def increase(self, amount: int) -> None:
        self.value += amount
"""
        ),
        module_identity=module_identity,
    ).summary_for("Counter.increase")
    return SafeObjectSetupPlan(
        module_identity=module_identity,
        target_identity=target_identity,
        object_slots=(
            SafeObjectSlot(
                slot_id="root",
                blueprint=SafeObjectConstructionBlueprint(
                    module_identity=module_identity,
                    class_name="Counter",
                    class_fingerprint=summary.receiver.class_fingerprint,
                ),
            ),
        ),
        setup_calls=(
            SafeSetupCall(
                receiver_slot_id="root",
                method_summary=summary,
                arguments=(
                    SafeTypedArgumentBinding(
                        parameter_name="amount",
                        type_name="int",
                        value=1,
                    ),
                ),
            ),
        ),
        target_bindings=(
            ()
            if receiver_slot_id is not None
            else (
                SafeTargetBinding(
                    parameter_name="subject",
                    object_slot_id="root",
                ),
            )
        ),
        receiver_slot_id=receiver_slot_id,
    )


def create_setup_source_file(
    tmp_path: Path,
    *,
    module_identity: str,
    setup_raises: bool = False,
) -> Path:
    setup_body = (
        'raise ValueError("setup failed")'
        if setup_raises
        else "self.value += amount"
    )
    source_file = tmp_path / f"{module_identity}.py"
    source_file.write_text(
        f"""
class Counter:
    def __init__(self):
        self.value = 0

    def increase(self, amount: int) -> None:
        {setup_body}

    def inspect_self(self) -> int:
        return self.value


def inspect(subject) -> int:
    return subject.value


def fail(subject) -> None:
    raise ValueError("target failed")
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


def test_filter_materializes_a_fresh_setup_instance_for_each_scenario(
    tmp_path: Path,
) -> None:
    module_identity = "runtime_setup_fresh"
    source_file = create_setup_source_file(
        tmp_path,
        module_identity=module_identity,
    )
    plan = create_setup_plan(
        module_identity,
        target_identity="inspect",
    )
    first = replace(
        create_scenario(
            scenario_id="fresh-first",
            keyword_arguments=(),
            expected_result=1,
        ),
        setup_plan=plan,
    )
    second = replace(
        create_scenario(
            scenario_id="fresh-second",
            keyword_arguments=(),
            expected_result=1,
        ),
        setup_plan=plan,
    )

    result = RealRLTrainingService()._filter_executable_scenarios(
        source_file=source_file,
        module_path=module_identity,
        function_name="inspect",
        scenarios=(first, second),
        import_root=tmp_path,
    )

    assert result == (first, second)


def test_filter_uses_materialized_setup_instance_as_actual_method_receiver(
    tmp_path: Path,
) -> None:
    module_identity = "runtime_method_receiver"
    source_file = create_setup_source_file(
        tmp_path,
        module_identity=module_identity,
    )
    plan = create_setup_plan(
        module_identity,
        target_identity="Counter.inspect_self",
        receiver_slot_id="root",
    )
    scenario = replace(
        create_scenario(
            scenario_id="method-receiver",
            keyword_arguments=(),
            expected_result=1,
        ),
        setup_plan=plan,
        constructor_arguments=(),
        target_class_name="Counter",
    )

    result = RealRLTrainingService()._filter_executable_scenarios(
        source_file=source_file,
        module_path=module_identity,
        function_name="inspect_self",
        scenarios=(scenario,),
        import_root=tmp_path,
    )

    assert result == (scenario,)


def test_filter_keeps_target_exception_after_successful_setup(
    tmp_path: Path,
) -> None:
    module_identity = "runtime_setup_target_failure"
    source_file = create_setup_source_file(
        tmp_path,
        module_identity=module_identity,
    )
    plan = create_setup_plan(
        module_identity,
        target_identity="fail",
    )
    scenario = replace(
        create_scenario(
            scenario_id="target-expected-exception",
            keyword_arguments=(),
            expected_exception="ValueError",
        ),
        setup_plan=plan,
    )

    result = RealRLTrainingService()._filter_executable_scenarios(
        source_file=source_file,
        module_path=module_identity,
        function_name="fail",
        scenarios=(scenario,),
        import_root=tmp_path,
    )

    assert result == (scenario,)


def test_filter_does_not_count_setup_exception_as_target_expected_exception(
    tmp_path: Path,
) -> None:
    module_identity = "runtime_setup_stage_failure"
    source_file = create_setup_source_file(
        tmp_path,
        module_identity=module_identity,
        setup_raises=True,
    )
    plan = create_setup_plan(
        module_identity,
        target_identity="fail",
    )
    scenario = replace(
        create_scenario(
            scenario_id="setup-is-not-target",
            keyword_arguments=(),
            expected_exception="ValueError",
        ),
        setup_plan=plan,
    )

    result = RealRLTrainingService()._filter_executable_scenarios(
        source_file=source_file,
        module_path=module_identity,
        function_name="fail",
        scenarios=(scenario,),
        import_root=tmp_path,
    )

    assert result == ()


def test_filter_reports_setup_failure_separately_from_target_mismatch(
    tmp_path: Path,
) -> None:
    module_identity = "runtime_setup_rejection_category"
    source_file = create_setup_source_file(
        tmp_path,
        module_identity=module_identity,
        setup_raises=True,
    )
    setup_failure = replace(
        create_scenario(
            scenario_id="setup-failure",
            keyword_arguments=(),
            expected_result=1,
        ),
        setup_plan=create_setup_plan(
            module_identity,
            target_identity="inspect",
        ),
    )
    mismatch = create_scenario(
        scenario_id="target-mismatch",
        keyword_arguments=(("value", 1),),
        expected_result="impossible",
    )

    scenarios, rejection_counts = (
        RealRLTrainingService()._filter_executable_scenarios_with_rejections(
            source_file=source_file,
            module_path=module_identity,
            function_name="inspect",
            scenarios=(setup_failure,),
            import_root=tmp_path,
        )
    )

    assert scenarios == ()
    assert rejection_counts == (("SETUP_EXECUTION_REJECTED", 1),)

    target = lambda *, value: value
    outcome = RealRLTrainingService._scenario_validation_outcome(
        target_function=target,
        scenario=mismatch,
    )
    assert outcome.category == "EXECUTION_MISMATCH"
    assert outcome.matched is False
