from __future__ import annotations

from collections.abc import Sequence

from analyzer.safe_custom_object import SafeObjectConstructionBlueprint
from models.project_coverage_result import ProjectTestCandidate


class ProjectPytestGenerator:
    """Farklı modül/fonksiyon adaylarından tek, deterministik pytest üretir."""

    def generate(self, candidates: Sequence[ProjectTestCandidate]) -> str:
        values = tuple(candidates)
        if not values:
            raise ValueError("En az bir project test adayı gereklidir.")
        if any(not isinstance(value, ProjectTestCandidate) for value in values):
            raise TypeError("candidates yalnız ProjectTestCandidate içermelidir.")
        identifiers = tuple(value.project_test_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("project_test_id değerleri benzersiz olmalıdır.")

        lines = ['"""Otomatik oluşturulmuş birleşik project pytest paketi."""', ""]
        if any(value.scenario.expects_exception for value in values):
            lines.extend(("import pytest", ""))
        for index, candidate in enumerate(values, start=1):
            imported_target = (
                candidate.scenario.target_class_name
                or candidate.function_name
            )
            lines.append(
                f"from {candidate.module_path} import "
                f"{imported_target} as _project_target_{index:04d}"
            )
            object_class_names = tuple(
                dict.fromkeys(
                    (
                        *(
                            value.class_name
                            for _, value in candidate.scenario.keyword_arguments
                            if isinstance(value, SafeObjectConstructionBlueprint)
                        ),
                        *(
                            slot.blueprint.class_name
                            for slot in (
                                candidate.scenario.setup_plan.object_slots
                                if candidate.scenario.setup_plan is not None
                                else ()
                            )
                        ),
                    )
                )
            )
            for object_index, class_name in enumerate(
                object_class_names,
                start=1,
            ):
                lines.append(
                    f"from {candidate.module_path} import {class_name} "
                    f"as _project_object_class_{index:04d}_{object_index:03d}"
                )
        lines.extend(("", ""))

        for index, candidate in enumerate(values, start=1):
            scenario = candidate.scenario
            lines.extend(
                (
                    f"def test_project_candidate_{index:04d}() -> None:",
                    f"    {scenario.description!r}",
                    f"    # Project test kimliği: {candidate.project_test_id}",
                )
            )
            object_aliases = {
                class_name: f"_project_object_class_{index:04d}_{object_index:03d}"
                for object_index, class_name in enumerate(
                    dict.fromkeys(
                        (
                            *(
                                value.class_name
                                for _, value in scenario.keyword_arguments
                                if isinstance(value, SafeObjectConstructionBlueprint)
                            ),
                            *(
                                slot.blueprint.class_name
                                for slot in (
                                    scenario.setup_plan.object_slots
                                    if scenario.setup_plan is not None
                                    else ()
                                )
                            ),
                        )
                    ),
                    start=1,
                )
            }
            argument_expressions: dict[str, str] = {}
            if scenario.setup_plan is not None:
                setup_variables = {
                    slot.slot_id: (
                        f"setup_object_{index:04d}_{slot_index:03d}"
                    )
                    for slot_index, slot in enumerate(
                        scenario.setup_plan.object_slots,
                        start=1,
                    )
                }
                for slot in scenario.setup_plan.object_slots:
                    constructor_arguments = ", ".join(
                        f"{name}={value!r}"
                        for name, value in slot.blueprint.constructor_arguments
                    )
                    lines.append(
                        f"    {setup_variables[slot.slot_id]} = "
                        f"{object_aliases[slot.blueprint.class_name]}"
                        f"({constructor_arguments})"
                    )
                for setup_call in scenario.setup_plan.setup_calls:
                    rendered_arguments = ", ".join(
                        (
                            f"{argument.parameter_name}="
                            f"{setup_variables[argument.object_slot_id]}"
                            if argument.object_slot_id is not None
                            else f"{argument.parameter_name}={argument.value!r}"
                        )
                        for argument in setup_call.arguments
                    )
                    lines.append(
                        f"    {setup_variables[setup_call.receiver_slot_id]}."
                        f"{setup_call.method_summary.receiver.method_identity}"
                        f"({rendered_arguments})"
                    )
                argument_expressions.update(
                    {
                        binding.parameter_name: setup_variables[
                            binding.object_slot_id
                        ]
                        for binding in scenario.setup_plan.target_bindings
                    }
                )
            object_value_index = 0
            for name, value in scenario.keyword_arguments:
                if not isinstance(value, SafeObjectConstructionBlueprint):
                    continue
                object_value_index += 1
                variable_name = (
                    f"object_input_{index:04d}_{object_value_index:03d}"
                )
                constructor_arguments = ", ".join(
                    f"{argument_name}={argument_value!r}"
                    for argument_name, argument_value
                    in value.constructor_arguments
                )
                lines.append(
                    f"    {variable_name} = {object_aliases[value.class_name]}"
                    f"({constructor_arguments})"
                )
                argument_expressions[name] = variable_name
            invocation_arguments = list(scenario.keyword_arguments)
            supplied_names = {name for name, _ in invocation_arguments}
            if scenario.setup_plan is not None:
                invocation_arguments.extend(
                    (binding.parameter_name, None)
                    for binding in scenario.setup_plan.target_bindings
                    if binding.parameter_name not in supplied_names
                )
            arguments = ", ".join(
                (
                    f"{name}={argument_expressions[name]}"
                    if name in argument_expressions
                    else f"{name}={value!r}"
                )
                for name, value in invocation_arguments
            )
            if scenario.target_class_name is None:
                call = f"_project_target_{index:04d}({arguments})"
            else:
                receiver_variable = (
                    setup_variables[scenario.setup_plan.receiver_slot_id]
                    if scenario.setup_plan is not None
                    and scenario.setup_plan.receiver_slot_id is not None
                    else None
                )
                if receiver_variable is None:
                    constructor_arguments = ", ".join(
                        f"{name}={value!r}"
                        for name, value in scenario.constructor_arguments
                    )
                    lines.append(
                        f"    target = _project_target_{index:04d}"
                        f"({constructor_arguments})"
                    )
                else:
                    lines.append(f"    target = {receiver_variable}")
                call = f"target.{candidate.function_name}({arguments})"
            if scenario.expects_exception:
                if scenario.expected_exception is None:
                    raise ValueError("Beklenen exception adı belirtilmelidir.")
                lines.extend(
                    (
                        "",
                        f"    with pytest.raises({scenario.expected_exception}):",
                        f"        {call}",
                    )
                )
            else:
                lines.extend(
                    (
                        "",
                        f"    result = {call}",
                        f"    assert result == {scenario.expected_result!r}",
                    )
                )
            if index < len(values):
                lines.extend(("", ""))
        return "\n".join(lines).rstrip() + "\n"
