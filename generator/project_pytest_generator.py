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
                    value.class_name
                    for _, value in candidate.scenario.keyword_arguments
                    if isinstance(value, SafeObjectConstructionBlueprint)
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
                        value.class_name
                        for _, value in scenario.keyword_arguments
                        if isinstance(value, SafeObjectConstructionBlueprint)
                    ),
                    start=1,
                )
            }
            argument_expressions: dict[str, str] = {}
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
            arguments = ", ".join(
                (
                    f"{name}={argument_expressions[name]}"
                    if name in argument_expressions
                    else f"{name}={value!r}"
                )
                for name, value in scenario.keyword_arguments
            )
            if scenario.target_class_name is None:
                call = f"_project_target_{index:04d}({arguments})"
            else:
                constructor_arguments = ", ".join(
                    f"{name}={value!r}"
                    for name, value in scenario.constructor_arguments
                )
                lines.append(
                    f"    target = _project_target_{index:04d}"
                    f"({constructor_arguments})"
                )
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
