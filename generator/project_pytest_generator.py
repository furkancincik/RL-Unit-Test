from __future__ import annotations

from collections.abc import Sequence

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
            arguments = ", ".join(
                f"{name}={value!r}" for name, value in scenario.keyword_arguments
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
