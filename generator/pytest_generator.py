from __future__ import annotations

import keyword
from collections.abc import Sequence
from typing import Any

from analyzer.safe_custom_object import (
    MAX_SAFE_OBJECTS_PER_SCENARIO,
    SafeObjectConstructionBlueprint,
)
from generator.safe_method_setup_plan import SafeObjectSetupPlan
from generator.scenario_generator import Scenario


class PytestGenerator:
    """
    Test senaryolarından çalıştırılabilir pytest kaynak kodu üretir.

    Scenario içerisinde bulunan fonksiyon girdilerini, beklenen sonucu
    veya beklenen exception bilgisini kullanarak gerçek test
    fonksiyonları oluşturur.
    """

    def generate(
        self,
        module_path: str,
        function_name: str,
        scenarios: Sequence[Scenario],
    ) -> str:
        """
        Verilen test senaryoları için pytest kaynak kodu üretir.

        Args:
            module_path:
                Test edilecek fonksiyonun bulunduğu Python modül yolu.
                Örnek: ``datasets.sample_code``.

            function_name:
                Test edilecek fonksiyonun adı.

            scenarios:
                DQM tabanlı ve somut girdiler içeren test senaryoları.

        Returns:
            Geçerli ve çalıştırılabilir pytest kaynak kodu.

        Raises:
            TypeError:
                Senaryo koleksiyonunda geçersiz tür bulunduğunda.

            ValueError:
                Modül yolu, fonksiyon adı veya senaryolar geçersiz
                olduğunda.
        """
        normalized_module_path = module_path.strip()
        normalized_function_name = function_name.strip()

        self._validate_input(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=scenarios,
        )

        requires_pytest_import = any(
            scenario.expects_exception
            for scenario in scenarios
        )
        target_class_name = self._target_class_name(scenarios)
        custom_object_class_names = {
            value.class_name
            for scenario in scenarios
            for _, value in scenario.keyword_arguments
            if isinstance(value, SafeObjectConstructionBlueprint)
        }
        custom_object_class_names.update(
            slot.blueprint.class_name
            for scenario in scenarios
            if scenario.setup_plan is not None
            for slot in scenario.setup_plan.object_slots
        )

        code_lines = self._create_header(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            target_class_name=target_class_name,
            custom_object_class_names=custom_object_class_names,
            requires_pytest_import=requires_pytest_import,
        )

        generated_test_names: set[str] = set()

        for index, scenario in enumerate(scenarios):
            test_name = self._create_test_name(
                function_name=normalized_function_name,
                scenario=scenario,
            )

            if test_name in generated_test_names:
                raise ValueError(
                    "Aynı pytest fonksiyon adı birden fazla kez "
                    f"üretilemez: {test_name}"
                )

            generated_test_names.add(test_name)

            code_lines.extend(
                self._create_test_function(
                    function_name=normalized_function_name,
                    scenario=scenario,
                    test_name=test_name,
                    target_class_name=target_class_name,
                )
            )

            if index < len(scenarios) - 1:
                code_lines.extend(["", ""])

        return "\n".join(code_lines).rstrip() + "\n"

    @staticmethod
    def _create_header(
        module_path: str,
        function_name: str,
        target_class_name: str | None,
        custom_object_class_names: set[str],
        requires_pytest_import: bool,
    ) -> list[str]:
        """
        Üretilen test dosyasının import ve açıklama bölümünü oluşturur.
        """
        code_lines: list[str] = [
            '"""Otomatik oluşturulmuş pytest testleri."""',
            "",
        ]

        if requires_pytest_import:
            code_lines.extend(
                [
                    "import pytest",
                    "",
                ]
            )

        imported_targets = sorted(
            {
                target_class_name or function_name,
                *custom_object_class_names,
            }
        )
        code_lines.extend(
            [
                f"from {module_path} import {', '.join(imported_targets)}",
                "",
                "",
            ]
        )

        return code_lines

    @classmethod
    def _validate_input(
        cls,
        module_path: str,
        function_name: str,
        scenarios: Sequence[Scenario],
    ) -> None:
        """
        Pytest üretim girdilerini doğrular.
        """
        if not module_path:
            raise ValueError(
                "Modül yolu boş olamaz."
            )

        if not function_name:
            raise ValueError(
                "Fonksiyon adı boş olamaz."
            )

        if not scenarios:
            raise ValueError(
                "En az bir test senaryosu gereklidir."
            )

        if not cls._is_valid_module_path(module_path):
            raise ValueError(
                f"Geçersiz Python modül yolu: {module_path}"
            )

        if not cls._is_valid_identifier(function_name):
            raise ValueError(
                f"Geçersiz Python fonksiyon adı: "
                f"{function_name}"
            )

        for scenario in scenarios:
            cls._validate_scenario(scenario)

    @classmethod
    def _validate_scenario(
        cls,
        scenario: Scenario,
    ) -> None:
        """
        Tek bir test senaryosunun geçerliliğini doğrular.
        """
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "Bütün senaryolar Scenario türünde olmalıdır."
            )

        if not scenario.scenario_id.strip():
            raise ValueError(
                "Senaryo kimliği boş olamaz."
            )

        if scenario.path_index < 1:
            raise ValueError(
                "Yürütme yolu numarası 1 veya daha büyük "
                "olmalıdır."
            )

        if scenario.priority_rank < 1:
            raise ValueError(
                "Öncelik sırası 1 veya daha büyük olmalıdır."
            )

        argument_names = [
            argument_name
            for argument_name, _ in scenario.keyword_arguments
        ]

        if any(
            not cls._is_valid_identifier(argument_name)
            for argument_name in argument_names
        ):
            raise ValueError(
                "Senaryo geçersiz fonksiyon parametre adı "
                "içeriyor."
            )

        if len(set(argument_names)) != len(argument_names):
            raise ValueError(
                "Senaryo tekrar eden fonksiyon parametreleri "
                "içeremez."
            )

        constructor_names = [
            argument_name
            for argument_name, _ in scenario.constructor_arguments
        ]
        if any(
            not cls._is_valid_identifier(argument_name)
            for argument_name in constructor_names
        ):
            raise ValueError("Senaryo geçersiz constructor parametresi içeriyor.")
        if len(set(constructor_names)) != len(constructor_names):
            raise ValueError("Constructor parametreleri tekrar edemez.")
        if scenario.target_class_name is not None and not cls._is_valid_identifier(
            scenario.target_class_name
        ):
            raise ValueError("Senaryo target class adı geçersizdir.")
        if scenario.target_class_name is None and scenario.constructor_arguments:
            raise ValueError("Constructor arguments yalnız method hedefinde kullanılabilir.")
        blueprint_count = sum(
            isinstance(value, SafeObjectConstructionBlueprint)
            for _, value in scenario.keyword_arguments
        )
        if blueprint_count > MAX_SAFE_OBJECTS_PER_SCENARIO:
            raise ValueError("Scenario custom object limitini aşıyor.")
        if scenario.setup_plan is not None:
            if not isinstance(scenario.setup_plan, SafeObjectSetupPlan):
                raise TypeError("Scenario setup_plan türü geçersizdir.")
            if len(scenario.setup_plan.object_slots) > MAX_SAFE_OBJECTS_PER_SCENARIO:
                raise ValueError("Scenario setup object limitini aşıyor.")

        if (
            scenario.expected_exception is not None
            and not cls._is_valid_identifier(
                scenario.expected_exception
            )
        ):
            raise ValueError(
                "Beklenen exception adı geçersizdir."
            )

    @classmethod
    def _is_valid_module_path(
        cls,
        module_path: str,
    ) -> bool:
        """
        Noktayla ayrılmış Python modül yolunu doğrular.
        """
        parts = module_path.split(".")

        return bool(parts) and all(
            cls._is_valid_identifier(part)
            for part in parts
        )

    @staticmethod
    def _is_valid_identifier(
        value: str,
    ) -> bool:
        """
        Değerin geçerli Python tanımlayıcısı olup olmadığını kontrol eder.
        """
        return (
            isinstance(value, str)
            and value.isidentifier()
            and not keyword.iskeyword(value)
        )

    def _create_test_function(
        self,
        function_name: str,
        scenario: Scenario,
        test_name: str,
        target_class_name: str | None,
    ) -> list[str]:
        """
        Tek bir Scenario için gerçek pytest fonksiyonu oluşturur.
        """
        description_literal = repr(
            scenario.description
        )

        code_lines: list[str] = [
            f"def {test_name}() -> None:",
            f"    {description_literal}",
            (
                f"    # Senaryo kimliği: "
                f"{scenario.scenario_id}"
            ),
            (
                f"    # DQM önceliği: "
                f"{scenario.priority_level}"
            ),
            (
                f"    # CFG düğüm yolu: "
                f"{list(scenario.node_ids)}"
            ),
            (
                f"    # CFG kenarları: "
                f"{list(scenario.edge_labels)}"
            ),
        ]
        argument_expressions: dict[str, str] = {}
        if scenario.setup_plan is not None:
            plan_lines, plan_arguments = self._create_setup_plan_lines(
                scenario.setup_plan,
            )
            code_lines.extend(plan_lines)
            argument_expressions.update(plan_arguments)
        for argument_name, argument_value in scenario.keyword_arguments:
            if not isinstance(argument_value, SafeObjectConstructionBlueprint):
                continue
            variable_name = f"{argument_name}_object"
            constructor_call = self._create_function_call(
                function_name=argument_value.class_name,
                keyword_arguments=argument_value.constructor_arguments,
            )
            code_lines.append(f"    {variable_name} = {constructor_call}")
            argument_expressions[argument_name] = variable_name

        invocation_arguments = list(scenario.keyword_arguments)
        supplied_argument_names = {name for name, _ in invocation_arguments}
        if scenario.setup_plan is not None:
            invocation_arguments.extend(
                (binding.parameter_name, None)
                for binding in scenario.setup_plan.target_bindings
                if binding.parameter_name not in supplied_argument_names
            )

        if target_class_name is None:
            function_call = self._create_function_call(
                function_name=function_name,
                keyword_arguments=tuple(invocation_arguments),
                argument_expressions=argument_expressions,
            )
        else:
            receiver_expression = (
                f"setup_object_{scenario.setup_plan.receiver_slot_id}"
                if scenario.setup_plan is not None
                and scenario.setup_plan.receiver_slot_id is not None
                else None
            )
            if receiver_expression is None:
                constructor_call = self._create_function_call(
                    function_name=target_class_name,
                    keyword_arguments=scenario.constructor_arguments,
                )
                code_lines.append(f"    target = {constructor_call}")
            else:
                code_lines.append(f"    target = {receiver_expression}")
            function_call = self._create_function_call(
                function_name=f"target.{function_name}",
                keyword_arguments=tuple(invocation_arguments),
                argument_expressions=argument_expressions,
            )

        if scenario.expects_exception:
            code_lines.extend(
                self._create_exception_assertion(
                    function_call=function_call,
                    expected_exception=(
                        scenario.expected_exception
                    ),
                )
            )
        else:
            code_lines.extend(
                self._create_result_assertion(
                    function_call=function_call,
                    expected_result=scenario.expected_result,
                )
            )

        return code_lines

    def _create_setup_plan_lines(
        self,
        plan: SafeObjectSetupPlan,
    ) -> tuple[list[str], dict[str, str]]:
        """Constructor ve setup çağrılarını kanıtlanmış sırayla render eder."""
        slot_variables = {
            slot.slot_id: f"setup_object_{slot.slot_id}"
            for slot in plan.object_slots
        }
        lines: list[str] = []
        for slot in plan.object_slots:
            constructor = self._create_function_call(
                function_name=slot.blueprint.class_name,
                keyword_arguments=slot.blueprint.constructor_arguments,
            )
            lines.append(f"    {slot_variables[slot.slot_id]} = {constructor}")
        for call in plan.setup_calls:
            values: list[tuple[str, Any]] = []
            expressions: dict[str, str] = {}
            for argument in call.arguments:
                values.append((argument.parameter_name, argument.value))
                if argument.object_slot_id is not None:
                    expressions[argument.parameter_name] = slot_variables[
                        argument.object_slot_id
                    ]
            rendered = self._create_function_call(
                function_name=(
                    f"{slot_variables[call.receiver_slot_id]}."
                    f"{call.method_summary.receiver.method_identity}"
                ),
                keyword_arguments=tuple(values),
                argument_expressions=expressions,
            )
            lines.append(f"    {rendered}")
        return (
            lines,
            {
                binding.parameter_name: slot_variables[binding.object_slot_id]
                for binding in plan.target_bindings
            },
        )

    @classmethod
    def _target_class_name(
        cls,
        scenarios: Sequence[Scenario],
    ) -> str | None:
        values = {scenario.target_class_name for scenario in scenarios}
        if len(values) != 1:
            raise ValueError("Senaryolar aynı callable target'ı kullanmalıdır.")
        value = next(iter(values))
        if value is not None and not cls._is_valid_identifier(value):
            raise ValueError("Target class adı geçersizdir.")
        return value

    @staticmethod
    def _create_function_call(
        function_name: str,
        keyword_arguments: tuple[
            tuple[str, Any],
            ...,
        ],
        argument_expressions: dict[str, str] | None = None,
    ) -> str:
        """
        Scenario parametrelerinden fonksiyon çağrısı oluşturur.

        Örnek:
            ``calculate_score(score=50)``
        """
        rendered_arguments = ", ".join(
            (
                f"{argument_name}={argument_expressions[argument_name]}"
                if argument_expressions is not None
                and argument_name in argument_expressions
                else f"{argument_name}={argument_value!r}"
            )
            for argument_name, argument_value
            in keyword_arguments
        )

        return (
            f"{function_name}({rendered_arguments})"
        )

    @staticmethod
    def _create_result_assertion(
        function_call: str,
        expected_result: Any,
    ) -> list[str]:
        """
        Normal dönüş değeri bekleyen assertion satırlarını oluşturur.
        """
        return [
            "",
            f"    result = {function_call}",
            (
                f"    assert result == "
                f"{expected_result!r}"
            ),
        ]

    @staticmethod
    def _create_exception_assertion(
        function_call: str,
        expected_exception: str | None,
    ) -> list[str]:
        """
        Exception bekleyen pytest kontrolünü oluşturur.
        """
        if expected_exception is None:
            raise ValueError(
                "Beklenen exception adı belirtilmelidir."
            )

        return [
            "",
            (
                f"    with pytest.raises("
                f"{expected_exception}"
                f"):"
            ),
            f"        {function_call}",
        ]

    @staticmethod
    def _create_test_name(
        function_name: str,
        scenario: Scenario,
    ) -> str:
        """
        Kararlı ve benzersiz pytest fonksiyon adı oluşturur.
        """
        return (
            f"test_{function_name}_"
            f"path_{scenario.path_index}_"
            f"priority_{scenario.priority_rank}"
        )
