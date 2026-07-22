from __future__ import annotations

import keyword
from collections.abc import Sequence

from generator.scenario_generator import TestScenario


class PytestGenerator:
    """Test senaryolarından geçerli pytest kaynak kodu üretir."""

    def generate(
        self,
        module_path: str,
        function_name: str,
        scenarios: Sequence[TestScenario],
    ) -> str:
        """
        Verilen test senaryoları için pytest kaynak kodu üretir.

        Args:
            module_path: Test edilecek fonksiyonun bulunduğu Python
                modül yolu. Örnek: ``datasets.sample_code``.
            function_name: Test edilecek fonksiyonun adı.
            scenarios: DQM tabanlı test senaryoları.

        Returns:
            Geçerli Python sözdizimine sahip pytest kaynak kodu.

        Raises:
            TypeError: Senaryo koleksiyonunda geçersiz tür bulunduğunda.
            ValueError: Modül yolu, fonksiyon adı veya senaryolar
                geçersiz olduğunda.
        """
        normalized_module_path = module_path.strip()
        normalized_function_name = function_name.strip()

        self._validate_input(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=scenarios,
        )

        code_lines: list[str] = [
            '"""Otomatik oluşturulmuş pytest test taslakları."""',
            "",
            (
                f"from {normalized_module_path} "
                f"import {normalized_function_name}"
            ),
            "",
            "",
        ]

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
                )
            )

            if index < len(scenarios) - 1:
                code_lines.extend(["", ""])

        return "\n".join(code_lines).rstrip() + "\n"

    @classmethod
    def _validate_input(
        cls,
        module_path: str,
        function_name: str,
        scenarios: Sequence[TestScenario],
    ) -> None:
        """
        Pytest üretim girdilerini doğrular.

        Args:
            module_path: Normalize edilmiş Python modül yolu.
            function_name: Normalize edilmiş fonksiyon adı.
            scenarios: Test senaryoları.

        Raises:
            TypeError: Senaryo türlerinden biri geçersiz olduğunda.
            ValueError: Girdilerden biri geçersiz olduğunda.
        """
        if not module_path:
            raise ValueError("Modül yolu boş olamaz.")

        if not function_name:
            raise ValueError("Fonksiyon adı boş olamaz.")

        if not scenarios:
            raise ValueError("En az bir test senaryosu gereklidir.")

        if not cls._is_valid_module_path(module_path):
            raise ValueError(
                f"Geçersiz Python modül yolu: {module_path}"
            )

        if not cls._is_valid_identifier(function_name):
            raise ValueError(
                f"Geçersiz Python fonksiyon adı: {function_name}"
            )

        for scenario in scenarios:
            cls._validate_scenario(scenario)

    @classmethod
    def _validate_scenario(
        cls,
        scenario: TestScenario,
    ) -> None:
        """
        Tek bir test senaryosunun geçerliliğini doğrular.

        Args:
            scenario: Doğrulanacak test senaryosu.

        Raises:
            TypeError: Nesne TestScenario türünde değilse.
            ValueError: Senaryonun alanları geçersizse.
        """
        if not isinstance(scenario, TestScenario):
            raise TypeError(
                "Bütün senaryolar TestScenario türünde olmalıdır."
            )

        if not scenario.scenario_id.strip():
            raise ValueError("Senaryo kimliği boş olamaz.")

        if scenario.path_index < 1:
            raise ValueError(
                "Yürütme yolu numarası 1 veya daha büyük olmalıdır."
            )

        if scenario.priority_rank < 1:
            raise ValueError(
                "Öncelik sırası 1 veya daha büyük olmalıdır."
            )

    @classmethod
    def _is_valid_module_path(
        cls,
        module_path: str,
    ) -> bool:
        """
        Noktayla ayrılmış Python modül yolunu doğrular.

        Args:
            module_path: Doğrulanacak modül yolu.

        Returns:
            Bütün parçalar geçerli Python tanımlayıcılarıysa True.
        """
        parts = module_path.split(".")

        return bool(parts) and all(
            cls._is_valid_identifier(part)
            for part in parts
        )

    @staticmethod
    def _is_valid_identifier(value: str) -> bool:
        """
        Bir değerin geçerli Python tanımlayıcısı olduğunu doğrular.

        Args:
            value: Kontrol edilecek değer.

        Returns:
            Değer geçerli bir tanımlayıcı ve anahtar kelime değilse True.
        """
        return value.isidentifier() and not keyword.iskeyword(value)

    def _create_test_function(
        self,
        function_name: str,
        scenario: TestScenario,
        test_name: str,
    ) -> list[str]:
        """
        Tek bir senaryo için pytest test fonksiyonu oluşturur.

        Args:
            function_name: Test edilecek fonksiyonun adı.
            scenario: Test fonksiyonu oluşturulacak senaryo.
            test_name: Oluşturulan pytest fonksiyonunun adı.

        Returns:
            Test fonksiyonunu oluşturan kaynak kod satırları.
        """
        description_literal = repr(scenario.description)

        return [
            f"def {test_name}() -> None:",
            f"    {description_literal}",
            f"    # Senaryo kimliği: {scenario.scenario_id}",
            f"    # DQM önceliği: {scenario.priority_level}",
            f"    # Normalize DQM skoru: {scenario.dqm_score}",
            f"    # CFG düğüm yolu: {list(scenario.node_ids)}",
            f"    # CFG kenarları: {list(scenario.edge_labels)}",
            f"    _target_function = {function_name}",
            "",
            "    assert callable(_target_function)",
            "    # TODO: Senaryoya uygun girdiler ve doğrulamalar üretilecek.",
        ]

    @staticmethod
    def _create_test_name(
        function_name: str,
        scenario: TestScenario,
    ) -> str:
        """
        Senaryo için kararlı ve geçerli pytest fonksiyon adı oluşturur.

        Args:
            function_name: Test edilen fonksiyonun adı.
            scenario: Test senaryosu.

        Returns:
            ``test_`` ile başlayan pytest fonksiyon adı.
        """
        return (
            f"test_{function_name}_"
            f"path_{scenario.path_index}_"
            f"priority_{scenario.priority_rank}"
        )