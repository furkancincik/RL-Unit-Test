from __future__ import annotations

import keyword
from collections.abc import Sequence

from generator.scenario_generator import Scenario


class PytestGenerator:
    """Test senaryolarÄ±ndan geÃ§erli pytest kaynak kodu Ã¼retir."""

    def generate(
        self,
        module_path: str,
        function_name: str,
        scenarios: Sequence[Scenario],
    ) -> str:
        """
        Verilen test senaryolarÄ± iÃ§in pytest kaynak kodu Ã¼retir.

        Args:
            module_path: Test edilecek fonksiyonun bulunduÄŸu Python
                modÃ¼l yolu. Ã–rnek: ``datasets.sample_code``.
            function_name: Test edilecek fonksiyonun adÄ±.
            scenarios: DQM tabanlÄ± test senaryolarÄ±.

        Returns:
            GeÃ§erli Python sÃ¶zdizimine sahip pytest kaynak kodu.

        Raises:
            TypeError: Senaryo koleksiyonunda geÃ§ersiz tÃ¼r bulunduÄŸunda.
            ValueError: ModÃ¼l yolu, fonksiyon adÄ± veya senaryolar
                geÃ§ersiz olduÄŸunda.
        """
        normalized_module_path = module_path.strip()
        normalized_function_name = function_name.strip()

        self._validate_input(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=scenarios,
        )

        code_lines: list[str] = [
            '"""Otomatik oluÅŸturulmuÅŸ pytest test taslaklarÄ±."""',
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
                    "AynÄ± pytest fonksiyon adÄ± birden fazla kez "
                    f"Ã¼retilemez: {test_name}"
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
        scenarios: Sequence[Scenario],
    ) -> None:
        """
        Pytest Ã¼retim girdilerini doÄŸrular.

        Args:
            module_path: Normalize edilmiÅŸ Python modÃ¼l yolu.
            function_name: Normalize edilmiÅŸ fonksiyon adÄ±.
            scenarios: Test senaryolarÄ±.

        Raises:
            TypeError: Senaryo tÃ¼rlerinden biri geÃ§ersiz olduÄŸunda.
            ValueError: Girdilerden biri geÃ§ersiz olduÄŸunda.
        """
        if not module_path:
            raise ValueError("ModÃ¼l yolu boÅŸ olamaz.")

        if not function_name:
            raise ValueError("Fonksiyon adÄ± boÅŸ olamaz.")

        if not scenarios:
            raise ValueError("En az bir test senaryosu gereklidir.")

        if not cls._is_valid_module_path(module_path):
            raise ValueError(
                f"GeÃ§ersiz Python modÃ¼l yolu: {module_path}"
            )

        if not cls._is_valid_identifier(function_name):
            raise ValueError(
                f"GeÃ§ersiz Python fonksiyon adÄ±: {function_name}"
            )

        for scenario in scenarios:
            cls._validate_scenario(scenario)

    @classmethod
    def _validate_scenario(
        cls,
        scenario: Scenario,
    ) -> None:
        """
        Tek bir test senaryosunun geÃ§erliliÄŸini doÄŸrular.

        Args:
            scenario: DoÄŸrulanacak test senaryosu.

        Raises:
            TypeError: Nesne Scenario tÃ¼rÃ¼nde deÄŸilse.
            ValueError: Senaryonun alanlarÄ± geÃ§ersizse.
        """
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "BÃ¼tÃ¼n senaryolar Scenario tÃ¼rÃ¼nde olmalÄ±dÄ±r."
            )

        if not scenario.scenario_id.strip():
            raise ValueError("Senaryo kimliÄŸi boÅŸ olamaz.")

        if scenario.path_index < 1:
            raise ValueError(
                "YÃ¼rÃ¼tme yolu numarasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r."
            )

        if scenario.priority_rank < 1:
            raise ValueError(
                "Ã–ncelik sÄ±rasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r."
            )

    @classmethod
    def _is_valid_module_path(
        cls,
        module_path: str,
    ) -> bool:
        """
        Noktayla ayrÄ±lmÄ±ÅŸ Python modÃ¼l yolunu doÄŸrular.

        Args:
            module_path: DoÄŸrulanacak modÃ¼l yolu.

        Returns:
            BÃ¼tÃ¼n parÃ§alar geÃ§erli Python tanÄ±mlayÄ±cÄ±larÄ±ysa True.
        """
        parts = module_path.split(".")

        return bool(parts) and all(
            cls._is_valid_identifier(part)
            for part in parts
        )

    @staticmethod
    def _is_valid_identifier(value: str) -> bool:
        """
        Bir deÄŸerin geÃ§erli Python tanÄ±mlayÄ±cÄ±sÄ± olduÄŸunu doÄŸrular.

        Args:
            value: Kontrol edilecek deÄŸer.

        Returns:
            DeÄŸer geÃ§erli bir tanÄ±mlayÄ±cÄ± ve anahtar kelime deÄŸilse True.
        """
        return value.isidentifier() and not keyword.iskeyword(value)

    def _create_test_function(
        self,
        function_name: str,
        scenario: Scenario,
        test_name: str,
    ) -> list[str]:
        """
        Tek bir senaryo iÃ§in pytest test fonksiyonu oluÅŸturur.

        Args:
            function_name: Test edilecek fonksiyonun adÄ±.
            scenario: Test fonksiyonu oluÅŸturulacak senaryo.
            test_name: OluÅŸturulan pytest fonksiyonunun adÄ±.

        Returns:
            Test fonksiyonunu oluÅŸturan kaynak kod satÄ±rlarÄ±.
        """
        description_literal = repr(scenario.description)

        return [
            f"def {test_name}() -> None:",
            f"    {description_literal}",
            f"    # Senaryo kimliÄŸi: {scenario.scenario_id}",
            f"    # DQM Ã¶nceliÄŸi: {scenario.priority_level}",
            f"    # Normalize DQM skoru: {scenario.dqm_score}",
            f"    # CFG dÃ¼ÄŸÃ¼m yolu: {list(scenario.node_ids)}",
            f"    # CFG kenarlarÄ±: {list(scenario.edge_labels)}",
            f"    _target_function = {function_name}",
            "",
            "    assert callable(_target_function)",
            "    # TODO: Senaryoya uygun girdiler ve doÄŸrulamalar Ã¼retilecek.",
        ]

    @staticmethod
    def _create_test_name(
        function_name: str,
        scenario: Scenario,
    ) -> str:
        """
        Senaryo iÃ§in kararlÄ± ve geÃ§erli pytest fonksiyon adÄ± oluÅŸturur.

        Args:
            function_name: Test edilen fonksiyonun adÄ±.
            scenario: Test senaryosu.

        Returns:
            ``test_`` ile baÅŸlayan pytest fonksiyon adÄ±.
        """
        return (
            f"test_{function_name}_"
            f"path_{scenario.path_index}_"
            f"priority_{scenario.priority_rank}"
        )

