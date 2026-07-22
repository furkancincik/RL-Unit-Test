from __future__ import annotations

from dataclasses import dataclass

from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore


@dataclass(frozen=True, slots=True)
class TestScenario:
    """
    Bir yÃ¼rÃ¼tme yolu iÃ§in oluÅŸturulan Ã¶ncelikli test senaryosunu temsil eder.

    Attributes:
        scenario_id: Senaryonun benzersiz kimliÄŸi.
        name: Test senaryosunun okunabilir adÄ±.
        path_index: Senaryonun iliÅŸkili olduÄŸu yÃ¼rÃ¼tme yolu numarasÄ±.
        priority_rank: DQM sÄ±ralamasÄ±ndaki Ã¶ncelik sÄ±rasÄ±.
        priority_level: Senaryonun High, Medium veya Low Ã¶ncelik seviyesi.
        dqm_score: Normalize edilmiÅŸ DQM skoru.
        node_ids: YÃ¼rÃ¼tme yolunda ziyaret edilen CFG dÃ¼ÄŸÃ¼mleri.
        edge_labels: YÃ¼rÃ¼tme yolundaki kenar etiketleri.
        contains_loop: Yolun dÃ¶ngÃ¼ iÃ§erip iÃ§ermediÄŸi.
        contains_exception: Yolun istisna akÄ±ÅŸÄ± iÃ§erip iÃ§ermediÄŸi.
        description: Senaryonun aÃ§Ä±klamasÄ±.
    """

    scenario_id: str
    name: str
    path_index: int
    priority_rank: int
    priority_level: str
    dqm_score: float
    node_ids: tuple[int, ...]
    edge_labels: tuple[str, ...]
    contains_loop: bool
    contains_exception: bool
    description: str


class TestScenarioGenerator:
    """DQM sonuÃ§larÄ±ndan Ã¶nceliklendirilmiÅŸ test senaryolarÄ± Ã¼retir."""

    def generate(
        self,
        function_name: str,
        paths: list[ExecutionPath],
        scores: list[DQMScore],
    ) -> list[TestScenario]:
        """
        Bir fonksiyona ait yÃ¼rÃ¼tme yollarÄ±nÄ± test senaryolarÄ±na dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.

        Args:
            function_name: Analiz edilen fonksiyonun adÄ±.
            paths: CFG Ã¼zerinden elde edilen yÃ¼rÃ¼tme yollarÄ±.
            scores: DQM tarafÄ±ndan Ã¶nceliklendirilmiÅŸ yol skorlarÄ±.

        Returns:
            DQM Ã¶nceliÄŸine gÃ¶re sÄ±ralanmÄ±ÅŸ test senaryolarÄ±.

        Raises:
            ValueError: Fonksiyon adÄ± boÅŸsa veya geÃ§ersiz yol numarasÄ± varsa.
        """
        normalized_function_name = function_name.strip()

        if not normalized_function_name:
            raise ValueError("Fonksiyon adÄ± boÅŸ olamaz.")

        scenarios: list[TestScenario] = []

        for priority_rank, score in enumerate(scores, start=1):
            path = self._get_path(
                paths=paths,
                path_index=score.path_index,
            )

            scenario = TestScenario(
                scenario_id=self._create_scenario_id(
                    function_name=normalized_function_name,
                    priority_rank=priority_rank,
                ),
                name=self._create_scenario_name(
                    function_name=normalized_function_name,
                    path_index=score.path_index,
                ),
                path_index=score.path_index,
                priority_rank=priority_rank,
                priority_level=score.priority_level,
                dqm_score=score.normalized_score,
                node_ids=tuple(path.node_ids),
                edge_labels=tuple(path.edge_labels),
                contains_loop=score.contains_loop,
                contains_exception=score.contains_exception,
                description=self._create_description(
                    function_name=normalized_function_name,
                    score=score,
                ),
            )

            scenarios.append(scenario)

        return scenarios

    @staticmethod
    def _get_path(
        paths: list[ExecutionPath],
        path_index: int,
    ) -> ExecutionPath:
        """
        Bir tabanlÄ± yol numarasÄ±na gÃ¶re yÃ¼rÃ¼tme yolunu dÃ¶ndÃ¼rÃ¼r.

        Args:
            paths: Mevcut yÃ¼rÃ¼tme yollarÄ±.
            path_index: DQM sonucundaki bir tabanlÄ± yol numarasÄ±.

        Returns:
            Ä°stenen yÃ¼rÃ¼tme yolu.

        Raises:
            ValueError: Yol numarasÄ± mevcut yollarla eÅŸleÅŸmiyorsa.
        """
        zero_based_index = path_index - 1

        if zero_based_index < 0 or zero_based_index >= len(paths):
            raise ValueError(
                f"GeÃ§ersiz yÃ¼rÃ¼tme yolu numarasÄ±: {path_index}"
            )

        return paths[zero_based_index]

    @staticmethod
    def _create_scenario_id(
        function_name: str,
        priority_rank: int,
    ) -> str:
        """Senaryo iÃ§in kararlÄ± bir kimlik oluÅŸturur."""
        return (
            f"{function_name.lower()}_"
            f"scenario_{priority_rank:03d}"
        )

    @staticmethod
    def _create_scenario_name(
        function_name: str,
        path_index: int,
    ) -> str:
        """Okunabilir test senaryosu adÄ± oluÅŸturur."""
        return (
            f"{function_name} fonksiyonu "
            f"yÃ¼rÃ¼tme yolu {path_index}"
        )

    @staticmethod
    def _create_description(
        function_name: str,
        score: DQMScore,
    ) -> str:
        """DQM bilgilerine gÃ¶re senaryo aÃ§Ä±klamasÄ± oluÅŸturur."""
        properties: list[str] = [
            f"{score.priority_level} Ã¶ncelikli",
            f"DQM skoru {score.normalized_score}",
        ]

        if score.contains_loop:
            properties.append("dÃ¶ngÃ¼ akÄ±ÅŸÄ± iÃ§eriyor")

        if score.contains_exception:
            properties.append("istisna akÄ±ÅŸÄ± iÃ§eriyor")

        joined_properties = ", ".join(properties)

        return (
            f"{function_name} fonksiyonunun "
            f"{score.path_index}. yÃ¼rÃ¼tme yolu iÃ§in oluÅŸturuldu; "
            f"{joined_properties}."
        )

