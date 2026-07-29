from __future__ import annotations

from dataclasses import dataclass

from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore


@dataclass(frozen=True, slots=True)
class Scenario:
    """
    Bir yürütme yolu için oluşturulan öncelikli test senaryosunu temsil eder.

    Attributes:
        scenario_id: Senaryonun benzersiz kimliği.
        name: Test senaryosunun okunabilir adı.
        path_index: Senaryonun ilişkili olduğu yürütme yolu numarası.
        priority_rank: DQM sıralamasındaki öncelik sırası.
        priority_level: Senaryonun High, Medium veya Low öncelik seviyesi.
        dqm_score: Normalize edilmiş DQM skoru.
        node_ids: Yürütme yolunda ziyaret edilen CFG düğümleri.
        edge_labels: Yürütme yolundaki kenar etiketleri.
        contains_loop: Yolun döngü içerip içermediği.
        contains_exception: Yolun istisna akışı içerip içermediği.
        description: Senaryonun açıklaması.
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


class ScenarioGenerator:
    """DQM sonuçlarından önceliklendirilmiş test senaryoları üretir."""

    def generate(
        self,
        function_name: str,
        paths: list[ExecutionPath],
        scores: list[DQMScore],
    ) -> list[Scenario]:
        """
        Bir fonksiyona ait yürütme yollarını test senaryolarına dönüştürür.

        Args:
            function_name: Analiz edilen fonksiyonun adı.
            paths: CFG üzerinden elde edilen yürütme yolları.
            scores: DQM tarafından önceliklendirilmiş yol skorları.

        Returns:
            DQM önceliğine göre sıralanmış test senaryoları.

        Raises:
            ValueError: Fonksiyon adı boşsa veya geçersiz yol numarası varsa.
        """
        normalized_function_name = function_name.strip()

        if not normalized_function_name:
            raise ValueError("Fonksiyon adı boş olamaz.")

        scenarios: list[Scenario] = []

        for priority_rank, score in enumerate(scores, start=1):
            path = self._get_path(
                paths=paths,
                path_index=score.path_index,
            )

            scenario = Scenario(
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
        Bir tabanlı yol numarasına göre yürütme yolunu döndürür.

        Args:
            paths: Mevcut yürütme yolları.
            path_index: DQM sonucundaki bir tabanlı yol numarası.

        Returns:
            İstenen yürütme yolu.

        Raises:
            ValueError: Yol numarası mevcut yollarla eşleşmiyorsa.
        """
        zero_based_index = path_index - 1

        if zero_based_index < 0 or zero_based_index >= len(paths):
            raise ValueError(
                f"Geçersiz yürütme yolu numarası: {path_index}"
            )

        return paths[zero_based_index]

    @staticmethod
    def _create_scenario_id(
        function_name: str,
        priority_rank: int,
    ) -> str:
        """Senaryo için kararlı bir kimlik oluşturur."""
        return (
            f"{function_name.lower()}_"
            f"scenario_{priority_rank:03d}"
        )

    @staticmethod
    def _create_scenario_name(
        function_name: str,
        path_index: int,
    ) -> str:
        """Okunabilir test senaryosu adı oluşturur."""
        return (
            f"{function_name} fonksiyonu "
            f"yürütme yolu {path_index}"
        )

    @staticmethod
    def _create_description(
        function_name: str,
        score: DQMScore,
    ) -> str:
        """DQM bilgilerine göre senaryo açıklaması oluşturur."""
        properties: list[str] = [
            f"{score.priority_level} öncelikli",
            f"DQM skoru {score.normalized_score}",
        ]

        if score.contains_loop:
            properties.append("döngü akışı içeriyor")

        if score.contains_exception:
            properties.append("istisna akışı içeriyor")

        joined_properties = ", ".join(properties)

        return (
            f"{function_name} fonksiyonunun "
            f"{score.path_index}. yürütme yolu için oluşturuldu; "
            f"{joined_properties}."
        )

