from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore
from generator.path_input_generator import (
    GeneratedTestInput,
    PathInputGenerator,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """
    Bir yürütme yolu için oluşturulan öncelikli test senaryosunu
    temsil eder.

    Attributes:
        scenario_id:
            Senaryonun benzersiz kimliği.

        name:
            Test senaryosunun okunabilir adı.

        path_index:
            Senaryonun ilişkili olduğu yürütme yolu numarası.

        priority_rank:
            DQM sıralamasındaki öncelik sırası.

        priority_level:
            Senaryonun High, Medium veya Low öncelik seviyesi.

        dqm_score:
            Normalize edilmiş DQM skoru.

        node_ids:
            Yürütme yolunda ziyaret edilen CFG düğümleri.

        edge_labels:
            Yürütme yolundaki kenar etiketleri.

        contains_loop:
            Yolun döngü içerip içermediği.

        contains_exception:
            Yolun istisna akışı içerip içermediği.

        description:
            Senaryonun açıklaması.

        keyword_arguments:
            Fonksiyon çağrısında kullanılacak parametreler.

        expected_result:
            Fonksiyon çağrısından beklenen sonuç.

        expected_exception:
            Beklenen exception sınıfının adı.
            Exception beklenmiyorsa None olur.
    """

    scenario_id: str
    name: str
    path_index: int
    priority_rank: int
    priority_level: str
    dqm_score: float
    node_ids: tuple[int, ...]
    edge_labels: tuple[str | None, ...]
    contains_loop: bool
    contains_exception: bool
    description: str

    keyword_arguments: tuple[tuple[str, Any], ...] = ()
    expected_result: Any = None
    expected_exception: str | None = None

    @property
    def keyword_argument_dict(self) -> dict[str, Any]:
        """
        Fonksiyon çağrısı için kullanılacak anahtar kelimeli
        argümanları sözlük biçiminde döndürür.
        """
        return dict(self.keyword_arguments)

    @property
    def expects_exception(self) -> bool:
        """
        Senaryonun bir exception bekleyip beklemediğini belirtir.
        """
        return self.expected_exception is not None


class ScenarioGenerator:
    """
    DQM sonuçlarından önceliklendirilmiş ve çalıştırılabilir
    test senaryoları üretir.
    """

    def __init__(
        self,
        path_input_generator: PathInputGenerator | None = None,
    ) -> None:
        """
        ScenarioGenerator bağımlılıklarını hazırlar.

        Args:
            path_input_generator:
                Yürütme yollarından somut test girdileri üreten bileşen.
                Verilmezse varsayılan PathInputGenerator oluşturulur.
        """
        self._path_input_generator = (
            path_input_generator or PathInputGenerator()
        )

    def generate(
        self,
        function_name: str,
        paths: list[ExecutionPath],
        scores: list[DQMScore],
        parameter_names: tuple[str, ...] = (),
    ) -> list[Scenario]:
        """
        Bir fonksiyona ait yürütme yollarını test senaryolarına
        dönüştürür.

        Args:
            function_name:
                Analiz edilen fonksiyonun adı.

            paths:
                CFG üzerinden elde edilen yürütme yolları.

            scores:
                DQM tarafından önceliklendirilmiş yol skorları.

            parameter_names:
                Test edilen fonksiyonun parametre adları.

        Returns:
            DQM önceliğine göre sıralanmış test senaryoları.

        Raises:
            TypeError:
                Parametre adları tuple değilse.

            ValueError:
                Fonksiyon adı boşsa, parametre adı geçersizse veya
                DQM sonucu geçersiz bir yürütme yolunu gösteriyorsa.
        """
        normalized_function_name = function_name.strip()

        if not normalized_function_name:
            raise ValueError(
                "Fonksiyon adı boş olamaz."
            )

        self._validate_parameter_names(parameter_names)

        scenarios: list[Scenario] = []

        for priority_rank, score in enumerate(
            scores,
            start=1,
        ):
            path = self._get_path(
                paths=paths,
                path_index=score.path_index,
            )

            generated_input = (
                self._path_input_generator.generate(
                    path=path,
                    parameter_names=parameter_names,
                )
            )

            scenario = self._create_scenario(
                function_name=normalized_function_name,
                path=path,
                score=score,
                priority_rank=priority_rank,
                generated_input=generated_input,
            )

            scenarios.append(scenario)

        return scenarios

    def _create_scenario(
        self,
        function_name: str,
        path: ExecutionPath,
        score: DQMScore,
        priority_rank: int,
        generated_input: GeneratedTestInput,
    ) -> Scenario:
        """
        Yürütme yolu, DQM skoru ve test girdisinden Scenario üretir.
        """
        return Scenario(
            scenario_id=self._create_scenario_id(
                function_name=function_name,
                priority_rank=priority_rank,
            ),
            name=self._create_scenario_name(
                function_name=function_name,
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
                function_name=function_name,
                score=score,
                generated_input=generated_input,
            ),
            keyword_arguments=(
                generated_input.keyword_arguments
            ),
            expected_result=(
                generated_input.expected_result
            ),
            expected_exception=(
                generated_input.expected_exception
            ),
        )

    @staticmethod
    def _validate_parameter_names(
        parameter_names: tuple[str, ...],
    ) -> None:
        """
        Fonksiyon parametre adlarını doğrular.
        """
        if not isinstance(parameter_names, tuple):
            raise TypeError(
                "parameter_names bir tuple olmalıdır."
            )

        if any(
            not isinstance(parameter_name, str)
            or not parameter_name.strip()
            for parameter_name in parameter_names
        ):
            raise ValueError(
                "parameter_names yalnızca boş olmayan string "
                "değerler içermelidir."
            )

        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(
                "parameter_names tekrar eden değer içeremez."
            )

    @staticmethod
    def _get_path(
        paths: list[ExecutionPath],
        path_index: int,
    ) -> ExecutionPath:
        """
        Bir tabanlı yol numarasına göre yürütme yolunu döndürür.
        """
        zero_based_index = path_index - 1

        if (
            zero_based_index < 0
            or zero_based_index >= len(paths)
        ):
            raise ValueError(
                f"Geçersiz yürütme yolu numarası: {path_index}"
            )

        return paths[zero_based_index]

    @staticmethod
    def _create_scenario_id(
        function_name: str,
        priority_rank: int,
    ) -> str:
        """
        Senaryo için kararlı bir kimlik oluşturur.
        """
        return (
            f"{function_name.lower()}_"
            f"scenario_{priority_rank:03d}"
        )

    @staticmethod
    def _create_scenario_name(
        function_name: str,
        path_index: int,
    ) -> str:
        """
        Okunabilir test senaryosu adı oluşturur.
        """
        return (
            f"{function_name} fonksiyonu "
            f"yürütme yolu {path_index}"
        )

    @staticmethod
    def _create_description(
        function_name: str,
        score: DQMScore,
        generated_input: GeneratedTestInput,
    ) -> str:
        """
        DQM ve test girdisi bilgilerine göre açıklama oluşturur.
        """
        properties: list[str] = [
            f"{score.priority_level} öncelikli",
            f"DQM skoru {score.normalized_score}",
        ]

        if score.contains_loop:
            properties.append(
                "döngü akışı içeriyor"
            )

        if score.contains_exception:
            properties.append(
                "istisna akışı içeriyor"
            )

        if generated_input.expected_exception is not None:
            properties.append(
                "beklenen exception "
                f"{generated_input.expected_exception}"
            )
        else:
            properties.append(
                "beklenen sonuç "
                f"{generated_input.expected_result!r}"
            )

        joined_properties = ", ".join(properties)

        return (
            f"{function_name} fonksiyonunun "
            f"{score.path_index}. yürütme yolu için oluşturuldu; "
            f"{joined_properties}."
        )