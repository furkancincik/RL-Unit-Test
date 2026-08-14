from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore
from generator.path_input_generator import (
    GeneratedTestInput,
    PathInputGenerator,
    UnreachablePathError,
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
            Ulaşılabilir senaryolar arasındaki öncelik sırası.

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

    Mantıksal olarak çelişkili kısıtlar içeren yürütme yolları
    ulaşılamaz kabul edilir ve senaryo listesine eklenmez.
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
            path_input_generator
            if path_input_generator is not None
            else PathInputGenerator()
        )
        self._skipped_path_indices: tuple[int, ...] = ()

    @property
    def skipped_path_indices(self) -> tuple[int, ...]:
        """
        Son generate çağrısında ulaşılamaz olduğu için atlanan
        bir tabanlı yürütme yolu numaralarını döndürür.
        """
        return self._skipped_path_indices

    @property
    def skipped_path_count(self) -> int:
        """Son generate çağrısında atlanan yol sayısını döndürür."""
        return len(self._skipped_path_indices)

    def generate(
        self,
        function_name: str,
        paths: list[ExecutionPath],
        scores: list[DQMScore],
        parameter_names: tuple[str, ...] = (),
        parameter_types: dict[str, str] | None = None,
        candidate_values_by_path: dict[
            int,
            dict[str, Any],
        ] | None = None,
    ) -> list[Scenario]:
        """
        Bir fonksiyona ait yürütme yollarını test senaryolarına
        dönüştürür.

        Ulaşılamaz yolların kısıtları PathInputGenerator tarafından
        UnreachablePathError ile bildirilir. Bu yollar atlanır ve
        kalan senaryoların priority_rank değerleri kesintisiz biçimde
        yeniden numaralandırılır.

        Args:
            function_name:
                Analiz edilen fonksiyonun adı.

            paths:
                CFG üzerinden elde edilen yürütme yolları.

            scores:
                DQM tarafından önceliklendirilmiş yol skorları.

            parameter_names:
                Test edilen fonksiyonun parametre adları.

            parameter_types:
                Parametre type hint eşlemesi.

            candidate_values_by_path:
                Path index -> candidate value sözlüğü eşlemesi.
                Feasibility / InputCandidateGenerator katmanından gelen
                adaylar ilgili yürütme yolu için PathInputGenerator'a
                aktarılır.

        Returns:
            DQM önceliğine göre sıralanmış, yalnızca ulaşılabilir
            test senaryoları.

        Raises:
            TypeError:
                Fonksiyon adı string değilse, yollar veya skorlar liste
                değilse ya da parametre adları tuple değilse.

            ValueError:
                Fonksiyon adı boşsa, parametre adı geçersizse veya
                DQM sonucu geçersiz bir yürütme yolunu gösteriyorsa.
        """
        normalized_function_name = self._normalize_function_name(
            function_name
        )
        self._validate_paths(paths)
        self._validate_scores(scores)
        self._validate_parameter_names(parameter_names)

        self._validate_candidate_values_by_path(
            candidate_values_by_path
        )

        normalized_parameter_types = (
            self._normalize_parameter_types(
                parameter_types=parameter_types,
                parameter_names=parameter_names,
            )
        )

        scenarios: list[Scenario] = []
        skipped_path_indices: list[int] = []

        for score in scores:
            path = self._get_path(
                paths=paths,
                path_index=score.path_index,
            )

            try:
                generated_input = (
                    self._path_input_generator.generate(
                        path=path,
                        parameter_names=parameter_names,
                        parameter_types=normalized_parameter_types,
                        candidate_values=(
                            candidate_values_by_path.get(
                                score.path_index
                            )
                            if candidate_values_by_path
                            is not None
                            else None
                        ),
                    )
                )
            except UnreachablePathError:
                skipped_path_indices.append(
                    score.path_index
                )
                continue

            priority_rank = len(scenarios) + 1

            scenario = self._create_scenario(
                function_name=normalized_function_name,
                path=path,
                score=score,
                priority_rank=priority_rank,
                generated_input=generated_input,
            )

            scenarios.append(scenario)

        self._skipped_path_indices = tuple(
            skipped_path_indices
        )

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
    def _normalize_function_name(
        function_name: str,
    ) -> str:
        """Fonksiyon adını doğrular ve normalize eder."""
        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        normalized_function_name = function_name.strip()

        if not normalized_function_name:
            raise ValueError(
                "Fonksiyon adı boş olamaz."
            )

        return normalized_function_name

    @staticmethod
    def _validate_paths(
        paths: list[ExecutionPath],
    ) -> None:
        """Yürütme yolu listesini doğrular."""
        if not isinstance(paths, list):
            raise TypeError(
                "paths bir liste olmalıdır."
            )

        if any(
            not isinstance(path, ExecutionPath)
            for path in paths
        ):
            raise TypeError(
                "paths yalnızca ExecutionPath "
                "nesneleri içermelidir."
            )

    @staticmethod
    def _validate_scores(
        scores: list[DQMScore],
    ) -> None:
        """DQM skoru listesini doğrular."""
        if not isinstance(scores, list):
            raise TypeError(
                "scores bir liste olmalıdır."
            )

        if any(
            not isinstance(score, DQMScore)
            for score in scores
        ):
            raise TypeError(
                "scores yalnızca DQMScore "
                "nesneleri içermelidir."
            )

    @staticmethod
    def _validate_parameter_names(
        parameter_names: tuple[str, ...],
    ) -> None:
        """Fonksiyon parametre adlarını doğrular."""
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
    def _validate_candidate_values_by_path(
        candidate_values_by_path: dict[
            int,
            dict[str, Any],
        ] | None,
    ) -> None:
        """
        Path index bazlı candidate input eşlemesini doğrular.
        """
        if candidate_values_by_path is None:
            return

        if not isinstance(
            candidate_values_by_path,
            dict,
        ):
            raise TypeError(
                "candidate_values_by_path bir dict veya None olmalıdır."
            )

        for path_index, candidate_values in (
            candidate_values_by_path.items()
        ):
            if (
                not isinstance(path_index, int)
                or isinstance(path_index, bool)
                or path_index <= 0
            ):
                raise ValueError(
                    "candidate_values_by_path anahtarları "
                    "pozitif int path index değerleri olmalıdır."
                )

            if not isinstance(
                candidate_values,
                dict,
            ):
                raise TypeError(
                    "candidate_values_by_path değerleri "
                    "dict olmalıdır."
                )

            for variable_name in candidate_values:
                if (
                    not isinstance(variable_name, str)
                    or not variable_name.strip()
                ):
                    raise ValueError(
                        "candidate_values_by_path içindeki "
                        "değişken adları boş olmayan string "
                        "değerler olmalıdır."
                    )

    @staticmethod
    def _normalize_parameter_types(
        *,
        parameter_types: dict[str, str] | None,
        parameter_names: tuple[str, ...],
    ) -> dict[str, str]:
        """Parametre type hint eşlemesini doğrular."""
        if parameter_types is None:
            return {}

        if not isinstance(parameter_types, dict):
            raise TypeError(
                "parameter_types bir dict veya None olmalıdır."
            )

        normalized: dict[str, str] = {}

        for parameter_name, type_name in parameter_types.items():
            if (
                not isinstance(parameter_name, str)
                or not parameter_name.strip()
            ):
                raise ValueError(
                    "parameter_types anahtarları boş olmayan "
                    "string değerler olmalıdır."
                )

            if parameter_name not in parameter_names:
                raise ValueError(
                    "parameter_types bilinmeyen parametre içeriyor: "
                    f"{parameter_name}"
                )

            if (
                not isinstance(type_name, str)
                or not type_name.strip()
            ):
                raise ValueError(
                    "parameter_types değerleri boş olmayan "
                    "string değerler olmalıdır."
                )

            normalized[parameter_name] = type_name.strip()

        return normalized

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
