from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.scenario_generator import (
    Scenario,
    ScenarioGenerator,
)
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_suite_coverage_transition import (
    ScenarioSuiteCoverageTransition,
)
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)
from rl.state_encoder import StateEncoder
from rl.training_report_formatter import TrainingReportFormatter
from rl.training_session import (
    TrainingSession,
    TrainingSessionResult,
)
from rl.training_statistics import TrainingStatistics


@dataclass(frozen=True, slots=True)
class RealRLTrainingResult:
    """
    Gerçek coverage tabanlı RL eğitim işleminin sonucudur.

    Attributes:
        source_file:
            Analiz ve coverage uygulanan Python dosyası.

        module_path:
            Kaynak dosyanın Python import yolu.

        function_name:
            RL eğitimi gerçekleştirilen fonksiyon.

        scenarios:
            Fonksiyon için üretilen test senaryoları.

        session_result:
            Çok episode eğitim oturumunun sonucu.

        statistics:
            Eğitim sırasında hesaplanan istatistikler.

        q_table_state_count:
            Eğitim sonunda Q-Table içerisinde bulunan state sayısı.

        report:
            Terminalde veya dosyada kullanılabilecek eğitim raporu.
    """

    source_file: Path
    module_path: str
    function_name: str
    scenarios: tuple[Scenario, ...]
    session_result: TrainingSessionResult
    statistics: TrainingStatistics
    q_table_state_count: int
    report: str

    @property
    def success(self) -> bool:
        """Gerçek RL eğitim oturumunun başarı durumunu döndürür."""
        return self.session_result.success

    @property
    def scenario_count(self) -> int:
        """Üretilen senaryo sayısını döndürür."""
        return len(self.scenarios)

    @property
    def completed_episode_count(self) -> int:
        """Tamamlanan episode sayısını döndürür."""
        return self.session_result.completed_episode_count


class RealRLTrainingService:
    """
    Bir Python fonksiyonu için analizden gerçek RL eğitimine kadar
    bütün süreci uçtan uca yönetir.

    İş akışı:

        PythonAnalyzer
                ↓
        ControlFlowGraphBuilder
                ↓
        CFGPathAnalyzer
                ↓
        DecisionQualityMatrix
                ↓
        ScenarioGenerator
                ↓
        ScenarioActionMapper
                ↓
        ScenarioSuiteCoverageTransition
                ↓
        CoverageEnvironment
                ↓
        QLearningAgent ve QLearningTrainer
                ↓
        TrainingSession
                ↓
        TrainingStatistics ve eğitim raporu
    """

    def __init__(
        self,
        analyzer: PythonAnalyzer | None = None,
        cfg_builder: ControlFlowGraphBuilder | None = None,
        path_analyzer: CFGPathAnalyzer | None = None,
        dqm: DecisionQualityMatrix | None = None,
        scenario_generator: ScenarioGenerator | None = None,
        report_formatter: TrainingReportFormatter | None = None,
    ) -> None:
        """Servisin analiz ve raporlama bağımlılıklarını hazırlar."""
        self._analyzer = analyzer or PythonAnalyzer()
        self._cfg_builder = (
            cfg_builder or ControlFlowGraphBuilder()
        )
        self._path_analyzer = (
            path_analyzer or CFGPathAnalyzer()
        )
        self._dqm = dqm or DecisionQualityMatrix()
        self._scenario_generator = (
            scenario_generator or ScenarioGenerator()
        )
        self._report_formatter = (
            report_formatter or TrainingReportFormatter()
        )

    def run(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        output_directory: str | Path,
        episode_count: int = 3,
        epsilon: float = 0.0,
        learning_rate: float = 0.5,
        discount_factor: float = 0.9,
        random_seed: int | None = 42,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
    ) -> RealRLTrainingResult:
        """
        Seçilen fonksiyon için gerçek coverage tabanlı RL eğitimi yapar.

        Args:
            source_file:
                Analiz edilecek Python kaynak dosyası.

            module_path:
                Dosyanın Python import yolu.

            function_name:
                RL eğitimi gerçekleştirilecek fonksiyonun adı.

            output_directory:
                Kümülatif pytest dosyasının yazılacağı klasör.

            episode_count:
                Gerçekleştirilecek eğitim episode sayısı.

            epsilon:
                Epsilon-greedy politikasının keşif oranı.

            learning_rate:
                Q-Learning öğrenme oranı.

            discount_factor:
                Gelecekteki ödüllerin ağırlığı.

            random_seed:
                Tekrarlanabilir action seçimi için rastgelelik tohumu.

            overwrite:
                Üretilen test dosyasının üzerine yazma izni.

            timeout_seconds:
                Her coverage çalıştırması için süre sınırı.

        Returns:
            Senaryoları, eğitim sonucunu, istatistikleri ve raporu
            içeren RealRLTrainingResult.

        Raises:
            FileNotFoundError:
                Kaynak dosya bulunamazsa.

            ValueError:
                Fonksiyon bulunamazsa veya ilgili aşamalardan sonuç
                üretilemezse.

            TypeError:
                Parametrelerden biri geçersiz türdeyse.
        """
        normalized_source_file = self._normalize_source_file(
            source_file
        )
        normalized_module_path = self._normalize_module_path(
            module_path
        )
        normalized_function_name = self._normalize_function_name(
            function_name
        )
        normalized_output_directory = (
            self._normalize_output_directory(
                output_directory
            )
        )

        self._validate_episode_count(episode_count)
        self._validate_probability(
            name="epsilon",
            value=epsilon,
        )
        self._validate_probability(
            name="learning_rate",
            value=learning_rate,
        )
        self._validate_probability(
            name="discount_factor",
            value=discount_factor,
        )
        self._validate_random_seed(random_seed)
        self._validate_overwrite(overwrite)
        self._validate_timeout(timeout_seconds)

        analysis_result = self._analyzer.analyze_file(
            normalized_source_file
        )

        graphs = self._cfg_builder.build_from_file(
            normalized_source_file
        )

        function = next(
            (
                candidate
                for candidate in analysis_result.functions
                if candidate.name == normalized_function_name
            ),
            None,
        )

        if function is None:
            raise ValueError(
                "Kaynak dosyada belirtilen fonksiyon bulunamadı: "
                f"{normalized_function_name}"
            )

        graph = next(
            (
                candidate
                for candidate in graphs
                if candidate.function_name
                == normalized_function_name
            ),
            None,
        )

        if graph is None:
            raise ValueError(
                "Fonksiyon için Control Flow Graph bulunamadı: "
                f"{normalized_function_name}"
            )

        paths = self._path_analyzer.find_paths(
            graph
        )

        if not paths:
            raise ValueError(
                "Fonksiyon için yürütme yolu bulunamadı: "
                f"{normalized_function_name}"
            )

        scores = self._dqm.evaluate_paths(
            function=function,
            paths=paths,
        )

        if not scores:
            raise ValueError(
                "Fonksiyon için DQM sonucu üretilemedi: "
                f"{normalized_function_name}"
            )

        parameter_names = self._extract_parameter_names(
            function.parameters
        )

        scenarios = self._scenario_generator.generate(
            function_name=normalized_function_name,
            paths=paths,
            scores=scores,
            parameter_names=parameter_names,
        )

        if not scenarios:
            raise ValueError(
                "Fonksiyon için test senaryosu üretilemedi: "
                f"{normalized_function_name}"
            )

        scenario_tuple = tuple(
            scenarios
        )

        mapper = ScenarioActionMapper(
            scenarios=scenario_tuple,
        )

        suite_transition = ScenarioSuiteCoverageTransition(
            source_file=normalized_source_file,
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            output_directory=normalized_output_directory,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
        )

        transition_adapter = ScenarioTransitionAdapter(
            mapper=mapper,
            transition_function=suite_transition,
        )

        initial_state = self._create_initial_state(
            graph_node_count=len(graph.nodes),
            branch_count=function.branch_count,
        )

        environment = CoverageEnvironment(
            initial_state=initial_state,
            actions=mapper.actions,
            transition_function=transition_adapter,
            episode_reset_callback=suite_transition.reset,
        )

        q_table = QTable()

        policy = EpsilonGreedyPolicy(
            epsilon=epsilon,
            random_generator=random.Random(
                random_seed
            ),
        )

        agent = QLearningAgent(
            q_table=q_table,
            policy=policy,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        state_encoder = StateEncoder(
            coverage_bucket_size=10.0,
            missing_lines_bucket_size=2,
            uncovered_branches_bucket_size=1,
        )

        trainer = QLearningTrainer(
            agent=agent,
            state_encoder=state_encoder,
        )

        statistics = TrainingStatistics()

        session = TrainingSession(
            trainer=trainer,
            statistics=statistics,
        )

        session_result = session.run(
            environment=environment,
            episode_count=episode_count,
        )

        report = self._report_formatter.format_session(
            result=session_result,
            statistics=statistics,
        )

        return RealRLTrainingResult(
            source_file=normalized_source_file,
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=scenario_tuple,
            session_result=session_result,
            statistics=statistics,
            q_table_state_count=len(q_table),
            report=report,
        )

    @staticmethod
    def _create_initial_state(
        *,
        graph_node_count: int,
        branch_count: int,
    ) -> CoverageState:
        """
        Gerçek ölçüm öncesindeki başlangıç RL durumunu oluşturur.

        İlk coverage ölçümü henüz gerçekleştirilmediği için eksik
        satırlar CFG düğüm sayısını koruyan temsili değerlerdir.
        """
        representative_line_count = max(
            1,
            graph_node_count,
        )

        return CoverageState(
            coverage_percentage=0.0,
            executed_tests=0,
            missing_lines=tuple(
                range(
                    1,
                    representative_line_count + 1,
                )
            ),
            uncovered_branches=max(
                0,
                branch_count,
            ),
        )

    @staticmethod
    def _extract_parameter_names(
        parameters: object,
    ) -> tuple[str, ...]:
        """Analyzer tarafından bulunan parametre adlarını normalize eder."""
        if not isinstance(
            parameters,
            (list, tuple),
        ):
            raise TypeError(
                "Fonksiyon parametreleri liste veya tuple olmalıdır."
            )

        normalized_parameters: list[str] = []

        for parameter in parameters:
            if not isinstance(parameter, str):
                raise TypeError(
                    "Fonksiyon parametre adları string olmalıdır."
                )

            normalized_parameter = parameter.strip()

            if not normalized_parameter:
                raise ValueError(
                    "Fonksiyon parametre adı boş olamaz."
                )

            normalized_parameters.append(
                normalized_parameter
            )

        if (
            len(set(normalized_parameters))
            != len(normalized_parameters)
        ):
            raise ValueError(
                "Fonksiyon parametre adları tekrar edemez."
            )

        return tuple(
            normalized_parameters
        )

    @staticmethod
    def _normalize_source_file(
        source_file: str | Path,
    ) -> Path:
        """Kaynak dosyayı doğrular ve mutlak yola dönüştürür."""
        if not isinstance(source_file, (str, Path)):
            raise TypeError(
                "source_file string veya Path olmalıdır."
            )

        if (
            isinstance(source_file, str)
            and not source_file.strip()
        ):
            raise ValueError(
                "source_file boş olamaz."
            )

        path = Path(source_file)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "source_file uzantısı .py olmalıdır."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Kaynak dosya bulunamadı: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"source_file bir dosya olmalıdır: {path}"
            )

        return path.resolve()

    @staticmethod
    def _normalize_module_path(
        module_path: str,
    ) -> str:
        """Python modül yolunu doğrular."""
        if not isinstance(module_path, str):
            raise TypeError(
                "module_path string olmalıdır."
            )

        normalized_module_path = module_path.strip()

        if not normalized_module_path:
            raise ValueError(
                "module_path boş olamaz."
            )

        if any(
            not part.isidentifier()
            for part in normalized_module_path.split(".")
        ):
            raise ValueError(
                "Geçersiz Python modül yolu: "
                f"{normalized_module_path}"
            )

        return normalized_module_path

    @staticmethod
    def _normalize_function_name(
        function_name: str,
    ) -> str:
        """Fonksiyon adını doğrular."""
        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        normalized_function_name = function_name.strip()

        if not normalized_function_name:
            raise ValueError(
                "function_name boş olamaz."
            )

        if not normalized_function_name.isidentifier():
            raise ValueError(
                "Geçersiz Python fonksiyon adı: "
                f"{normalized_function_name}"
            )

        return normalized_function_name

    @staticmethod
    def _normalize_output_directory(
        output_directory: str | Path,
    ) -> Path:
        """Çıktı klasörünü doğrular."""
        if not isinstance(
            output_directory,
            (str, Path),
        ):
            raise TypeError(
                "output_directory string veya Path olmalıdır."
            )

        if (
            isinstance(output_directory, str)
            and not output_directory.strip()
        ):
            raise ValueError(
                "output_directory boş olamaz."
            )

        return Path(
            output_directory
        )

    @staticmethod
    def _validate_episode_count(
        episode_count: int,
    ) -> None:
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
        ):
            raise TypeError(
                "episode_count bir tam sayı olmalıdır."
            )

        if episode_count < 1:
            raise ValueError(
                "episode_count 1 veya daha büyük olmalıdır."
            )

    @staticmethod
    def _validate_probability(
        *,
        name: str,
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{name} sayısal olmalıdır."
            )

        normalized_value = float(
            value
        )

        if not math.isfinite(
            normalized_value
        ):
            raise ValueError(
                f"{name} sonlu bir sayı olmalıdır."
            )

        if not 0.0 <= normalized_value <= 1.0:
            raise ValueError(
                f"{name} 0 ile 1 arasında olmalıdır."
            )

    @staticmethod
    def _validate_random_seed(
        random_seed: int | None,
    ) -> None:
        if (
            random_seed is not None
            and (
                isinstance(random_seed, bool)
                or not isinstance(random_seed, int)
            )
        ):
            raise TypeError(
                "random_seed bir tam sayı veya None olmalıdır."
            )

    @staticmethod
    def _validate_overwrite(
        overwrite: bool,
    ) -> None:
        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite bool olmalıdır."
            )

    @staticmethod
    def _validate_timeout(
        timeout_seconds: float,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(
                timeout_seconds,
                (int, float),
            )
        ):
            raise TypeError(
                "timeout_seconds sayısal olmalıdır."
            )

        normalized_timeout = float(
            timeout_seconds
        )

        if (
            not math.isfinite(normalized_timeout)
            or normalized_timeout <= 0.0
        ):
            raise ValueError(
                "timeout_seconds sonlu ve sıfırdan büyük olmalıdır."
            )