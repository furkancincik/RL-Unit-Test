from __future__ import annotations

import copy
import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.data_flow_analyzer import DataFlowAnalyzer
from cfg.path_analyzer import CFGPathAnalyzer
from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathFeasibilityAnalyzer,
    PathFeasibilityResult,
)
from cfg.path_state_analyzer import PathStateAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.input_candidate_generator import InputCandidateGenerator
from generator.scenario_generator import (
    Scenario,
    ScenarioGenerator,
)
from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
)
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
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
from services.coverage_reachability_service import (
    CoverageReachabilityService,
)


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

        final_coverage_result:
            Eğitim sonunda elde edilen son başarılı dosya veya
            fonksiyon bazlı coverage sonucu.

        reachability_result:
            Senaryo havuzu coverage sonucu ile bounded execution path
            feasibility analizinin satır bazlı sınıflandırması.

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
    final_coverage_result: CoverageResult | FunctionCoverageResult
    report: str
    reachability_result: (
        FunctionCoverageReachabilityResult | None
    ) = None

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

    @property
    def function_coverage(
        self,
    ) -> FunctionCoverageResult | None:
        """Son sonuç fonksiyon bazlıysa fonksiyon coverage'ını döndürür."""
        if isinstance(
            self.final_coverage_result,
            FunctionCoverageResult,
        ):
            return self.final_coverage_result

        return None

    @property
    def file_coverage(self) -> CoverageResult:
        """
        Son ölçüme ait dosya geneli coverage sonucunu döndürür.
        """
        if isinstance(
            self.final_coverage_result,
            FunctionCoverageResult,
        ):
            return self.final_coverage_result.file_coverage

        return self.final_coverage_result

    @property
    def has_full_function_coverage(self) -> bool | None:
        """
        Fonksiyon bazlı ölçüm varsa tam coverage durumunu döndürür.
        """
        function_coverage = self.function_coverage

        if function_coverage is None:
            return None

        return function_coverage.has_full_coverage

    @property
    def has_full_file_coverage(self) -> bool:
        """Dosya geneli coverage'ın tam olup olmadığını döndürür."""
        return self.file_coverage.has_full_coverage


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
        DataFlowAnalyzer + PathStateAnalyzer
                ↓
        PathFeasibilityAnalyzer
                ↓
        InputCandidateGenerator
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
        data_flow_analyzer: DataFlowAnalyzer | None = None,
        path_state_analyzer: PathStateAnalyzer | None = None,
        path_feasibility_analyzer: PathFeasibilityAnalyzer | None = None,
        input_candidate_generator: InputCandidateGenerator | None = None,
        dqm: DecisionQualityMatrix | None = None,
        scenario_generator: ScenarioGenerator | None = None,
        report_formatter: TrainingReportFormatter | None = None,
        coverage_reachability_service: (
            CoverageReachabilityService | None
        ) = None,
    ) -> None:
        """Servisin analiz ve raporlama bağımlılıklarını hazırlar."""
        self._analyzer = analyzer or PythonAnalyzer()
        self._cfg_builder = (
            cfg_builder or ControlFlowGraphBuilder()
        )
        self._path_analyzer = (
            path_analyzer or CFGPathAnalyzer()
        )
        self._data_flow_analyzer = (
            data_flow_analyzer or DataFlowAnalyzer()
        )
        self._path_state_analyzer = (
            path_state_analyzer or PathStateAnalyzer()
        )
        self._path_feasibility_analyzer = (
            path_feasibility_analyzer
            or PathFeasibilityAnalyzer()
        )
        self._input_candidate_generator = (
            input_candidate_generator
            or InputCandidateGenerator()
        )
        self._dqm = dqm or DecisionQualityMatrix()
        self._scenario_generator = (
            scenario_generator or ScenarioGenerator()
        )
        self._report_formatter = (
            report_formatter or TrainingReportFormatter()
        )
        self._coverage_reachability_service = (
            coverage_reachability_service
            or CoverageReachabilityService()
        )

    def run(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        output_directory: str | Path,
        max_visits_per_node: int = 3,
        episode_count: int = 3,
        epsilon: float = 0.30,
        epsilon_decay_rate: float | None = 0.95,
        minimum_epsilon: float = 0.05,
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

            max_visits_per_node:
                Bir CFG düğümünün tek execution path içerisinde en
                fazla kaç kez ziyaret edilebileceği. Bu sınır döngü
                açılımını ve path sayısını kontrollü tutar. Varsayılan
                3 değeri sıfır, bir ve iki iterasyonlu yolları kapsar.

            episode_count:
                Gerçekleştirilecek eğitim episode sayısı.

            epsilon:
                Epsilon-greedy politikasının başlangıç keşif oranı.

            epsilon_decay_rate:
                Her tamamlanan episode sonrasında epsilon değerine
                uygulanacak decay katsayısı. None verilirse epsilon
                sabit kalır.

            minimum_epsilon:
                Decay aktifken epsilon değerinin düşebileceği
                minimum keşif oranı.

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

        self._validate_max_visits_per_node(
            max_visits_per_node
        )
        self._validate_episode_count(episode_count)
        self._validate_probability(
            name="epsilon",
            value=epsilon,
        )
        self._validate_optional_probability(
            name="epsilon_decay_rate",
            value=epsilon_decay_rate,
        )
        self._validate_probability(
            name="minimum_epsilon",
            value=minimum_epsilon,
        )
        self._validate_epsilon_configuration(
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
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
            graph,
            max_visits_per_node=max_visits_per_node,
        )

        if not paths:
            raise ValueError(
                "Fonksiyon için yürütme yolu bulunamadı: "
                f"{normalized_function_name}"
            )

        parameter_names = self._extract_parameter_names(
            function.parameters
        )

        data_flow_result = (
            self._data_flow_analyzer.analyze_file(
                source_file=normalized_source_file,
                function_name=normalized_function_name,
            )
        )

        path_states = tuple(
            self._path_state_analyzer.analyze_file(
                source_file=normalized_source_file,
                function_name=normalized_function_name,
                path=path,
            )
            for path in paths
        )

        feasibility_results = (
            self._path_feasibility_analyzer.analyze_paths(
                tuple(paths),
                data_flow_result=data_flow_result,
                path_states=path_states,
                parameter_types=function.parameter_types,
            )
        )

        retained_path_indices = {
            path_index
            for path_index, feasibility_result
            in enumerate(
                feasibility_results,
                start=1,
            )
            if (
                feasibility_result.status
                != FeasibilityStatus.INFEASIBLE
            )
        }

        if not retained_path_indices:
            raise ValueError(
                "Fonksiyon için FEASIBLE veya UNKNOWN "
                "yürütme yolu bulunamadı: "
                f"{normalized_function_name}"
            )

        candidate_values_by_path = (
            self._build_candidate_values_by_path(
                paths=paths,
                feasibility_results=feasibility_results,
                path_states=path_states,
                data_flow_result=data_flow_result,
            )
        )

        all_scores = self._dqm.evaluate_paths(
            function=function,
            paths=paths,
        )

        scores = [
            score
            for score in all_scores
            if score.path_index in retained_path_indices
        ]

        if not scores:
            raise ValueError(
                "Fonksiyon için uygulanabilir DQM sonucu "
                "üretilemedi: "
                f"{normalized_function_name}"
            )

        scenarios = self._scenario_generator.generate(
            function_name=normalized_function_name,
            paths=paths,
            scores=scores,
            parameter_names=parameter_names,
            parameter_types=function.parameter_types,
            candidate_values_by_path=(
                candidate_values_by_path
            ),
        )

        if not scenarios:
            raise ValueError(
                "Fonksiyon için test senaryosu üretilemedi: "
                f"{normalized_function_name}"
            )

        scenario_tuple = self._filter_executable_scenarios(
            source_file=normalized_source_file,
            function_name=normalized_function_name,
            scenarios=tuple(scenarios),
        )

        if not scenario_tuple:
            raise ValueError(
                "Üretilen senaryoların hiçbiri hedef fonksiyonda "
                "beklenen sonucu üretmedi: "
                f"{normalized_function_name}"
            )

        mapper = ScenarioActionMapper(
            scenarios=scenario_tuple,
        )

        suite_transition = ScenarioSuiteCoverageTransition(
            source_file=normalized_source_file,
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            output_directory=normalized_output_directory,
            function_start_line=function.line_number,
            function_end_line=function.end_line_number,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
        )

        baseline_coverage_result = (
            suite_transition.measure_scenarios(
                scenario_tuple
            )
        )

        if not isinstance(
            baseline_coverage_result,
            FunctionCoverageResult,
        ):
            raise RuntimeError(
                "Fonksiyon bazlı senaryo havuzu coverage "
                "sonucu üretilemedi."
            )

        reachability_result = (
            self._coverage_reachability_service.analyze(
                coverage_result=baseline_coverage_result,
                paths=tuple(paths),
                feasibility_results=feasibility_results,
                max_visits_per_node=max_visits_per_node,
            )
        )

        scenario_pool_target_coverage = (
            baseline_coverage_result.line_coverage_percent
        )

        if scenario_pool_target_coverage <= 0.0:
            raise RuntimeError(
                "Geçerli senaryo havuzu pozitif bir coverage "
                "hedefi üretemedi."
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
            target_coverage_percentage=(
                scenario_pool_target_coverage
            ),
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
            executed_tests_bucket_size=1,
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
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
        )

        final_coverage_result = (
            suite_transition.last_coverage_result
        )

        if final_coverage_result is None:
            raise RuntimeError(
                "RL eğitimi tamamlandı ancak son coverage sonucu "
                "bulunamadı."
            )

        report = self._report_formatter.format_session(
            result=session_result,
            statistics=statistics,
            function_name=normalized_function_name,
            coverage_result=final_coverage_result,
            reachability_result=reachability_result,
        )

        return RealRLTrainingResult(
            source_file=normalized_source_file,
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=scenario_tuple,
            session_result=session_result,
            statistics=statistics,
            q_table_state_count=len(q_table),
            final_coverage_result=final_coverage_result,
            report=report,
            reachability_result=reachability_result,
        )

    def _build_candidate_values_by_path(
        self,
        *,
        paths: list,
        feasibility_results: tuple[
            PathFeasibilityResult,
            ...
        ],
        path_states: tuple,
        data_flow_result: Any,
    ) -> dict[
        int,
        dict[str, Any],
    ]:
        """
        FEASIBLE path'ler için feasibility constraint'lerini ve varsa
        relational witness değerlerini ScenarioGenerator'ın tüketebileceği
        path-index bazlı candidate sözlüğüne dönüştürür.

        UNKNOWN path'ler güvenlik gereği pipeline'da tutulur ancak kesin
        feasibility kanıtı olmadığı için bu katmandan candidate almaz.
        INFEASIBLE path'ler ise daha önce pipeline'dan elenir.
        """
        candidate_values_by_path: dict[
            int,
            dict[str, Any],
        ] = {}

        for path_index, (
            path,
            feasibility_result,
            path_state,
        ) in enumerate(
            zip(
                paths,
                feasibility_results,
                path_states,
            ),
            start=1,
        ):
            if (
                feasibility_result.status
                != FeasibilityStatus.FEASIBLE
            ):
                continue

            relational_witness = None

            if feasibility_result.relational_constraints:
                domains = (
                    self._path_feasibility_analyzer
                    ._build_domains(
                        feasibility_result.constraints
                    )
                )

                self._path_feasibility_analyzer._apply_data_flow_ranges(
                    domains=domains,
                    data_flow_result=data_flow_result,
                )

                self._path_feasibility_analyzer._apply_path_symbolic_state(
                    domains=domains,
                    path_state=path_state,
                )

                relational_witness = (
                    self._path_feasibility_analyzer
                    ._find_relational_witness(
                        domains=domains,
                        relational_constraints=(
                            feasibility_result
                            .relational_constraints
                        ),
                    )
                )

            candidate = (
                self._input_candidate_generator.generate(
                    feasibility_result=feasibility_result,
                    relational_witness=relational_witness,
                )
            )

            path_input_values = (
                candidate.path_input_value_dict
            )

            if path_input_values:
                candidate_values_by_path[
                    path_index
                ] = path_input_values

        return candidate_values_by_path

    def _filter_executable_scenarios(
        self,
        *,
        source_file: Path,
        function_name: str,
        scenarios: tuple[Scenario, ...],
    ) -> tuple[Scenario, ...]:
        """
        Senaryoları hedef fonksiyonda somut olarak çalıştırır.

        Üretilen girdinin gerçek sonucu, senaryoda beklenen sonuçla
        uyuşmuyorsa senaryo ulaşılamaz veya yanlış modellenmiş kabul
        edilir ve RL aksiyon kümesine eklenmez.
        """
        target_function = self._load_target_function(
            source_file=source_file,
            function_name=function_name,
        )

        executable_scenarios: list[Scenario] = []

        for scenario in scenarios:
            if self._scenario_matches_execution(
                target_function=target_function,
                scenario=scenario,
            ):
                executable_scenarios.append(scenario)

        return tuple(executable_scenarios)

    @staticmethod
    def _load_target_function(
        *,
        source_file: Path,
        function_name: str,
    ) -> Callable[..., Any]:
        """
        Kaynak dosyayı izole bir modül adıyla yükler ve hedef
        fonksiyonu döndürür.
        """
        module_name = (
            "_rl_unit_test_validation_"
            f"{abs(hash(source_file.resolve()))}"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            source_file,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Somut senaryo doğrulaması için kaynak modül "
                "yüklenemedi: "
                f"{source_file}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        target_function = getattr(
            module,
            function_name,
            None,
        )

        if target_function is None:
            raise ValueError(
                "Somut doğrulama için hedef fonksiyon "
                "kaynak modülde bulunamadı: "
                f"{function_name}"
            )

        if not callable(target_function):
            raise TypeError(
                "Somut doğrulama hedefi çağrılabilir olmalıdır: "
                f"{function_name}"
            )

        return target_function

    @classmethod
    def _scenario_matches_execution(
        cls,
        *,
        target_function: Callable[..., Any],
        scenario: Scenario,
    ) -> bool:
        """
        Tek bir senaryonun gerçek çalışma sonucu ile beklenen
        davranışının uyuşup uyuşmadığını döndürür.
        """
        keyword_arguments = copy.deepcopy(
            scenario.keyword_argument_dict
        )

        try:
            actual_result = target_function(
                **keyword_arguments
            )
        except Exception as error:
            if scenario.expected_exception is None:
                return False

            return any(
                exception_type.__name__
                == scenario.expected_exception
                for exception_type in type(error).mro()
            )

        if scenario.expected_exception is not None:
            return False

        return cls._values_match(
            actual=actual_result,
            expected=scenario.expected_result,
        )

    @staticmethod
    def _values_match(
        *,
        actual: Any,
        expected: Any,
    ) -> bool:
        """
        Gerçek ve beklenen değerleri güvenli biçimde karşılaştırır.
        """
        if (
            isinstance(actual, float)
            and isinstance(expected, float)
        ):
            if math.isnan(actual) and math.isnan(expected):
                return True

            return math.isclose(
                actual,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )

        try:
            comparison = actual == expected
        except Exception:
            return False

        return (
            comparison
            if isinstance(comparison, bool)
            else False
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
    def _validate_max_visits_per_node(
        max_visits_per_node: int,
    ) -> None:
        """CFG path üretimindeki düğüm ziyaret sınırını doğrular."""
        if (
            isinstance(max_visits_per_node, bool)
            or not isinstance(max_visits_per_node, int)
        ):
            raise TypeError(
                "max_visits_per_node bir tam sayı olmalıdır."
            )

        if max_visits_per_node < 1:
            raise ValueError(
                "max_visits_per_node 1 veya daha büyük olmalıdır."
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
    def _validate_optional_probability(
        *,
        name: str,
        value: float | None,
    ) -> None:
        """
        None veya 0-1 aralığında sonlu olasılık değeri doğrular.
        """
        if value is None:
            return

        RealRLTrainingService._validate_probability(
            name=name,
            value=value,
        )

    @staticmethod
    def _validate_epsilon_configuration(
        *,
        epsilon: float,
        epsilon_decay_rate: float | None,
        minimum_epsilon: float,
    ) -> None:
        """
        Epsilon decay parametrelerinin birlikte geçerli olmasını doğrular.
        """
        if epsilon_decay_rate is None:
            return

        if float(minimum_epsilon) > float(epsilon):
            raise ValueError(
                "minimum_epsilon başlangıç epsilon "
                "değerinden büyük olamaz."
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
