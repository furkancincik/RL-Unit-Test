from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
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
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
    PipelineStage,
)
from models.scenario_minimization_result import ScenarioMinimizationResult
from models.strategy_comparison_result import StrategyComparisonResult
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
from services.pipeline_timeout_service import (
    GlobalPipelineTimeoutRunner,
    PipelineDiagnosticCheckpointStore,
)
from services.scenario_coverage_minimization_service import (
    ScenarioCoverageMinimizationService,
)
from services.strategy_comparison_service import StrategyComparisonService


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
            Eğitim oturumunun en iyi episode'u için kaydedilmiş dosya
            veya fonksiyon bazlı coverage sonucu.

        reachability_result:
            Senaryo havuzu coverage sonucu ile bounded execution path
            feasibility analizinin satır bazlı sınıflandırması.

        report:
            Terminalde veya dosyada kullanılabilecek eğitim raporu.

        minimization_result:
            Açıkça etkinleştirildiğinde validated scenario havuzu için
            üretilen deterministic greedy baseline sonucu.

        scenario_pool_coverage_result:
            Concrete-valid senaryo havuzunun erişebildiği immutable
            coverage hedefi.
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
    diagnostic: PipelineDiagnosticResult | None = None
    minimization_result: ScenarioMinimizationResult | None = None
    strategy_comparison_result: StrategyComparisonResult | None = None
    scenario_pool_coverage_result: FunctionCoverageResult | None = None

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


class _ControlledPipelineFailure(ValueError):
    """Public diagnostic olarak döndürülebilen kontrollü domain duruşu."""


@dataclass(slots=True)
class _PipelineDiagnosticAccumulator:
    """Tek bir run boyunca doğrulanmış diagnostic değerleri biriktirir."""

    source_file: Path
    function_name: str
    started_at: float
    current_stage: PipelineStage | None = None
    current_stage_started_at: float | None = None
    last_completed_stage: PipelineStage | None = None
    stage_durations: list[tuple[PipelineStage, float]] | None = None
    bounded_path_count: int | None = None
    feasible_path_count: int | None = None
    infeasible_path_count: int | None = None
    unknown_path_count: int | None = None
    candidate_generated_path_count: int | None = None
    input_generation_accepted_count: int | None = None
    input_generation_rejected_count: int | None = None
    scenario_generation_accepted_count: int | None = None
    scenario_generation_rejected_count: int | None = None
    pre_concrete_scenario_count: int | None = None
    concrete_validation_accepted_count: int | None = None
    concrete_validation_rejected_count: int | None = None
    final_scenario_count: int | None = None
    rl_executed_test_count: int | None = None
    q_table_state_count: int | None = None
    scenario_rejection_counts: tuple[tuple[str, int], ...] = ()
    concrete_rejection_counts: tuple[tuple[str, int], ...] = ()
    line_coverage_percent: float | None = None
    branch_coverage_percent: float | None = None
    reachability_counts: tuple[tuple[str, int], ...] = ()
    checkpoint_store: PipelineDiagnosticCheckpointStore | None = None

    def __post_init__(self) -> None:
        if self.stage_durations is None:
            self.stage_durations = []

    def start_stage(self, stage: PipelineStage) -> float:
        self.current_stage = stage
        self.current_stage_started_at = time.perf_counter()
        self.publish()
        return self.current_stage_started_at

    def complete_stage(self, stage: PipelineStage, started_at: float) -> None:
        duration = max(0.0, time.perf_counter() - started_at)
        assert self.stage_durations is not None
        self.stage_durations.append((stage, duration))
        self.last_completed_stage = stage
        self.current_stage = None
        self.current_stage_started_at = None
        self.publish()

    def publish(self) -> None:
        """Son doğrulanmış snapshot'ı varsa durable store'a yazar."""
        if self.checkpoint_store is None:
            return
        self.checkpoint_store.write(
            self.build(status=PipelineRunStatus.PARTIAL)
        )

    def build(
        self,
        *,
        status: PipelineRunStatus,
        error: Exception | None = None,
    ) -> PipelineDiagnosticResult:
        scenario_rejections = tuple(sorted(self.scenario_rejection_counts))
        concrete_rejections = tuple(sorted(self.concrete_rejection_counts))
        reachability_counts = tuple(sorted(self.reachability_counts))
        stage_durations = list(self.stage_durations or ())
        if (
            self.current_stage is not None
            and self.current_stage_started_at is not None
        ):
            stage_durations.append(
                (
                    self.current_stage,
                    max(
                        0.0,
                        time.perf_counter()
                        - self.current_stage_started_at,
                    ),
                )
            )

        controlled_error = isinstance(error, _ControlledPipelineFailure)
        return PipelineDiagnosticResult(
            status=status,
            source_file=self.source_file,
            function_name=self.function_name,
            last_completed_stage=self.last_completed_stage,
            stopped_stage=(
                self.current_stage
                if status is not PipelineRunStatus.COMPLETED
                else None
            ),
            error_category=(
                "CONTROLLED_FAILURE"
                if controlled_error
                else "UNEXPECTED_FAILURE"
                if error is not None
                else None
            ),
            error_message=(
                (str(error).strip() or type(error).__name__)
                if controlled_error
                else "Beklenmeyen pipeline hatası."
                if error is not None
                else None
            ),
            exception_type=(type(error).__name__ if error is not None else None),
            total_duration_seconds=max(
                0.0, time.perf_counter() - self.started_at
            ),
            stage_durations=tuple(stage_durations),
            funnel=PipelineFunnelSnapshot(
                bounded_path_count=self.bounded_path_count,
                feasible_path_count=self.feasible_path_count,
                infeasible_path_count=self.infeasible_path_count,
                unknown_path_count=self.unknown_path_count,
                candidate_generated_path_count=(
                    self.candidate_generated_path_count
                ),
                input_generation_accepted_count=(
                    self.input_generation_accepted_count
                ),
                input_generation_rejected_count=(
                    self.input_generation_rejected_count
                ),
                scenario_generation_accepted_count=(
                    self.scenario_generation_accepted_count
                ),
                scenario_generation_rejected_count=(
                    self.scenario_generation_rejected_count
                ),
                pre_concrete_scenario_count=self.pre_concrete_scenario_count,
                concrete_validation_accepted_count=(
                    self.concrete_validation_accepted_count
                ),
                concrete_validation_rejected_count=(
                    self.concrete_validation_rejected_count
                ),
                final_scenario_count=self.final_scenario_count,
                rl_executed_test_count=self.rl_executed_test_count,
                q_table_state_count=self.q_table_state_count,
            ),
            scenario_rejection_counts=scenario_rejections,
            concrete_rejection_counts=concrete_rejections,
            line_coverage_percent=self.line_coverage_percent,
            branch_coverage_percent=self.branch_coverage_percent,
            reachability_counts=reachability_counts,
        )


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
        diagnostic_checkpoint_path: Path | None = None,
        global_timeout_runner: GlobalPipelineTimeoutRunner | None = None,
        scenario_minimization_service: (
            ScenarioCoverageMinimizationService | None
        ) = None,
        strategy_comparison_service: StrategyComparisonService | None = None,
    ) -> None:
        """Servisin analiz ve raporlama bağımlılıklarını hazırlar."""
        self._supports_process_isolation = all(
            dependency is None
            for dependency in (
                analyzer,
                cfg_builder,
                path_analyzer,
                data_flow_analyzer,
                path_state_analyzer,
                path_feasibility_analyzer,
                input_candidate_generator,
                dqm,
                scenario_generator,
                report_formatter,
                coverage_reachability_service,
                scenario_minimization_service,
                strategy_comparison_service,
            )
        )
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
        self._scenario_minimization_service = (
            scenario_minimization_service
            or ScenarioCoverageMinimizationService()
        )
        self._strategy_comparison_service = (
            strategy_comparison_service or StrategyComparisonService()
        )
        self._active_diagnostic_accumulator: (
            _PipelineDiagnosticAccumulator | None
        ) = None
        self._last_diagnostic_result: PipelineDiagnosticResult | None = None
        self._diagnostic_checkpoint_store = (
            PipelineDiagnosticCheckpointStore(diagnostic_checkpoint_path)
            if diagnostic_checkpoint_path is not None
            else None
        )
        self._global_timeout_runner = (
            global_timeout_runner or GlobalPipelineTimeoutRunner()
        )

    @property
    def last_diagnostic_result(self) -> PipelineDiagnosticResult | None:
        """Son run'a ait immutable diagnostic sonucu döndürür."""
        return self._last_diagnostic_result

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
        pipeline_timeout_seconds: float | None = None,
        run_greedy_baseline: bool = False,
        greedy_timeout_seconds: float | None = None,
        run_strategy_comparison: bool = False,
        comparison_timeout_seconds: float | None = None,
        import_root: str | Path | None = None,
    ) -> RealRLTrainingResult | PipelineDiagnosticResult:
        """Mevcut exception davranışını koruyarak production run çalıştırır."""
        self._active_diagnostic_accumulator = None
        self._last_diagnostic_result = None
        self._validate_pipeline_timeout(pipeline_timeout_seconds)
        if not isinstance(run_greedy_baseline, bool):
            raise TypeError("run_greedy_baseline bool olmalıdır.")
        if not isinstance(run_strategy_comparison, bool):
            raise TypeError("run_strategy_comparison bool olmalıdır.")
        self._validate_optional_timeout(
            greedy_timeout_seconds,
            "greedy_timeout_seconds",
        )
        self._validate_optional_timeout(
            comparison_timeout_seconds,
            "comparison_timeout_seconds",
        )

        if pipeline_timeout_seconds is not None:
            if not self._supports_process_isolation:
                raise ValueError(
                    "Global pipeline timeout yalnız varsayılan production "
                    "bağımlılıklarıyla kullanılabilir."
                )
            result = self._global_timeout_runner.run(
                run_arguments={
                    "source_file": source_file,
                    "module_path": module_path,
                    "function_name": function_name,
                    "output_directory": output_directory,
                    "max_visits_per_node": max_visits_per_node,
                    "episode_count": episode_count,
                    "epsilon": epsilon,
                    "epsilon_decay_rate": epsilon_decay_rate,
                    "minimum_epsilon": minimum_epsilon,
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "random_seed": random_seed,
                    "overwrite": overwrite,
                    "timeout_seconds": timeout_seconds,
                    "pipeline_timeout_seconds": None,
                    "run_greedy_baseline": run_greedy_baseline,
                    "greedy_timeout_seconds": greedy_timeout_seconds,
                    "run_strategy_comparison": run_strategy_comparison,
                    "comparison_timeout_seconds": comparison_timeout_seconds,
                    "import_root": import_root,
                },
                source_file=Path(source_file).resolve(),
                function_name=function_name,
                timeout_seconds=pipeline_timeout_seconds,
            )
            if isinstance(result, PipelineDiagnosticResult):
                result = replace(
                    result,
                    pipeline_timeout_seconds=pipeline_timeout_seconds,
                )
                self._last_diagnostic_result = result
            else:
                if result.diagnostic is not None:
                    result = replace(
                        result,
                        diagnostic=replace(
                            result.diagnostic,
                            pipeline_timeout_seconds=(
                                pipeline_timeout_seconds
                            ),
                        ),
                    )
                self._last_diagnostic_result = result.diagnostic
            return result

        try:
            result = self._run_pipeline(
                source_file=source_file,
                module_path=module_path,
                function_name=function_name,
                output_directory=output_directory,
                max_visits_per_node=max_visits_per_node,
                episode_count=episode_count,
                epsilon=epsilon,
                epsilon_decay_rate=epsilon_decay_rate,
                minimum_epsilon=minimum_epsilon,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                random_seed=random_seed,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
                run_greedy_baseline=run_greedy_baseline,
                greedy_timeout_seconds=greedy_timeout_seconds,
                run_strategy_comparison=run_strategy_comparison,
                comparison_timeout_seconds=comparison_timeout_seconds,
                import_root=import_root,
            )
        except Exception as error:
            accumulator = self._active_diagnostic_accumulator
            if accumulator is not None:
                self._last_diagnostic_result = accumulator.build(
                    status=(
                        PipelineRunStatus.PARTIAL
                        if accumulator.last_completed_stage is not None
                        else PipelineRunStatus.FAILED
                    ),
                    error=error,
                )
                self._write_diagnostic_checkpoint(
                    self._last_diagnostic_result
                )
            raise

        self._last_diagnostic_result = result.diagnostic
        return result

    def run_with_diagnostics(
        self,
        **run_arguments: Any,
    ) -> RealRLTrainingResult | PipelineDiagnosticResult:
        """Kontrollü domain duruşlarında ara diagnostic sonucu döndürür."""
        try:
            return self.run(**run_arguments)
        except _ControlledPipelineFailure:
            if self._last_diagnostic_result is None:
                raise RuntimeError(
                    "Kontrollü pipeline duruşu diagnostic sonuç üretmedi."
                )
            return self._last_diagnostic_result

    def _run_pipeline(
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
        run_greedy_baseline: bool = False,
        greedy_timeout_seconds: float | None = None,
        run_strategy_comparison: bool = False,
        comparison_timeout_seconds: float | None = None,
        import_root: str | Path | None = None,
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

            run_greedy_baseline:
                True olduğunda RL'den bağımsız exact-coverage greedy
                baseline çalıştırılır. Varsayılan False eski akışı korur.

            greedy_timeout_seconds:
                Yalnız minimization aşamasının toplam süre sınırı.

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
        normalized_import_root = self._normalize_import_root(
            import_root,
            normalized_source_file,
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

        diagnostic = _PipelineDiagnosticAccumulator(
            source_file=normalized_source_file,
            function_name=normalized_function_name,
            started_at=time.perf_counter(),
            checkpoint_store=self._diagnostic_checkpoint_store,
        )
        self._active_diagnostic_accumulator = diagnostic

        stage_started = diagnostic.start_stage(
            PipelineStage.SOURCE_ANALYSIS
        )
        analysis_result = self._analyzer.analyze_file(
            normalized_source_file
        )
        diagnostic.complete_stage(
            PipelineStage.SOURCE_ANALYSIS, stage_started
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.CFG_CONSTRUCTION
        )
        graphs = self._cfg_builder.build_from_file(
            normalized_source_file
        )
        diagnostic.complete_stage(
            PipelineStage.CFG_CONSTRUCTION, stage_started
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.FUNCTION_DISCOVERY
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
            raise _ControlledPipelineFailure(
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
            raise _ControlledPipelineFailure(
                "Fonksiyon için Control Flow Graph bulunamadı: "
                f"{normalized_function_name}"
            )

        diagnostic.complete_stage(
            PipelineStage.FUNCTION_DISCOVERY, stage_started
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.PATH_DISCOVERY
        )
        paths = self._path_analyzer.find_paths(
            graph,
            max_visits_per_node=max_visits_per_node,
        )

        if not paths:
            raise _ControlledPipelineFailure(
                "Fonksiyon için yürütme yolu bulunamadı: "
                f"{normalized_function_name}"
            )

        diagnostic.bounded_path_count = len(paths)
        diagnostic.complete_stage(
            PipelineStage.PATH_DISCOVERY, stage_started
        )

        parameter_names = self._extract_parameter_names(
            function.parameters
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.PATH_FEASIBILITY
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

        diagnostic.feasible_path_count = sum(
            result.status is FeasibilityStatus.FEASIBLE
            for result in feasibility_results
        )
        diagnostic.infeasible_path_count = sum(
            result.status is FeasibilityStatus.INFEASIBLE
            for result in feasibility_results
        )
        diagnostic.unknown_path_count = sum(
            result.status is FeasibilityStatus.UNKNOWN
            for result in feasibility_results
        )
        diagnostic.complete_stage(
            PipelineStage.PATH_FEASIBILITY, stage_started
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
            raise _ControlledPipelineFailure(
                "Fonksiyon için FEASIBLE veya UNKNOWN "
                "yürütme yolu bulunamadı: "
                f"{normalized_function_name}"
            )

        stage_started = diagnostic.start_stage(
            PipelineStage.CANDIDATE_GENERATION
        )
        candidate_values_by_path = (
            self._build_candidate_values_by_path(
                paths=paths,
                feasibility_results=feasibility_results,
                path_states=path_states,
                data_flow_result=data_flow_result,
            )
        )
        diagnostic.candidate_generated_path_count = len(
            candidate_values_by_path
        )
        diagnostic.complete_stage(
            PipelineStage.CANDIDATE_GENERATION, stage_started
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
            raise _ControlledPipelineFailure(
                "Fonksiyon için uygulanabilir DQM sonucu "
                "üretilemedi: "
                f"{normalized_function_name}"
            )

        stage_started = diagnostic.start_stage(
            PipelineStage.SCENARIO_GENERATION
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

        scenario_rejections = getattr(
            self._scenario_generator, "rejections", ()
        )
        if not isinstance(scenario_rejections, tuple):
            scenario_rejections = ()
        rejection_counter = Counter(
            rejection.category.value for rejection in scenario_rejections
        )
        diagnostic.scenario_rejection_counts = tuple(
            sorted(rejection_counter.items())
        )
        diagnostic.input_generation_accepted_count = len(scenarios)
        diagnostic.input_generation_rejected_count = len(
            scenario_rejections
        )
        diagnostic.scenario_generation_accepted_count = len(scenarios)
        diagnostic.scenario_generation_rejected_count = len(
            scenario_rejections
        )
        diagnostic.pre_concrete_scenario_count = len(scenarios)

        if not scenarios:
            raise _ControlledPipelineFailure(
                "Fonksiyon için test senaryosu üretilemedi: "
                f"{normalized_function_name}"
            )

        diagnostic.complete_stage(
            PipelineStage.SCENARIO_GENERATION, stage_started
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.CONCRETE_VALIDATION
        )
        scenario_tuple = self._filter_executable_scenarios(
            source_file=normalized_source_file,
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=tuple(scenarios),
            import_root=normalized_import_root,
        )

        concrete_rejected_count = len(scenarios) - len(scenario_tuple)
        diagnostic.concrete_validation_accepted_count = len(scenario_tuple)
        diagnostic.concrete_validation_rejected_count = (
            concrete_rejected_count
        )
        diagnostic.concrete_rejection_counts = (
            (("EXECUTION_MISMATCH", concrete_rejected_count),)
            if concrete_rejected_count
            else ()
        )
        diagnostic.final_scenario_count = len(scenario_tuple)

        if not scenario_tuple:
            raise _ControlledPipelineFailure(
                "Üretilen senaryoların hiçbiri hedef fonksiyonda "
                "beklenen sonucu üretmedi: "
                f"{normalized_function_name}"
            )

        diagnostic.complete_stage(
            PipelineStage.CONCRETE_VALIDATION, stage_started
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
            import_root=normalized_import_root,
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.COVERAGE_MEASUREMENT
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

        diagnostic.line_coverage_percent = (
            baseline_coverage_result.line_coverage_percent
        )

        minimization_result = None
        if run_greedy_baseline or run_strategy_comparison:
            minimization_result = self._scenario_minimization_service.minimize(
                source_file=normalized_source_file,
                module_path=normalized_module_path,
                function_name=normalized_function_name,
                function_start_line=function.line_number,
                function_end_line=function.end_line_number,
                scenarios=scenario_tuple,
                output_root=normalized_output_directory,
                timeout_seconds=timeout_seconds,
                minimization_timeout_seconds=greedy_timeout_seconds,
                full_pool_coverage=baseline_coverage_result,
                import_root=normalized_import_root,
            )
        diagnostic.branch_coverage_percent = (
            baseline_coverage_result.branch_coverage_percent
        )
        diagnostic.reachability_counts = (
            (
                "COVERED",
                len(reachability_result.covered_lines),
            ),
            (
                "FEASIBLE_UNCOVERED",
                len(reachability_result.feasible_uncovered_lines),
            ),
            (
                "INFEASIBLE_ONLY",
                len(reachability_result.infeasible_only_lines),
            ),
            (
                "UNRESOLVED",
                len(reachability_result.unresolved_lines),
            ),
        )
        diagnostic.complete_stage(
            PipelineStage.COVERAGE_MEASUREMENT, stage_started
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

        comparison_hyperparameters = (
            ("episode_count", episode_count),
            ("epsilon", epsilon),
            ("epsilon_decay_rate", epsilon_decay_rate),
            ("minimum_epsilon", minimum_epsilon),
            ("learning_rate", learning_rate),
            ("discount_factor", discount_factor),
            ("target_coverage_policy", "VALIDATED_POOL_LINE_PERCENTAGE"),
        )
        if run_strategy_comparison:
            if minimization_result is None:
                raise RuntimeError("Strategy comparison greedy baseline gerektirir.")
            self._strategy_comparison_service.write_pending(
                source_file=normalized_source_file,
                function_name=normalized_function_name,
                scenarios=scenario_tuple,
                full_pool_coverage=baseline_coverage_result,
                greedy_result=minimization_result,
                requested_rl_episode_count=episode_count,
                output_root=normalized_output_directory,
                rl_hyperparameters=comparison_hyperparameters,
                random_seed=random_seed,
            )

        episode_coverage_results: dict[
            int, CoverageResult | FunctionCoverageResult
        ] = {}

        def episode_completed_callback(episode: object) -> None:
            episode_number = getattr(episode, "episode_number", None)
            episode_coverage = suite_transition.last_coverage_result
            if isinstance(episode_number, int) and episode_coverage is not None:
                episode_coverage_results[episode_number] = episode_coverage
            if run_strategy_comparison:
                progress_session = TrainingSessionResult(
                    episodes=statistics.episodes,
                    requested_episode_count=episode_count,
                    completed_episode_count=len(statistics.episodes),
                )
                self._strategy_comparison_service.write_pending(
                    source_file=normalized_source_file,
                    function_name=normalized_function_name,
                    scenarios=scenario_tuple,
                    full_pool_coverage=baseline_coverage_result,
                    greedy_result=minimization_result,
                    requested_rl_episode_count=episode_count,
                    output_root=normalized_output_directory,
                    rl_hyperparameters=comparison_hyperparameters,
                    random_seed=random_seed,
                    session_result=progress_session,
                )

        stage_started = diagnostic.start_stage(
            PipelineStage.RL_TRAINING
        )
        session_result = session.run(
            environment=environment,
            episode_count=episode_count,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
            episode_completed_callback=episode_completed_callback,
        )

        best_episode = session_result.best_episode
        final_coverage_result = episode_coverage_results.get(
            best_episode.episode_number
            if best_episode is not None
            else -1,
            suite_transition.last_coverage_result,
        )

        if final_coverage_result is None:
            raise RuntimeError(
                "RL eğitimi tamamlandı ancak son coverage sonucu "
                "bulunamadı."
            )

        diagnostic.rl_executed_test_count = (
            statistics.best_executed_test_count
        )

        strategy_comparison_result = None
        if run_strategy_comparison:
            strategy_comparison_result = self._strategy_comparison_service.compare(
                source_file=normalized_source_file,
                module_path=normalized_module_path,
                function_name=normalized_function_name,
                function_start_line=function.line_number,
                function_end_line=function.end_line_number,
                scenarios=scenario_tuple,
                full_pool_coverage=baseline_coverage_result,
                greedy_result=minimization_result,
                session_result=session_result,
                output_root=normalized_output_directory,
                timeout_seconds=timeout_seconds,
                comparison_timeout_seconds=comparison_timeout_seconds,
                rl_hyperparameters=comparison_hyperparameters,
                random_seed=random_seed,
                import_root=normalized_import_root,
            )
        diagnostic.q_table_state_count = len(q_table)
        diagnostic.complete_stage(
            PipelineStage.RL_TRAINING, stage_started
        )

        stage_started = diagnostic.start_stage(
            PipelineStage.REPORTING
        )
        report = self._report_formatter.format_session(
            result=session_result,
            statistics=statistics,
            function_name=normalized_function_name,
            coverage_result=final_coverage_result,
            reachability_result=reachability_result,
        )
        if strategy_comparison_result is not None:
            report = (
                f"{report}\n\n{strategy_comparison_result.format_summary()}"
            )
        diagnostic.complete_stage(
            PipelineStage.REPORTING, stage_started
        )

        diagnostic_result = diagnostic.build(
            status=PipelineRunStatus.COMPLETED
        )
        self._write_diagnostic_checkpoint(diagnostic_result)

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
            diagnostic=diagnostic_result,
            minimization_result=minimization_result,
            strategy_comparison_result=strategy_comparison_result,
            scenario_pool_coverage_result=baseline_coverage_result,
        )

    def _write_diagnostic_checkpoint(
        self,
        diagnostic: PipelineDiagnosticResult,
    ) -> None:
        if self._diagnostic_checkpoint_store is not None:
            self._diagnostic_checkpoint_store.write(diagnostic)

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
        module_path: str | None = None,
        function_name: str,
        scenarios: tuple[Scenario, ...],
        import_root: Path | None = None,
    ) -> tuple[Scenario, ...]:
        """
        Senaryoları hedef fonksiyonda somut olarak çalıştırır.

        Üretilen girdinin gerçek sonucu, senaryoda beklenen sonuçla
        uyuşmuyorsa senaryo ulaşılamaz veya yanlış modellenmiş kabul
        edilir ve RL aksiyon kümesine eklenmez.
        """
        previous_path = tuple(sys.path)
        previous_modules = frozenset(sys.modules)
        if import_root is not None:
            sys.path.insert(0, str(import_root))
        try:
            target_function = self._load_target_function(
                source_file=source_file,
                module_path=(
                    module_path
                    if import_root is not None and module_path is not None
                    else None
                ),
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
        finally:
            if import_root is not None:
                sys.path[:] = previous_path
                for imported_name in tuple(set(sys.modules) - previous_modules):
                    sys.modules.pop(imported_name, None)

    @staticmethod
    def _load_target_function(
        *,
        source_file: Path,
        module_path: str | None,
        function_name: str,
    ) -> Callable[..., Any]:
        """
        Kaynak dosyayı izole bir modül adıyla yükler ve hedef
        fonksiyonu döndürür.
        """
        if module_path is not None:
            module = importlib.import_module(module_path)
        else:
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
    def _normalize_import_root(
        import_root: str | Path | None,
        source_file: Path,
    ) -> Path | None:
        """Harici modül kökünü parent interpreter'a eklemeden doğrular."""
        if import_root is None:
            return None
        if not isinstance(import_root, (str, Path)):
            raise TypeError("import_root string veya Path olmalıdır.")
        if isinstance(import_root, str) and not import_root.strip():
            raise ValueError("import_root boş olamaz.")
        normalized = Path(import_root).resolve()
        if not normalized.is_dir():
            raise ValueError("import_root var olan bir klasör olmalıdır.")
        if not source_file.is_relative_to(normalized):
            raise ValueError("source_file import_root dışında olamaz.")
        return normalized

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

    @staticmethod
    def _validate_pipeline_timeout(
        pipeline_timeout_seconds: float | None,
    ) -> None:
        if pipeline_timeout_seconds is None:
            return
        if (
            isinstance(pipeline_timeout_seconds, bool)
            or not isinstance(pipeline_timeout_seconds, (int, float))
        ):
            raise TypeError(
                "pipeline_timeout_seconds sayısal veya None olmalıdır."
            )
        normalized_timeout = float(pipeline_timeout_seconds)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0.0:
            raise ValueError(
                "pipeline_timeout_seconds sonlu ve sıfırdan büyük olmalıdır."
            )

    @staticmethod
    def _validate_optional_timeout(value: float | None, name: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal veya None olmalıdır.")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} sonlu ve sıfırdan büyük olmalıdır.")
