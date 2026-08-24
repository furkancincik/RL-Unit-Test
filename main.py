import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from analyzer.python_analyzer import AnalysisResult, PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from reports.json_reporter import JSONReportWriter
from services.project_analysis_report_service import (
    ProjectAnalysisReportFormatter,
)
from services.automation_service import (
    AutomationService,
    AutomationSummary,
)
from services.coverage_workflow_service import (
    CoverageWorkflowService,
    CoverageWorkflowSummary,
)
from services.rl_demo_service import RLDemoService
from services.real_rl_training_service import (
    RealRLTrainingResult,
    RealRLTrainingService,
)
from models.project_analysis_result import (
    ProjectAnalysisResult,
    ProjectRunStatus,
    validate_qualified_target_name,
)
from services.source_analysis_orchestrator import (
    SourceAnalysisOrchestrator,
    SourceAnalysisValidationError,
)
from services.external_source_terminal_service import ExternalSourceTerminalAdapter


SOURCE_FILE = "datasets/sample_code.py"
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


GENERATED_TEST_DIRECTORY = Path("output/generated_tests")


CLI_OPERATIONS = (
    "menu",
    "analyze",
    "cfg",
    "dqm",
    "dqm-json",
    "test",
    "coverage",
    "demo",
    "rl",
)


@dataclass(frozen=True, slots=True)
class ApplicationConfiguration:
    """Komut satırından alınan hedef ve çalışma ayarlarını taşır."""

    source_file: Path | None
    module_path: str | None
    function_name: str | None
    output_directory: Path
    operation: str = "menu"
    all_functions: bool = False
    max_visits_per_node: int = 3
    episode_count: int = 3
    epsilon: float = 0.0
    learning_rate: float = 0.5
    discount_factor: float = 0.9
    random_seed: int = 42
    overwrite: bool = True
    timeout_seconds: float = 30.0
    pipeline_timeout_seconds: float | None = None


def _python_source_file(value: str) -> Path:
    """Var olan bir Python kaynak dosyası argümanı doğrular."""
    source_file = Path(value)

    if source_file.suffix.lower() != ".py":
        raise argparse.ArgumentTypeError(
            "source-file bir .py dosyası olmalıdır."
        )

    if not source_file.is_file():
        raise argparse.ArgumentTypeError(
            f"Python kaynak dosyası bulunamadı: {source_file}"
        )

    return source_file


def _python_identifier(value: str) -> str:
    """Python tanımlayıcısı olması gereken argümanı doğrular."""
    normalized_value = value.strip()

    if not normalized_value.isidentifier():
        raise argparse.ArgumentTypeError(
            f"Geçersiz Python tanımlayıcısı: {value!r}"
        )

    return normalized_value


def _qualified_target_name(value: str) -> str:
    """Top-level function veya exact ``Class.method`` hedefini doğrular."""
    try:
        return validate_qualified_target_name(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"Geçersiz qualified target: {value!r}"
        ) from error


def _python_module_path(value: str) -> str:
    """Nokta ayrımlı Python modül yolunu doğrular."""
    normalized_value = value.strip()
    path_parts = normalized_value.split(".")

    if (
        not normalized_value
        or any(
            not path_part.isidentifier()
            for path_part in path_parts
        )
    ):
        raise argparse.ArgumentTypeError(
            f"Geçersiz Python modül yolu: {value!r}"
        )

    return normalized_value


def _positive_int(value: str) -> int:
    """Sıfırdan büyük tam sayı argümanı doğrular."""
    try:
        normalized_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Pozitif tam sayı bekleniyordu: {value!r}"
        ) from error

    if normalized_value < 1:
        raise argparse.ArgumentTypeError(
            f"Değer 1 veya daha büyük olmalıdır: {value!r}"
        )

    return normalized_value


def _positive_float(value: str) -> float:
    """Sıfırdan büyük ondalık sayı argümanı doğrular."""
    try:
        normalized_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Pozitif sayı bekleniyordu: {value!r}"
        ) from error

    if not math.isfinite(normalized_value) or normalized_value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"Değer sıfırdan büyük olmalıdır: {value!r}"
        )

    return normalized_value


def _probability(value: str) -> float:
    """0.0 ile 1.0 arasındaki oran argümanını doğrular."""
    try:
        normalized_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Oran sayısal olmalıdır: {value!r}"
        ) from error

    if (
        not math.isfinite(normalized_value)
        or not 0.0 <= normalized_value <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            f"Oran 0.0 ile 1.0 arasında olmalıdır: {value!r}"
        )

    return normalized_value


def create_argument_parser() -> argparse.ArgumentParser:
    """Komut satırı argüman ayrıştırıcısını oluşturur."""
    parser = argparse.ArgumentParser(
        description=(
            "Python kaynak kodu için analiz, otomatik test üretimi "
            "ve gerçek coverage tabanlı RL eğitimi çalıştırır."
        ),
    )

    parser.add_argument(
        "--source-file",
        type=_python_source_file,
        default=None,
        help=(
            "Analiz edilecek Python dosyası. Production RL işlemi için "
            "açıkça verilmelidir."
        ),
    )
    parser.add_argument(
        "--module-path",
        type=_python_module_path,
        default=None,
        help=(
            "Kaynak dosyanın import edilebilir modül yolu. Production RL "
            "işlemi için açıkça verilmelidir."
        ),
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--function-name",
        type=_qualified_target_name,
        default=None,
        help=(
            "RL eğitimi uygulanacak top-level function veya Class.method hedefi. "
            "Production RL işleminde "
            "--all-functions kullanılmıyorsa açıkça verilmelidir."
        ),
    )
    target_group.add_argument(
        "--all-functions",
        action="store_true",
        help="Kaynak dosyadaki bütün keşfedilmiş fonksiyonları raporlar.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=GENERATED_TEST_DIRECTORY,
        help=(
            "Test ve rapor çıktılarının yazılacağı klasör "
            f"(varsayılan: {GENERATED_TEST_DIRECTORY})."
        ),
    )
    parser.add_argument(
        "--operation",
        choices=CLI_OPERATIONS,
        default="menu",
        help=(
            "Çalıştırılacak işlem. 'menu' interaktif menüyü açar; "
            "diğer değerler seçilen işlemi doğrudan çalıştırır."
        ),
    )
    parser.add_argument(
        "--max-visits-per-node",
        type=_positive_int,
        default=3,
        help="Bir CFG düğümünün yol başına azami ziyaret sayısı.",
    )
    parser.add_argument(
        "--episode-count",
        type=_positive_int,
        default=3,
        help="Gerçek RL eğitimi episode sayısı.",
    )
    parser.add_argument(
        "--epsilon",
        type=_probability,
        default=0.0,
        help="Epsilon-greedy başlangıç keşif oranı.",
    )
    parser.add_argument(
        "--learning-rate",
        type=_probability,
        default=0.5,
        help="Q-Learning öğrenme oranı.",
    )
    parser.add_argument(
        "--discount-factor",
        type=_probability,
        default=0.9,
        help="Q-Learning gelecek ödül iskonto oranı.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Tekrarlanabilir RL seçimi için rastgelelik tohumu.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="Her test veya coverage çalıştırmasının süre sınırı.",
    )
    parser.add_argument(
        "--pipeline-timeout-seconds",
        type=_positive_float,
        default=None,
        help="Her fonksiyonun production pipeline süre sınırı.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help=(
            "Mevcut çıktı dosyalarının üzerine yazılmasını "
            "engeller."
        ),
    )

    return parser


def parse_cli_arguments(
    argv: Sequence[str] | None = None,
) -> ApplicationConfiguration:
    """CLI argümanlarını çalışma yapılandırmasına dönüştürür."""
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)
    source_file = arguments.source_file
    module_path = arguments.module_path
    function_name = arguments.function_name

    if arguments.operation == "rl":
        if source_file is None:
            parser.error("--operation rl için --source-file zorunludur.")
        if module_path is None:
            parser.error("--operation rl için --module-path zorunludur.")
        if function_name is None and not arguments.all_functions:
            parser.error(
                "--operation rl için --function-name veya --all-functions zorunludur."
            )
    elif arguments.operation != "menu":
        source_file = source_file or Path(SOURCE_FILE)
        module_path = module_path or MODULE_PATH
        if function_name is None and not arguments.all_functions:
            function_name = FUNCTION_NAME

    return ApplicationConfiguration(
        source_file=source_file,
        module_path=module_path,
        function_name=function_name,
        output_directory=arguments.output_directory,
        operation=arguments.operation,
        all_functions=arguments.all_functions,
        max_visits_per_node=arguments.max_visits_per_node,
        episode_count=arguments.episode_count,
        epsilon=arguments.epsilon,
        learning_rate=arguments.learning_rate,
        discount_factor=arguments.discount_factor,
        random_seed=arguments.random_seed,
        overwrite=arguments.overwrite,
        timeout_seconds=arguments.timeout_seconds,
        pipeline_timeout_seconds=arguments.pipeline_timeout_seconds,
    )


def print_analyzer_report(
    source_file: str | Path = SOURCE_FILE,
) -> None:
    """Seçilen Python dosyası için statik analiz raporu üretir."""
    analyzer = PythonAnalyzer()
    result = analyzer.analyze_file(source_file)

    print("=" * 55)
    print("PYTHON DOSYA ANALİZ RAPORU")
    print("=" * 55)

    print(f"Dosya Adı                  : {result.file_name}")
    print(f"Fonksiyon Sayısı           : {result.function_count}")
    print(f"Sınıf Sayısı               : {result.class_count}")
    print(f"If Sayısı                  : {result.if_count}")
    print(f"For Sayısı                 : {result.for_count}")
    print(f"While Sayısı               : {result.while_count}")
    print(f"Try Sayısı                 : {result.try_count}")
    print(f"Return Sayısı              : {result.return_count}")

    print("\n" + "=" * 55)
    print("FONKSİYON DETAYLARI")
    print("=" * 55)

    if not result.functions:
        print("\nDosyada analiz edilebilir fonksiyon bulunamadı.")
        return

    for index, function in enumerate(result.functions, start=1):
        print(f"\nFonksiyon #{index}")
        print("-" * 55)
        print(f"Fonksiyon Adı              : {function.name}")
        print(f"Qualified Adı              : {function.qualified_name}")
        print(
            "Production Desteği         : "
            f"{'Destekleniyor' if function.is_supported else 'Desteklenmiyor'}"
        )
        if function.unsupported_reason is not None:
            print(f"Desteklenmeme Nedeni       : {function.unsupported_reason}")
        print(f"Parametreler               : {function.parameters}")
        print(f"Satır Numarası             : {function.line_number}")
        print(f"Async Fonksiyon mu?        : {function.is_async}")
        print(f"Return Sayısı              : {function.return_count}")
        print(f"Karar Noktası Sayısı       : {function.branch_count}")
        print(
            f"Cyclomatic Complexity      : "
            f"{function.cyclomatic_complexity}"
        )
        print(f"Risk Seviyesi              : {function.risk_level}")
        print(f"Docstring Var mı?          : {function.has_docstring}")
        print(
            f"Dönüş Type Hint Var mı?    : "
            f"{function.has_return_type_hint}"
        )
        print(
            f"Tip Belirtilen Parametre   : "
            f"{function.typed_parameter_count}"
        )


def print_cfg_report(
    source_file: str | Path = SOURCE_FILE,
) -> None:
    """Seçilen Python dosyasındaki fonksiyonlar için CFG raporu üretir."""
    builder = ControlFlowGraphBuilder()
    graphs = builder.build_from_file(source_file)

    if not graphs:
        print("Dosyada CFG üretilebilecek fonksiyon bulunamadı.")
        return

    for graph in graphs:
        print("=" * 60)
        print(f"CONTROL FLOW GRAPH: {graph.function_name}")
        print("=" * 60)

        print("\nDÜĞÜMLER")
        print("-" * 60)

        for node in graph.nodes:
            print(
                f"ID={node.node_id} | "
                f"Tür={node.node_type} | "
                f"Satır={node.line_number} | "
                f"Etiket={node.label}"
            )

        print("\nBAĞLANTILAR")
        print("-" * 60)

        for edge in graph.edges:
            edge_label = edge.label or "-"

            print(
                f"{edge.source_id} -> {edge.target_id} "
                f"| Yol={edge_label}"
            )


def print_dqm_report(
    source_file: str | Path = SOURCE_FILE,
    *,
    max_visits_per_node: int = 3,
) -> None:
    """Yürütme yolları için DQM önceliklendirme raporu üretir."""
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()

    analysis_result = analyzer.analyze_file(source_file)
    graphs = cfg_builder.build_from_file(source_file)

    if not analysis_result.functions or not graphs:
        print("DQM değerlendirmesi için fonksiyon bulunamadı.")
        return

    for function, graph in zip(
        analysis_result.functions,
        graphs,
        strict=True,
    ):
        paths = path_analyzer.find_paths(
            graph,
            max_visits_per_node=max_visits_per_node,
        )

        scores = dqm.evaluate_paths(
            function=function,
            paths=paths,
        )

        print("=" * 65)
        print(f"DQM RAPORU: {function.name}")
        print("=" * 65)

        print(
            f"Cyclomatic Complexity : "
            f"{function.cyclomatic_complexity}"
        )
        print(f"Risk Seviyesi         : {function.risk_level}")
        print(f"Yürütme Yolu Sayısı   : {len(paths)}")

        if not scores:
            print("\nDeğerlendirilebilecek yürütme yolu bulunamadı.")
            continue

        print("\nYOL ÖNCELİKLERİ")
        print("-" * 65)

        for rank, score in enumerate(scores, start=1):
            original_path = paths[score.path_index - 1]

            print(f"\nÖncelik Sırası #{rank}")
            print(f"Yol Numarası           : {score.path_index}")
            print(f"Düğüm Yolu             : {original_path.node_ids}")
            print(f"Kenar Etiketleri       : {original_path.edge_labels}")
            print(f"Yol Uzunluğu           : {score.path_length}")
            print(
                f"Karar Kenarı Sayısı    : "
                f"{score.decision_edge_count}"
            )
            print(f"Döngü İçeriyor mu?     : {score.contains_loop}")
            print(
                f"İstisna İçeriyor mu?   : "
                f"{score.contains_exception}"
            )
            print(f"Ham DQM Skoru          : {score.raw_score}")
            print(
                f"Normalize DQM Skoru    : "
                f"{score.normalized_score}"
            )
            print(f"Öncelik Seviyesi       : {score.priority_level}")


def create_dqm_json_report(
    source_file: str | Path = SOURCE_FILE,
    *,
    max_visits_per_node: int = 3,
) -> None:
    """DQM sonuçlarını JSON dosyası olarak kaydeder."""
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    reporter = JSONReportWriter()

    analysis_result = analyzer.analyze_file(source_file)
    graphs = cfg_builder.build_from_file(source_file)

    if not analysis_result.functions or not graphs:
        print("JSON raporu için analiz edilebilir fonksiyon bulunamadı.")
        return

    for function, graph in zip(
        analysis_result.functions,
        graphs,
        strict=True,
    ):
        paths = path_analyzer.find_paths(
            graph,
            max_visits_per_node=max_visits_per_node,
        )

        scores = dqm.evaluate_paths(
            function=function,
            paths=paths,
        )

        output_path = (
            "output/dqm_reports/"
            f"{function.name}_dqm_report.json"
        )

        result_path = reporter.write_dqm_report(
            function=function,
            paths=paths,
            scores=scores,
            output_path=output_path,
            source_file=source_file,
        )

        print(
            "JSON raporu başarıyla oluşturuldu: "
            f"{result_path}"
        )


def run_automated_test_pipeline(
    source_file: str | Path = SOURCE_FILE,
    module_path: str = MODULE_PATH,
    output_directory: str | Path = GENERATED_TEST_DIRECTORY,
    *,
    overwrite: bool = True,
    timeout_seconds: float = 30.0,
) -> AutomationSummary:
    """
    Kaynak dosya için otomatik pytest dosyaları üretir ve çalıştırır.

    Args:
        source_file: Analiz edilecek Python kaynak dosyası.
        module_path: Kaynak dosyanın Python modül yolu.
        output_directory: Üretilen testlerin kaydedileceği klasör.
        overwrite: Mevcut test dosyalarının yenilenmesine izin verir.
        timeout_seconds: Her test dosyası için çalışma süresi sınırı.

    Returns:
        Test üretme ve çalıştırma sonuçlarını içeren özet.
    """
    service = AutomationService()

    summary = service.generate_and_execute(
        source_file=source_file,
        module_path=module_path,
        output_directory=output_directory,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

    print("=" * 65)
    print("OTOMATİK TEST ÜRETİM VE ÇALIŞTIRMA RAPORU")
    print("=" * 65)

    print(
        f"Üretilen Test Dosyası     : "
        f"{summary.generated_file_count}"
    )
    print(
        f"Başarılı Test Dosyası     : "
        f"{summary.successful_file_count}"
    )
    print(
        f"Başarısız Test Dosyası    : "
        f"{summary.failed_file_count}"
    )
    print(
        f"Üretilen Senaryo Sayısı   : "
        f"{summary.total_generated_scenario_count}"
    )
    print(
        f"Başarılı Pytest Sayısı    : "
        f"{summary.total_passed_test_count}"
    )
    print(
        f"Başarısız Pytest Sayısı   : "
        f"{summary.total_failed_test_count}"
    )
    print(
        f"Genel İşlem Başarılı mı?  : "
        f"{summary.success}"
    )

    for index, result in enumerate(
        summary.results,
        start=1,
    ):
        print("\n" + "-" * 65)
        print(f"TEST DOSYASI #{index}")
        print("-" * 65)

        print(
            f"Fonksiyon Adı             : "
            f"{result.artifact.function_name}"
        )
        print(
            f"Senaryo Sayısı            : "
            f"{result.artifact.scenario_count}"
        )
        print(
            f"Dosya Yolu                : "
            f"{result.artifact.output_path}"
        )
        print(
            f"Çıkış Kodu                : "
            f"{result.execution.exit_code}"
        )
        print(
            f"Çalışma Süresi            : "
            f"{result.execution.duration_seconds:.3f} saniye"
        )
        print(
            f"Başarılı Test             : "
            f"{result.execution.passed_count}"
        )
        print(
            f"Başarısız Test            : "
            f"{result.execution.failed_count}"
        )
        print(
            f"Hatalı Test               : "
            f"{result.execution.error_count}"
        )
        print(
            f"Atlanan Test              : "
            f"{result.execution.skipped_count}"
        )

    return summary


def run_coverage_pipeline(
    source_file: str | Path = SOURCE_FILE,
    module_path: str = MODULE_PATH,
    output_directory: str | Path = GENERATED_TEST_DIRECTORY,
    *,
    overwrite: bool = True,
    timeout_seconds: float = 30.0,
) -> CoverageWorkflowSummary:
    """
    Kaynak dosya için test üretir ve coverage ölçümü gerçekleştirir.

    Args:
        source_file: Coverage uygulanacak Python kaynak dosyası.
        module_path: Kaynak dosyanın Python import yolu.
        output_directory: Üretilen testlerin kaydedileceği klasör.
        overwrite: Mevcut test dosyalarının yenilenmesine izin verir.
        timeout_seconds: Coverage ölçümü için çalışma süresi sınırı.

    Returns:
        Test üretimi ve coverage sonuçlarını içeren özet.
    """
    service = CoverageWorkflowService()

    summary = service.generate_and_measure(
        source_file=source_file,
        module_path=module_path,
        output_directory=output_directory,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

    print("=" * 65)
    print("OTOMATİK TEST COVERAGE RAPORU")
    print("=" * 65)

    print(
        f"Ölçülen Test Dosyası      : "
        f"{summary.measured_file_count}"
    )
    print(
        f"Başarılı Dosya            : "
        f"{summary.successful_file_count}"
    )
    print(
        f"Başarısız Dosya           : "
        f"{summary.failed_file_count}"
    )
    print(
        f"Tam Coverage Dosyası      : "
        f"{summary.full_coverage_file_count}"
    )
    print(
        f"Ortalama Satır Coverage   : "
        f"%{summary.average_line_coverage_percent:.2f}"
    )
    print(
        f"Ortalama Branch Coverage  : "
        f"%{summary.average_branch_coverage_percent:.2f}"
    )
    print(
        f"Genel İşlem Başarılı mı?  : "
        f"{summary.success}"
    )

    for index, result in enumerate(
        summary.results,
        start=1,
    ):
        coverage = result.coverage

        print("\n" + "-" * 65)
        print(f"COVERAGE SONUCU #{index}")
        print("-" * 65)

        print(
            f"Fonksiyon Adı             : "
            f"{result.artifact.function_name}"
        )
        print(
            f"Test Dosyası              : "
            f"{result.artifact.output_path}"
        )
        print(
            f"Satır Coverage            : "
            f"%{coverage.line_coverage_percent:.2f}"
        )
        print(
            f"Branch Coverage           : "
            f"%{coverage.branch_coverage_percent:.2f}"
        )
        print(
            f"Çalıştırılan Satır        : "
            f"{coverage.covered_line_count}"
        )
        print(
            f"Eksik Satır               : "
            f"{coverage.missing_line_count}"
        )
        print(
            f"Toplam Satır              : "
            f"{coverage.total_line_count}"
        )
        print(
            f"Çalıştırılan Branch       : "
            f"{coverage.covered_branch_count}"
        )
        print(
            f"Eksik Branch              : "
            f"{coverage.missing_branch_count}"
        )
        print(
            f"Toplam Branch             : "
            f"{coverage.total_branch_count}"
        )
        print(
            f"Test Çıkış Kodu           : "
            f"{coverage.test_exit_code}"
        )
        print(
            f"Çalışma Süresi            : "
            f"{coverage.duration_seconds:.3f} saniye"
        )
        print(
            f"Tam Coverage mı?          : "
            f"{coverage.has_full_coverage}"
        )

    return summary


def run_real_rl_training(
    source_file: str | Path = SOURCE_FILE,
    module_path: str = MODULE_PATH,
    function_name: str = FUNCTION_NAME,
    output_directory: str | Path = GENERATED_TEST_DIRECTORY,
    *,
    max_visits_per_node: int = 3,
    episode_count: int = 3,
    epsilon: float = 0.0,
    epsilon_decay_rate: float | None = None,
    minimum_epsilon: float = 0.0,
    learning_rate: float = 0.5,
    discount_factor: float = 0.9,
    random_seed: int | None = 42,
    overwrite: bool = True,
    timeout_seconds: float = 30.0,
) -> RealRLTrainingResult:
    """
    Kaynak dosya üzerinde gerçek coverage tabanlı RL eğitimi çalıştırır.

    Kaynak kod analizinden başlayarak CFG, yürütme yolları,
    DQM, test senaryosu üretimi, gerçek pytest ve coverage
    ölçümü, reward hesaplama ve Q-Table güncellemesine kadar
    bütün akışı çalıştırır.
    """
    service = RealRLTrainingService()

    print("=" * 65)
    print("GERÇEK RL COVERAGE EĞİTİMİ")
    print("=" * 65)
    print(f"Kaynak dosya       : {source_file}")
    print(f"Modül yolu         : {module_path}")
    print(f"Fonksiyon          : {function_name}")
    print(f"Düğüm ziyaret sınırı: {max_visits_per_node}")
    print(f"Episode sayısı     : {episode_count}")
    print("\nEğitim başlatılıyor...\n")

    result = service.run(
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
    )

    print(result.report)

    print("\n" + "=" * 65)
    print("GERÇEK RL EĞİTİM SONUCU")
    print("=" * 65)
    print(
        f"Üretilen senaryo sayısı : "
        f"{result.scenario_count}"
    )
    print(
        f"Tamamlanan episode      : "
        f"{result.completed_episode_count}"
    )
    print(
        f"Q-Table state sayısı    : "
        f"{result.q_table_state_count}"
    )
    print(
        f"Eğitim başarılı mı?     : "
        f"{result.success}"
    )

    return result


def run_source_analysis(
    source_file: str | Path,
    module_path: str,
    function_name: str | None,
    all_functions: bool,
    output_directory: str | Path = GENERATED_TEST_DIRECTORY,
    *,
    max_visits_per_node: int = 3,
    episode_count: int = 3,
    epsilon: float = 0.0,
    learning_rate: float = 0.5,
    discount_factor: float = 0.9,
    random_seed: int | None = 42,
    overwrite: bool = True,
    timeout_seconds: float = 30.0,
    pipeline_timeout_seconds: float | None = None,
) -> ProjectAnalysisResult:
    """Tek veya bütün fonksiyonlar için ortak production akışını çalıştırır."""
    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path=module_path,
        function_name=function_name,
        all_functions=all_functions,
        output_root=output_directory,
        max_visits_per_node=max_visits_per_node,
        episode_count=episode_count,
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        random_seed=random_seed,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        per_function_timeout_seconds=pipeline_timeout_seconds,
    )
    print(ProjectAnalysisReportFormatter().format(result))
    return result


def project_exit_code(result: ProjectAnalysisResult) -> int:
    """Project sonucunu kararlı CLI exit code'una dönüştürür."""
    if result.status in {ProjectRunStatus.COMPLETED, ProjectRunStatus.PARTIAL}:
        return 0 if result.has_usable_result else 3
    if result.status is ProjectRunStatus.TIMED_OUT:
        return 4
    return 3


def print_runtime_configuration(
    configuration: ApplicationConfiguration,
) -> None:
    """Etkin hedef ve çalışma ayarlarını ekrana yazdırır."""
    print("\n" + "=" * 65)
    print("AKTİF ÇALIŞMA YAPILANDIRMASI")
    print("=" * 65)
    print(f"Kaynak dosya          : {configuration.source_file}")
    print(f"Modül yolu            : {configuration.module_path}")
    print(
        "Hedef fonksiyon       : "
        f"{'TÜM FONKSİYONLAR' if configuration.all_functions else configuration.function_name}"
    )
    print(f"Çıktı klasörü         : {configuration.output_directory}")
    print(
        "Düğüm ziyaret sınırı : "
        f"{configuration.max_visits_per_node}"
    )
    print(f"Episode sayısı        : {configuration.episode_count}")
    print(
        "Fonksiyon pipeline timeout: "
        f"{configuration.pipeline_timeout_seconds or 'Yok'}"
    )


def run_configured_operation(
    configuration: ApplicationConfiguration,
) -> object | None:
    """Yapılandırmada seçilen tek CLI işlemini çalıştırır."""
    operation = configuration.operation

    if operation == "analyze":
        print_analyzer_report(configuration.source_file)
        return None

    if operation == "cfg":
        print_cfg_report(configuration.source_file)
        return None

    if operation == "dqm":
        print_dqm_report(
            configuration.source_file,
            max_visits_per_node=(
                configuration.max_visits_per_node
            ),
        )
        return None

    if operation == "dqm-json":
        create_dqm_json_report(
            configuration.source_file,
            max_visits_per_node=(
                configuration.max_visits_per_node
            ),
        )
        return None

    if operation == "test":
        return run_automated_test_pipeline(
            source_file=configuration.source_file,
            module_path=configuration.module_path,
            output_directory=configuration.output_directory,
            overwrite=configuration.overwrite,
            timeout_seconds=configuration.timeout_seconds,
        )

    if operation == "coverage":
        return run_coverage_pipeline(
            source_file=configuration.source_file,
            module_path=configuration.module_path,
            output_directory=configuration.output_directory,
            overwrite=configuration.overwrite,
            timeout_seconds=configuration.timeout_seconds,
        )

    if operation == "demo":
        return RLDemoService().run()

    if operation == "rl":
        return run_source_analysis(
            source_file=configuration.source_file,
            module_path=configuration.module_path,
            function_name=configuration.function_name,
            all_functions=configuration.all_functions,
            output_directory=configuration.output_directory,
            max_visits_per_node=(
                configuration.max_visits_per_node
            ),
            episode_count=configuration.episode_count,
            epsilon=configuration.epsilon,
            learning_rate=configuration.learning_rate,
            discount_factor=configuration.discount_factor,
            random_seed=configuration.random_seed,
            overwrite=configuration.overwrite,
            timeout_seconds=configuration.timeout_seconds,
            pipeline_timeout_seconds=(
                configuration.pipeline_timeout_seconds
            ),
        )

    raise ValueError(
        f"Doğrudan çalıştırılamayan işlem: {operation!r}"
    )


def _interactive_source_file() -> tuple[Path, AnalysisResult]:
    raw_value = input("Python kaynak dosyası: ").strip()
    if not raw_value:
        raise SourceAnalysisValidationError(
            "Python kaynak dosyası boş bırakılamaz."
        )
    try:
        source_file = _python_source_file(raw_value).resolve()
    except argparse.ArgumentTypeError as error:
        raise SourceAnalysisValidationError(str(error)) from error

    analysis = PythonAnalyzer().analyze_file(source_file)
    print(f"Seçilen kaynak       : {source_file}")
    return source_file, analysis


def _suggest_module_path(source_file: Path) -> str | None:
    try:
        relative_path = source_file.relative_to(Path.cwd().resolve())
    except ValueError:
        return None
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _interactive_module_path(source_file: Path) -> str:
    suggestion = _suggest_module_path(source_file)
    prompt = "Module path"
    if suggestion is not None:
        prompt += f" [{suggestion}]"
    raw_value = input(f"{prompt}: ").strip()
    if not raw_value:
        if suggestion is None:
            raise SourceAnalysisValidationError("Module path boş bırakılamaz.")
        return suggestion
    try:
        return _python_module_path(raw_value)
    except argparse.ArgumentTypeError as error:
        raise SourceAnalysisValidationError(str(error)) from error


def _interactive_target_selection(
    analysis: AnalysisResult,
) -> tuple[str | None, bool]:
    mode = input("Analiz modu [1=tek fonksiyon, 2=tüm fonksiyonlar]: ").strip()
    if mode == "2":
        return None, True
    if mode != "1":
        raise SourceAnalysisValidationError("Analiz modu 1 veya 2 olmalıdır.")

    functions = tuple(
        function
        for function in analysis.functions
        if function.is_supported
    )
    if not functions:
        raise SourceAnalysisValidationError(
            "Kaynak dosyada desteklenen fonksiyon bulunamadı."
        )
    print("Desteklenen fonksiyonlar:")
    for function in functions:
        print(f"- {function.qualified_name}")
    raw_name = input("Qualified target: ")
    try:
        normalized_name = validate_qualified_target_name(raw_name)
    except (TypeError, ValueError) as error:
        raise SourceAnalysisValidationError(
            "Geçerli bir qualified target girilmelidir."
        ) from error
    if normalized_name not in {
        function.qualified_name for function in functions
    }:
        raise SourceAnalysisValidationError(
            f"Desteklenen fonksiyon bulunamadı: {normalized_name}"
        )
    return normalized_name, False


def _interactive_output_root(source_file: Path) -> Path:
    safe_stem = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in source_file.stem
    ).strip("_") or "source"
    suggestion = (Path("output") / f"{safe_stem}_analysis").resolve()
    raw_value = input(f"Output directory [{suggestion}]: ").strip()
    return Path(raw_value).resolve() if raw_value else suggestion


def _interactive_value(
    label: str,
    default: int | float,
    parser: Callable[[str], int | float],
) -> int | float:
    raw_value = input(f"{label} [{default}]: ").strip()
    if not raw_value:
        return default
    try:
        return parser(raw_value)
    except argparse.ArgumentTypeError as error:
        raise SourceAnalysisValidationError(str(error)) from error


def _interactive_analysis_configuration(
    defaults: ApplicationConfiguration,
) -> ApplicationConfiguration:
    source_file, analysis = _interactive_source_file()
    module_path = _interactive_module_path(source_file)
    function_name, all_functions = _interactive_target_selection(analysis)
    output_directory = _interactive_output_root(source_file)
    advanced = input("Gelişmiş ayarlar kullanılsın mı? [e/H]: ").strip().lower()
    if advanced not in {"", "e", "evet", "h", "hayır", "hayir"}:
        raise SourceAnalysisValidationError("Gelişmiş ayar seçimi e veya h olmalıdır.")
    if advanced in {"", "h", "hayır", "hayir"}:
        return ApplicationConfiguration(
            source_file=source_file,
            module_path=module_path,
            function_name=function_name,
            all_functions=all_functions,
            output_directory=output_directory,
            operation="rl",
            max_visits_per_node=defaults.max_visits_per_node,
            episode_count=defaults.episode_count,
            epsilon=defaults.epsilon,
            learning_rate=defaults.learning_rate,
            discount_factor=defaults.discount_factor,
            random_seed=defaults.random_seed,
            overwrite=defaults.overwrite,
            timeout_seconds=defaults.timeout_seconds,
            pipeline_timeout_seconds=defaults.pipeline_timeout_seconds,
        )

    max_visits = _interactive_value(
        "Max visits per node", defaults.max_visits_per_node, _positive_int
    )
    episode_count = _interactive_value(
        "Episode count", defaults.episode_count, _positive_int
    )
    epsilon = _interactive_value("Epsilon", defaults.epsilon, _probability)
    learning_rate = _interactive_value(
        "Learning rate", defaults.learning_rate, _probability
    )
    discount_factor = _interactive_value(
        "Discount factor", defaults.discount_factor, _probability
    )
    seed_raw = input(f"Random seed [{defaults.random_seed}]: ").strip()
    try:
        random_seed = defaults.random_seed if not seed_raw else int(seed_raw)
    except ValueError as error:
        raise SourceAnalysisValidationError("Random seed tam sayı olmalıdır.") from error
    timeout = _interactive_value(
        "Pytest/coverage timeout", defaults.timeout_seconds, _positive_float
    )
    pipeline_default = defaults.pipeline_timeout_seconds
    pipeline_label = "Yok" if pipeline_default is None else str(pipeline_default)
    pipeline_raw = input(f"Pipeline timeout [{pipeline_label}]: ").strip()
    if not pipeline_raw:
        pipeline_timeout = pipeline_default
    else:
        try:
            pipeline_timeout = _positive_float(pipeline_raw)
        except argparse.ArgumentTypeError as error:
            raise SourceAnalysisValidationError(str(error)) from error

    return ApplicationConfiguration(
        source_file=source_file,
        module_path=module_path,
        function_name=function_name,
        all_functions=all_functions,
        output_directory=output_directory,
        operation="rl",
        max_visits_per_node=int(max_visits),
        episode_count=int(episode_count),
        epsilon=float(epsilon),
        learning_rate=float(learning_rate),
        discount_factor=float(discount_factor),
        random_seed=random_seed,
        overwrite=defaults.overwrite,
        timeout_seconds=float(timeout),
        pipeline_timeout_seconds=pipeline_timeout,
    )


def _run_interactive_project_analysis(
    defaults: ApplicationConfiguration,
) -> ProjectAnalysisResult:
    configuration = _interactive_analysis_configuration(defaults)
    return run_source_analysis(
        source_file=configuration.source_file,
        module_path=configuration.module_path,
        function_name=configuration.function_name,
        all_functions=configuration.all_functions,
        output_directory=configuration.output_directory,
        max_visits_per_node=configuration.max_visits_per_node,
        episode_count=configuration.episode_count,
        epsilon=configuration.epsilon,
        learning_rate=configuration.learning_rate,
        discount_factor=configuration.discount_factor,
        random_seed=configuration.random_seed,
        overwrite=configuration.overwrite,
        timeout_seconds=configuration.timeout_seconds,
        pipeline_timeout_seconds=configuration.pipeline_timeout_seconds,
    )


def _run_interactive_static_preview() -> None:
    source_file, _ = _interactive_source_file()
    print_analyzer_report(source_file)


def print_menu() -> None:
    """Uygulama ana menüsünü ekrana yazdırır."""
    print("\n" + "=" * 55)
    print("RL-UNIT-TEST DEVELOPER TOOL")
    print("=" * 55)
    print("1. Kaynak Kod / Proje Analizi")
    print("2. Hızlı Statik Ön İnceleme")
    print("3. Dış Kaynak Analizi")
    print("0. Çıkış")


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Komut satırı uygulamasını çalıştırır."""
    configuration = parse_cli_arguments(argv)

    if configuration.operation != "menu":
        print_runtime_configuration(configuration)
        try:
            result = run_configured_operation(configuration)
        except (
            FileNotFoundError,
            SyntaxError,
            SourceAnalysisValidationError,
        ) as error:
            print(f"CLI doğrulama hatası: {error}")
            return 2
        if configuration.operation == "rl":
            return project_exit_code(result)
        return 0

    while True:
        print_menu()

        choice = input("\nSeçiminiz: ").strip()

        if choice == "1":
            print()
            try:
                _run_interactive_project_analysis(configuration)
            except (
                FileNotFoundError,
                SyntaxError,
                SourceAnalysisValidationError,
            ) as error:
                print(f"Proje analizi başlatılamadı: {error}")
            continue

        if choice == "2":
            print()
            try:
                _run_interactive_static_preview()
            except (
                FileNotFoundError,
                SyntaxError,
                SourceAnalysisValidationError,
            ) as error:
                print(f"Statik ön inceleme başlatılamadı: {error}")

            continue

        if choice == "3":
            print()
            ExternalSourceTerminalAdapter(input_fn=input, output_fn=print).run_menu()
            continue

        if choice == "0":
            print("\nProgram sonlandırıldı.")
            return 0

        print(
            "\nGeçersiz seçim. "
            "Lütfen 0, 1, 2 veya 3 girin."
        )


if __name__ == "__main__":
    raise SystemExit(main())
