from pathlib import Path

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from reports.json_reporter import JSONReportWriter
from services.automation_service import (
    AutomationService,
    AutomationSummary,
)


SOURCE_FILE = "datasets/sample_code.py"
MODULE_PATH = "datasets.sample_code"
GENERATED_TEST_DIRECTORY = Path("output/generated_tests")


def print_analyzer_report() -> None:
    """Örnek Python dosyası için statik analiz raporu üretir."""
    analyzer = PythonAnalyzer()
    result = analyzer.analyze_file(SOURCE_FILE)

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


def print_cfg_report() -> None:
    """Örnek Python dosyasındaki fonksiyonlar için CFG raporu üretir."""
    builder = ControlFlowGraphBuilder()
    graphs = builder.build_from_file(SOURCE_FILE)

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


def print_dqm_report() -> None:
    """Yürütme yolları için DQM önceliklendirme raporu üretir."""
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()

    analysis_result = analyzer.analyze_file(SOURCE_FILE)
    graphs = cfg_builder.build_from_file(SOURCE_FILE)

    if not analysis_result.functions or not graphs:
        print("DQM değerlendirmesi için fonksiyon bulunamadı.")
        return

    for function, graph in zip(
        analysis_result.functions,
        graphs,
        strict=True,
    ):
        paths = path_analyzer.find_paths(graph)

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


def create_dqm_json_report() -> None:
    """DQM sonuçlarını JSON dosyası olarak kaydeder."""
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    reporter = JSONReportWriter()

    analysis_result = analyzer.analyze_file(SOURCE_FILE)
    graphs = cfg_builder.build_from_file(SOURCE_FILE)

    if not analysis_result.functions or not graphs:
        print("JSON raporu için analiz edilebilir fonksiyon bulunamadı.")
        return

    for function, graph in zip(
        analysis_result.functions,
        graphs,
        strict=True,
    ):
        paths = path_analyzer.find_paths(graph)

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
            source_file=SOURCE_FILE,
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


def print_menu() -> None:
    """Uygulama ana menüsünü ekrana yazdırır."""
    print("\n" + "=" * 55)
    print("RL UNIT TEST")
    print("=" * 55)
    print("1 - Python kod analizi")
    print("2 - Control Flow Graph")
    print("3 - DQM yol önceliklendirme")
    print("4 - DQM JSON raporu oluştur")
    print("5 - Otomatik test üret ve çalıştır")
    print("0 - Çıkış")


def main() -> None:
    """Komut satırı uygulamasını çalıştırır."""
    while True:
        print_menu()

        choice = input("\nSeçiminiz: ").strip()

        if choice == "1":
            print()
            print_analyzer_report()
            continue

        if choice == "2":
            print()
            print_cfg_report()
            continue

        if choice == "3":
            print()
            print_dqm_report()
            continue

        if choice == "4":
            print()
            create_dqm_json_report()
            continue

        if choice == "5":
            print()

            try:
                run_automated_test_pipeline()
            except (
                FileNotFoundError,
                TypeError,
                ValueError,
                TimeoutError,
                OSError,
            ) as error:
                print(
                    "Otomatik test işlemi tamamlanamadı: "
                    f"{error}"
                )

            continue

        if choice == "0":
            print("\nProgram sonlandırıldı.")
            break

        print(
            "\nGeçersiz seçim. "
            "Lütfen 0, 1, 2, 3, 4 veya 5 girin."
        )


if __name__ == "__main__":
    main()