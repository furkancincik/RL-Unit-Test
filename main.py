from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder


def print_analyzer_report() -> None:
    """Örnek Python dosyası için statik analiz raporu üretir."""
    analyzer = PythonAnalyzer()
    result = analyzer.analyze_file("datasets/sample_code.py")

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

    graphs = builder.build_from_file(
        "datasets/sample_code.py"
    )

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


def print_menu() -> None:
    """Uygulama ana menüsünü ekrana yazdırır."""
    print("\n" + "=" * 55)
    print("RL UNIT TEST")
    print("=" * 55)
    print("1 - Python kod analizi")
    print("2 - Control Flow Graph")
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

        if choice == "0":
            print("\nProgram sonlandırıldı.")
            break

        print("\nGeçersiz seçim. Lütfen 0, 1 veya 2 girin.")


if __name__ == "__main__":
    main()