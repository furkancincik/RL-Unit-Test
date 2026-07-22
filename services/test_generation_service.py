from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.file_writer import GeneratedTestFileWriter
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import TestScenarioGenerator


@dataclass(frozen=True, slots=True)
class GeneratedTestArtifact:
    """
    Otomatik test üretim işleminin sonucunu temsil eder.

    Attributes:
        function_name: Test üretilen fonksiyonun adı.
        scenario_count: Fonksiyon için oluşturulan senaryo sayısı.
        output_path: Oluşturulan pytest dosyasının yolu.
    """

    function_name: str
    scenario_count: int
    output_path: Path


class TestGenerationService:
    """Kaynak kod analizinden pytest dosyasına kadar süreci yönetir."""

    def __init__(
        self,
        analyzer: PythonAnalyzer | None = None,
        cfg_builder: ControlFlowGraphBuilder | None = None,
        path_analyzer: CFGPathAnalyzer | None = None,
        dqm: DecisionQualityMatrix | None = None,
        scenario_generator: TestScenarioGenerator | None = None,
        pytest_generator: PytestGenerator | None = None,
        file_writer: GeneratedTestFileWriter | None = None,
    ) -> None:
        """
        Test üretim servisinin bağımlılıklarını hazırlar.

        Bağımlılıklar dışarıdan verilmezse varsayılan uygulamalar
        otomatik olarak oluşturulur.

        Args:
            analyzer: Python statik analiz bileşeni.
            cfg_builder: Control Flow Graph üreticisi.
            path_analyzer: Yürütme yolu analiz bileşeni.
            dqm: DQM değerlendirme bileşeni.
            scenario_generator: Test senaryosu üreticisi.
            pytest_generator: Pytest kaynak kodu üreticisi.
            file_writer: Üretilen kodu dosyaya yazan bileşen.
        """
        self._analyzer = analyzer or PythonAnalyzer()
        self._cfg_builder = cfg_builder or ControlFlowGraphBuilder()
        self._path_analyzer = path_analyzer or CFGPathAnalyzer()
        self._dqm = dqm or DecisionQualityMatrix()
        self._scenario_generator = (
            scenario_generator or TestScenarioGenerator()
        )
        self._pytest_generator = (
            pytest_generator or PytestGenerator()
        )
        self._file_writer = (
            file_writer or GeneratedTestFileWriter()
        )

    def generate_for_file(
        self,
        source_file: str | Path,
        module_path: str,
        output_directory: str | Path,
        *,
        overwrite: bool = False,
    ) -> list[GeneratedTestArtifact]:
        """
        Kaynak dosyadaki fonksiyonlar için pytest dosyaları üretir.

        Her fonksiyon için analiz, CFG, yürütme yolu, DQM ve test
        senaryosu işlemleri uygulanır. Üretilen pytest kodu belirtilen
        çıktı klasörüne kaydedilir.

        Args:
            source_file: Analiz edilecek Python kaynak dosyası.
            module_path: Kaynak dosyanın Python import yolu.
                Örnek: ``datasets.sample_code``.
            output_directory: Üretilen testlerin kaydedileceği klasör.
            overwrite: Mevcut test dosyalarının üzerine yazılmasına
                izin verilip verilmediği.

        Returns:
            Oluşturulan test dosyalarına ait sonuç listesi.

        Raises:
            ValueError: Kaynak dosyada analiz edilebilir fonksiyon veya
                eşleşen CFG bulunmadığında.
        """
        normalized_source_file = self._normalize_source_file(
            source_file
        )
        normalized_output_directory = (
            self._normalize_output_directory(output_directory)
        )

        analysis_result = self._analyzer.analyze_file(
            normalized_source_file
        )
        graphs = self._cfg_builder.build_from_file(
            normalized_source_file
        )

        if not analysis_result.functions:
            raise ValueError(
                "Kaynak dosyada analiz edilebilir fonksiyon bulunamadı."
            )

        if not graphs:
            raise ValueError(
                "Kaynak dosyada test üretimine uygun CFG bulunamadı."
            )

        if len(analysis_result.functions) != len(graphs):
            raise ValueError(
                "Fonksiyon analizi ile CFG sonuçlarının sayısı eşleşmiyor."
            )

        artifacts: list[GeneratedTestArtifact] = []

        for function, graph in zip(
            analysis_result.functions,
            graphs,
            strict=True,
        ):
            paths = self._path_analyzer.find_paths(graph)

            scores = self._dqm.evaluate_paths(
                function=function,
                paths=paths,
            )

            scenarios = self._scenario_generator.generate(
                function_name=function.name,
                paths=paths,
                scores=scores,
            )

            generated_code = self._pytest_generator.generate(
                module_path=module_path,
                function_name=function.name,
                scenarios=scenarios,
            )

            output_path = (
                normalized_output_directory
                / f"test_{function.name}.py"
            )

            written_path = self._file_writer.write(
                source_code=generated_code,
                output_path=output_path,
                overwrite=overwrite,
            )

            artifacts.append(
                GeneratedTestArtifact(
                    function_name=function.name,
                    scenario_count=len(scenarios),
                    output_path=written_path,
                )
            )

        return artifacts

    @staticmethod
    def _normalize_source_file(
        source_file: str | Path,
    ) -> Path:
        """
        Kaynak dosya yolunu doğrular.

        Args:
            source_file: Analiz edilecek dosyanın yolu.

        Returns:
            Doğrulanmış Path nesnesi.

        Raises:
            TypeError: Yol string veya Path değilse.
            ValueError: Yol boşsa veya uzantısı ``.py`` değilse.
            FileNotFoundError: Kaynak dosya bulunamazsa.
        """
        if not isinstance(source_file, (str, Path)):
            raise TypeError(
                "Kaynak dosya yolu string veya Path olmalıdır."
            )

        if isinstance(source_file, str) and not source_file.strip():
            raise ValueError("Kaynak dosya yolu boş olamaz.")

        path = Path(source_file)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Kaynak dosyanın uzantısı .py olmalıdır."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Kaynak dosya bulunamadı: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Kaynak dosya yolu bir dosya olmalıdır: {path}"
            )

        return path

    @staticmethod
    def _normalize_output_directory(
        output_directory: str | Path,
    ) -> Path:
        """
        Çıktı klasörü yolunu doğrular.

        Args:
            output_directory: Test dosyalarının kaydedileceği klasör.

        Returns:
            Normalize edilmiş Path nesnesi.

        Raises:
            TypeError: Yol string veya Path değilse.
            ValueError: Yol boşsa.
        """
        if not isinstance(output_directory, (str, Path)):
            raise TypeError(
                "Çıktı klasörü string veya Path olmalıdır."
            )

        if (
            isinstance(output_directory, str)
            and not output_directory.strip()
        ):
            raise ValueError("Çıktı klasörü boş olamaz.")

        return Path(output_directory)