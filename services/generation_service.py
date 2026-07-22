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
    Otomatik test Ã¼retim iÅŸleminin sonucunu temsil eder.

    Attributes:
        function_name: Test Ã¼retilen fonksiyonun adÄ±.
        scenario_count: Fonksiyon iÃ§in oluÅŸturulan senaryo sayÄ±sÄ±.
        output_path: OluÅŸturulan pytest dosyasÄ±nÄ±n yolu.
    """

    function_name: str
    scenario_count: int
    output_path: Path


class GenerationService:
    """Kaynak kod analizinden pytest dosyasÄ±na kadar sÃ¼reci yÃ¶netir."""

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
        Test Ã¼retim servisinin baÄŸÄ±mlÄ±lÄ±klarÄ±nÄ± hazÄ±rlar.

        BaÄŸÄ±mlÄ±lÄ±klar dÄ±ÅŸarÄ±dan verilmezse varsayÄ±lan uygulamalar
        otomatik olarak oluÅŸturulur.

        Args:
            analyzer: Python statik analiz bileÅŸeni.
            cfg_builder: Control Flow Graph Ã¼reticisi.
            path_analyzer: YÃ¼rÃ¼tme yolu analiz bileÅŸeni.
            dqm: DQM deÄŸerlendirme bileÅŸeni.
            scenario_generator: Test senaryosu Ã¼reticisi.
            pytest_generator: Pytest kaynak kodu Ã¼reticisi.
            file_writer: Ãœretilen kodu dosyaya yazan bileÅŸen.
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
        Kaynak dosyadaki fonksiyonlar iÃ§in pytest dosyalarÄ± Ã¼retir.

        Her fonksiyon iÃ§in analiz, CFG, yÃ¼rÃ¼tme yolu, DQM ve test
        senaryosu iÅŸlemleri uygulanÄ±r. Ãœretilen pytest kodu belirtilen
        Ã§Ä±ktÄ± klasÃ¶rÃ¼ne kaydedilir.

        Args:
            source_file: Analiz edilecek Python kaynak dosyasÄ±.
            module_path: Kaynak dosyanÄ±n Python import yolu.
                Ã–rnek: ``datasets.sample_code``.
            output_directory: Ãœretilen testlerin kaydedileceÄŸi klasÃ¶r.
            overwrite: Mevcut test dosyalarÄ±nÄ±n Ã¼zerine yazÄ±lmasÄ±na
                izin verilip verilmediÄŸi.

        Returns:
            OluÅŸturulan test dosyalarÄ±na ait sonuÃ§ listesi.

        Raises:
            ValueError: Kaynak dosyada analiz edilebilir fonksiyon veya
                eÅŸleÅŸen CFG bulunmadÄ±ÄŸÄ±nda.
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
                "Kaynak dosyada analiz edilebilir fonksiyon bulunamadÄ±."
            )

        if not graphs:
            raise ValueError(
                "Kaynak dosyada test Ã¼retimine uygun CFG bulunamadÄ±."
            )

        if len(analysis_result.functions) != len(graphs):
            raise ValueError(
                "Fonksiyon analizi ile CFG sonuÃ§larÄ±nÄ±n sayÄ±sÄ± eÅŸleÅŸmiyor."
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
        Kaynak dosya yolunu doÄŸrular.

        Args:
            source_file: Analiz edilecek dosyanÄ±n yolu.

        Returns:
            DoÄŸrulanmÄ±ÅŸ Path nesnesi.

        Raises:
            TypeError: Yol string veya Path deÄŸilse.
            ValueError: Yol boÅŸsa veya uzantÄ±sÄ± ``.py`` deÄŸilse.
            FileNotFoundError: Kaynak dosya bulunamazsa.
        """
        if not isinstance(source_file, (str, Path)):
            raise TypeError(
                "Kaynak dosya yolu string veya Path olmalÄ±dÄ±r."
            )

        if isinstance(source_file, str) and not source_file.strip():
            raise ValueError("Kaynak dosya yolu boÅŸ olamaz.")

        path = Path(source_file)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Kaynak dosyanÄ±n uzantÄ±sÄ± .py olmalÄ±dÄ±r."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Kaynak dosya bulunamadÄ±: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Kaynak dosya yolu bir dosya olmalÄ±dÄ±r: {path}"
            )

        return path

    @staticmethod
    def _normalize_output_directory(
        output_directory: str | Path,
    ) -> Path:
        """
        Ã‡Ä±ktÄ± klasÃ¶rÃ¼ yolunu doÄŸrular.

        Args:
            output_directory: Test dosyalarÄ±nÄ±n kaydedileceÄŸi klasÃ¶r.

        Returns:
            Normalize edilmiÅŸ Path nesnesi.

        Raises:
            TypeError: Yol string veya Path deÄŸilse.
            ValueError: Yol boÅŸsa.
        """
        if not isinstance(output_directory, (str, Path)):
            raise TypeError(
                "Ã‡Ä±ktÄ± klasÃ¶rÃ¼ string veya Path olmalÄ±dÄ±r."
            )

        if (
            isinstance(output_directory, str)
            and not output_directory.strip()
        ):
            raise ValueError("Ã‡Ä±ktÄ± klasÃ¶rÃ¼ boÅŸ olamaz.")

        return Path(output_directory)

