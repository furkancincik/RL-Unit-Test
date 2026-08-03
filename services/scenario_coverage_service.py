from __future__ import annotations

import keyword
from dataclasses import dataclass
from pathlib import Path

from generator.file_writer import GeneratedTestFileWriter
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult
from services.coverage_service import CoverageService


@dataclass(frozen=True, slots=True)
class ScenarioCoverageResult:
    """
    Tek bir test senaryosu için gerçekleştirilen coverage
    işleminin sonucunu temsil eder.

    Attributes:
        scenario:
            Coverage ölçümü yapılan test senaryosu.

        test_file:
            Senaryo için oluşturulan pytest dosyasının yolu.

        coverage:
            CoverageService tarafından üretilen coverage sonucu.
    """

    scenario: Scenario
    test_file: Path
    coverage: CoverageResult

    @property
    def success(self) -> bool:
        """
        Test çalıştırma ve coverage işleminin başarılı olup
        olmadığını belirtir.
        """
        return self.coverage.success

    @property
    def has_full_coverage(self) -> bool:
        """
        Senaryonun tek başına tam satır ve branch coverage
        sağlayıp sağlamadığını belirtir.
        """
        return self.coverage.has_full_coverage


class ScenarioCoverageService:
    """
    Tek bir Scenario nesnesini pytest dosyasına dönüştürür
    ve hedef kaynak dosya üzerindeki coverage sonucunu ölçer.

    Bu servis RL algoritmasını veya Action nesnesini bilmez.
    Görevi yalnızca seçilmiş bir test senaryosunu çalıştırılabilir
    pytest koduna dönüştürmek ve coverage sonucunu üretmektir.
    """

    def __init__(
        self,
        pytest_generator: PytestGenerator | None = None,
        file_writer: GeneratedTestFileWriter | None = None,
        coverage_service: CoverageService | None = None,
    ) -> None:
        """
        Servisin bağımlılıklarını hazırlar.

        Bağımlılıklar dışarıdan verilmezse varsayılan uygulamalar
        otomatik olarak oluşturulur.
        """
        self._pytest_generator = (
            pytest_generator or PytestGenerator()
        )
        self._file_writer = (
            file_writer or GeneratedTestFileWriter()
        )
        self._coverage_service = (
            coverage_service or CoverageService()
        )

    def measure_scenario(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        scenario: Scenario,
        output_directory: str | Path,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
    ) -> ScenarioCoverageResult:
        """
        Tek bir senaryo için pytest oluşturur ve coverage ölçer.

        Args:
            source_file:
                Test edilen Python kaynak dosyası.

            module_path:
                Kaynak dosyanın Python import yolu.
                Örnek: ``datasets.sample_code``.

            function_name:
                Test edilen fonksiyonun adı.

            scenario:
                RL veya başka bir karar mekanizması tarafından
                seçilmiş test senaryosu.

            output_directory:
                Üretilen geçici/yönetilen test dosyasının
                kaydedileceği klasör.

            overwrite:
                Aynı isimli test dosyasının üzerine yazılmasına
                izin verilip verilmediği.

            timeout_seconds:
                Coverage işlemi için azami süre.

        Returns:
            Senaryo, test dosyası ve coverage sonucunu içeren
            ScenarioCoverageResult.

        Raises:
            TypeError:
                Girdilerden biri geçersiz türdeyse.

            ValueError:
                Modül yolu, fonksiyon adı veya çıktı klasörü
                geçersizse.

            FileNotFoundError:
                Kaynak dosya bulunamazsa.
        """
        normalized_source_file = self._normalize_source_file(
            source_file
        )
        normalized_module_path = self._normalize_module_path(
            module_path
        )
        normalized_function_name = (
            self._normalize_function_name(
                function_name
            )
        )
        normalized_scenario = self._validate_scenario(
            scenario
        )
        normalized_output_directory = (
            self._normalize_output_directory(
                output_directory
            )
        )

        generated_code = self._pytest_generator.generate(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=(normalized_scenario,),
        )

        output_path = (
            normalized_output_directory
            / self._create_test_file_name(
                function_name=normalized_function_name,
                scenario=normalized_scenario,
            )
        )

        written_path = self._file_writer.write(
            source_code=generated_code,
            output_path=output_path,
            overwrite=overwrite,
        )

        coverage_result = self._coverage_service.measure(
            source_file=normalized_source_file,
            test_file=written_path,
            timeout_seconds=timeout_seconds,
        )

        return ScenarioCoverageResult(
            scenario=normalized_scenario,
            test_file=written_path,
            coverage=coverage_result,
        )

    @staticmethod
    def _create_test_file_name(
        function_name: str,
        scenario: Scenario,
    ) -> str:
        """
        Senaryoya özel kararlı test dosyası adı oluşturur.
        """
        return (
            f"test_{function_name}_"
            f"scenario_{scenario.priority_rank:03d}.py"
        )

    @staticmethod
    def _normalize_source_file(
        source_file: str | Path,
    ) -> Path:
        """
        Kaynak dosya yolunu temel düzeyde doğrular.
        """
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
                "source_file bir dosya olmalıdır."
            )

        return path.resolve()

    @classmethod
    def _normalize_module_path(
        cls,
        module_path: str,
    ) -> str:
        """
        Python modül yolunu doğrular.
        """
        if not isinstance(module_path, str):
            raise TypeError(
                "module_path string olmalıdır."
            )

        normalized_module_path = module_path.strip()

        if not normalized_module_path:
            raise ValueError(
                "module_path boş olamaz."
            )

        parts = normalized_module_path.split(".")

        if any(
            not cls._is_valid_identifier(part)
            for part in parts
        ):
            raise ValueError(
                f"Geçersiz Python modül yolu: "
                f"{normalized_module_path}"
            )

        return normalized_module_path

    @classmethod
    def _normalize_function_name(
        cls,
        function_name: str,
    ) -> str:
        """
        Fonksiyon adını doğrular.
        """
        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        normalized_function_name = (
            function_name.strip()
        )

        if not normalized_function_name:
            raise ValueError(
                "function_name boş olamaz."
            )

        if not cls._is_valid_identifier(
            normalized_function_name
        ):
            raise ValueError(
                "Geçersiz Python fonksiyon adı: "
                f"{normalized_function_name}"
            )

        return normalized_function_name

    @staticmethod
    def _validate_scenario(
        scenario: Scenario,
    ) -> Scenario:
        """
        Seçilen test senaryosunu doğrular.
        """
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "scenario bir Scenario örneği olmalıdır."
            )

        return scenario

    @staticmethod
    def _normalize_output_directory(
        output_directory: str | Path,
    ) -> Path:
        """
        Test dosyasının yazılacağı klasörü doğrular.
        """
        if not isinstance(
            output_directory,
            (str, Path),
        ):
            raise TypeError(
                "output_directory string veya Path "
                "olmalıdır."
            )

        if (
            isinstance(output_directory, str)
            and not output_directory.strip()
        ):
            raise ValueError(
                "output_directory boş olamaz."
            )

        return Path(output_directory)

    @staticmethod
    def _is_valid_identifier(
        value: str,
    ) -> bool:
        """
        Değerin geçerli Python tanımlayıcısı olup
        olmadığını belirtir.
        """
        return (
            value.isidentifier()
            and not keyword.iskeyword(value)
        )