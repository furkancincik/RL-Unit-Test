from __future__ import annotations

import keyword
from dataclasses import dataclass
from pathlib import Path

from generator.file_writer import GeneratedTestFileWriter
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from services.coverage_service import CoverageService


CoverageOutput = CoverageResult | FunctionCoverageResult


@dataclass(frozen=True, slots=True)
class ScenarioSuiteCoverageResult:
    """
    Birden fazla test senaryosunun birlikte çalıştırılmasıyla
    elde edilen coverage sonucunu temsil eder.

    Attributes:
        scenarios:
            Aynı pytest dosyasında çalıştırılan test senaryoları.

        test_file:
            Senaryo grubundan oluşturulan pytest dosyasının yolu.

        coverage:
            Senaryoların birlikte çalıştırılmasıyla elde edilen
            dosya veya fonksiyon bazlı coverage sonucu.
    """

    scenarios: tuple[Scenario, ...]
    test_file: Path
    coverage: CoverageOutput

    def __post_init__(self) -> None:
        """Sonuç modelindeki alanları doğrular."""
        if not isinstance(self.scenarios, tuple):
            raise TypeError(
                "scenarios bir tuple olmalıdır."
            )

        if any(
            not isinstance(scenario, Scenario)
            for scenario in self.scenarios
        ):
            raise TypeError(
                "scenarios yalnızca Scenario "
                "nesneleri içermelidir."
            )

        if not isinstance(self.test_file, Path):
            raise TypeError(
                "test_file bir Path örneği olmalıdır."
            )

        if not isinstance(
            self.coverage,
            (
                CoverageResult,
                FunctionCoverageResult,
            ),
        ):
            raise TypeError(
                "coverage bir CoverageResult veya "
                "FunctionCoverageResult örneği olmalıdır."
            )

    @property
    def scenario_count(self) -> int:
        """Test paketindeki senaryo sayısını döndürür."""
        return len(self.scenarios)

    @property
    def success(self) -> bool:
        """
        Pytest ve coverage işleminin başarılı olup
        olmadığını belirtir.
        """
        return self.coverage.success

    @property
    def has_full_coverage(self) -> bool:
        """
        Test paketinin hedef kapsam için tam coverage sağlayıp
        sağlamadığını belirtir.

        Dosya bazlı sonuçta dosyanın, fonksiyon bazlı sonuçta
        hedef fonksiyonun tam coverage durumu kullanılır.
        """
        return self.coverage.has_full_coverage

    @property
    def is_function_coverage(self) -> bool:
        """
        Coverage sonucunun fonksiyon bazlı olup olmadığını belirtir.
        """
        return isinstance(
            self.coverage,
            FunctionCoverageResult,
        )


class ScenarioSuiteCoverageService:
    """
    Birden fazla Scenario nesnesini tek bir pytest dosyasına
    dönüştürür ve kümülatif coverage sonucunu ölçer.

    Satır aralığı verilmediğinde kaynak dosyanın tamamı ölçülür.
    Fonksiyon başlangıç ve bitiş satırları birlikte verildiğinde
    yalnızca hedef fonksiyonun coverage sonucu üretilir.

    Bu servis RL algoritmasını, Action nesnesini veya QTable
    yapısını bilmez. Yalnızca kendisine verilen senaryo grubunu
    çalıştırılabilir test paketine dönüştürür.
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
            pytest_generator
            if pytest_generator is not None
            else PytestGenerator()
        )

        self._file_writer = (
            file_writer
            if file_writer is not None
            else GeneratedTestFileWriter()
        )

        self._coverage_service = (
            coverage_service
            if coverage_service is not None
            else CoverageService()
        )

    def measure_scenarios(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        scenarios: tuple[Scenario, ...],
        output_directory: str | Path,
        function_start_line: int | None = None,
        function_end_line: int | None = None,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
    ) -> ScenarioSuiteCoverageResult:
        """
        Seçilmiş senaryoları aynı pytest dosyasında çalıştırır ve
        kümülatif coverage sonucunu ölçer.

        Args:
            source_file:
                Test edilen Python kaynak dosyası.

            module_path:
                Kaynak dosyanın Python import yolu.

            function_name:
                Test edilen fonksiyonun adı.

            scenarios:
                Birlikte çalıştırılacak test senaryoları.

            output_directory:
                Üretilen pytest dosyasının yazılacağı klasör.

            function_start_line:
                Hedef fonksiyonun başlangıç satırı.

                None olduğunda dosya bazlı coverage kullanılır.
                Fonksiyon bazlı ölçüm için function_end_line ile
                birlikte verilmelidir.

            function_end_line:
                Hedef fonksiyonun bitiş satırı.

                None olduğunda dosya bazlı coverage kullanılır.
                Fonksiyon bazlı ölçüm için function_start_line ile
                birlikte verilmelidir.

            overwrite:
                Aynı isimli dosyanın üzerine yazılmasına izin verir.

            timeout_seconds:
                Coverage işlemi için azami süre.

        Returns:
            Senaryoları, pytest dosyasını ve coverage sonucunu içeren
            ScenarioSuiteCoverageResult.

        Raises:
            TypeError:
                Parametrelerden biri geçersiz türdeyse.

            ValueError:
                Parametrelerden biri geçersiz değerdeyse veya
                fonksiyon satır aralığının yalnızca bir bölümü
                verilmişse.
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

        normalized_scenarios = self._normalize_scenarios(
            scenarios
        )

        normalized_output_directory = (
            self._normalize_output_directory(
                output_directory
            )
        )

        normalized_function_range = (
            self._normalize_function_range(
                function_start_line=function_start_line,
                function_end_line=function_end_line,
            )
        )

        normalized_overwrite = self._validate_overwrite(
            overwrite
        )

        normalized_timeout = self._validate_timeout(
            timeout_seconds
        )

        generated_code = self._pytest_generator.generate(
            module_path=normalized_module_path,
            function_name=normalized_function_name,
            scenarios=normalized_scenarios,
        )

        output_path = (
            normalized_output_directory
            / self._create_test_file_name(
                function_name=normalized_function_name,
            )
        )

        written_path = self._file_writer.write(
            source_code=generated_code,
            output_path=output_path,
            overwrite=normalized_overwrite,
        )

        coverage_result = self._measure_coverage(
            source_file=normalized_source_file,
            test_file=written_path,
            function_name=normalized_function_name,
            function_range=normalized_function_range,
            timeout_seconds=normalized_timeout,
        )

        return ScenarioSuiteCoverageResult(
            scenarios=normalized_scenarios,
            test_file=written_path,
            coverage=coverage_result,
        )

    def _measure_coverage(
        self,
        *,
        source_file: Path,
        test_file: Path,
        function_name: str,
        function_range: tuple[int, int] | None,
        timeout_seconds: float,
    ) -> CoverageOutput:
        """
        Satır aralığına göre dosya veya fonksiyon coverage ölçümü yapar.
        """
        if function_range is None:
            return self._coverage_service.measure(
                source_file=source_file,
                test_file=test_file,
                timeout_seconds=timeout_seconds,
            )

        function_start_line, function_end_line = (
            function_range
        )

        return self._coverage_service.measure_function(
            source_file=source_file,
            test_file=test_file,
            function_name=function_name,
            start_line=function_start_line,
            end_line=function_end_line,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _create_test_file_name(
        function_name: str,
    ) -> str:
        """Kümülatif test paketi için kararlı dosya adı oluşturur."""
        return (
            f"test_{function_name}_scenario_suite.py"
        )

    @staticmethod
    def _normalize_source_file(
        source_file: str | Path,
    ) -> Path:
        """Kaynak dosya yolunu doğrular."""
        if not isinstance(
            source_file,
            (
                str,
                Path,
            ),
        ):
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
        """Python modül yolunu doğrular."""
        if not isinstance(
            module_path,
            str,
        ):
            raise TypeError(
                "module_path string olmalıdır."
            )

        normalized_module_path = (
            module_path.strip()
        )

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
                "Geçersiz Python modül yolu: "
                f"{normalized_module_path}"
            )

        return normalized_module_path

    @classmethod
    def _normalize_function_name(
        cls,
        function_name: str,
    ) -> str:
        """Fonksiyon adını doğrular."""
        if not isinstance(
            function_name,
            str,
        ):
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
    def _normalize_scenarios(
        scenarios: tuple[Scenario, ...],
    ) -> tuple[Scenario, ...]:
        """Kümülatif test paketindeki senaryoları doğrular."""
        if not isinstance(
            scenarios,
            tuple,
        ):
            raise TypeError(
                "scenarios bir Scenario tuple'ı olmalıdır."
            )

        if not scenarios:
            raise ValueError(
                "En az bir Scenario gereklidir."
            )

        if any(
            not isinstance(
                scenario,
                Scenario,
            )
            for scenario in scenarios
        ):
            raise TypeError(
                "scenarios yalnızca Scenario nesneleri "
                "içermelidir."
            )

        scenario_ids = tuple(
            scenario.scenario_id
            for scenario in scenarios
        )

        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError(
                "scenarios tekrar eden scenario_id içeremez."
            )

        return scenarios

    @staticmethod
    def _normalize_output_directory(
        output_directory: str | Path,
    ) -> Path:
        """Çıktı klasörünü doğrular."""
        if not isinstance(
            output_directory,
            (
                str,
                Path,
            ),
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

        return Path(output_directory)

    @staticmethod
    def _normalize_function_range(
        *,
        function_start_line: int | None,
        function_end_line: int | None,
    ) -> tuple[int, int] | None:
        """
        Fonksiyon başlangıç ve bitiş satırlarını doğrular.

        İki değer de None ise dosya bazlı coverage kullanılacağını
        belirtmek için None döndürür.
        """
        if (
            function_start_line is None
            and function_end_line is None
        ):
            return None

        if (
            function_start_line is None
            or function_end_line is None
        ):
            raise ValueError(
                "function_start_line ve function_end_line "
                "birlikte verilmelidir."
            )

        if (
            isinstance(function_start_line, bool)
            or not isinstance(function_start_line, int)
        ):
            raise TypeError(
                "function_start_line bir tam sayı olmalıdır."
            )

        if (
            isinstance(function_end_line, bool)
            or not isinstance(function_end_line, int)
        ):
            raise TypeError(
                "function_end_line bir tam sayı olmalıdır."
            )

        if function_start_line < 1:
            raise ValueError(
                "function_start_line 1 veya daha büyük olmalıdır."
            )

        if function_end_line < function_start_line:
            raise ValueError(
                "function_end_line, function_start_line "
                "değerinden küçük olamaz."
            )

        return (
            function_start_line,
            function_end_line,
        )

    @staticmethod
    def _validate_overwrite(
        overwrite: bool,
    ) -> bool:
        """Dosya üzerine yazma seçeneğini doğrular."""
        if not isinstance(
            overwrite,
            bool,
        ):
            raise TypeError(
                "overwrite bool olmalıdır."
            )

        return overwrite

    @staticmethod
    def _validate_timeout(
        timeout_seconds: float,
    ) -> float:
        """Coverage zaman aşımı değerini doğrular."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(
                timeout_seconds,
                (
                    int,
                    float,
                ),
            )
        ):
            raise TypeError(
                "timeout_seconds sayısal olmalıdır."
            )

        normalized_timeout = float(
            timeout_seconds
        )

        if normalized_timeout <= 0:
            raise ValueError(
                "timeout_seconds sıfırdan büyük olmalıdır."
            )

        return normalized_timeout

    @staticmethod
    def _is_valid_identifier(
        value: str,
    ) -> bool:
        """Geçerli Python tanımlayıcısını kontrol eder."""
        return (
            value.isidentifier()
            and not keyword.iskeyword(value)
        )