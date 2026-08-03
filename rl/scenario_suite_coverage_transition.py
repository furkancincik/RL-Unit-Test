from __future__ import annotations

import keyword
from pathlib import Path

from generator.scenario_generator import Scenario
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from rl.coverage_state import CoverageState
from rl.coverage_state_mapper import CoverageStateMapper
from services.scenario_suite_coverage_service import (
    ScenarioSuiteCoverageService,
)


class ScenarioSuiteCoverageTransition:
    """
    Episode boyunca seçilen test senaryolarını biriktirir ve
    her adımda bütün seçilmiş senaryoların kümülatif coverage
    sonucundan yeni CoverageState oluşturur.

    Fonksiyon başlangıç ve bitiş satırları verilirse coverage
    yalnızca hedef fonksiyon için hesaplanır. Satır aralığı
    verilmezse eski davranış korunarak dosya geneli coverage
    sonucu kullanılır.

    Akış:

        CoverageState + Yeni Scenario
                    ↓
        Seçilmiş Scenario listesine ekleme
                    ↓
        ScenarioSuiteCoverageService
                    ↓
        Dosya veya fonksiyon bazlı coverage
                    ↓
        CoverageStateMapper
                    ↓
        Yeni CoverageState
    """

    __slots__ = (
        "_source_file",
        "_module_path",
        "_function_name",
        "_function_start_line",
        "_function_end_line",
        "_output_directory",
        "_coverage_service",
        "_overwrite",
        "_timeout_seconds",
        "_selected_scenarios",
        "_last_coverage_result",
    )

    def __init__(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        output_directory: str | Path,
        function_start_line: int | None = None,
        function_end_line: int | None = None,
        coverage_service: ScenarioSuiteCoverageService | None = None,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Kümülatif coverage geçiş bileşenini hazırlar."""
        self._source_file = self._normalize_source_file(
            source_file
        )

        self._module_path = self._normalize_module_path(
            module_path
        )

        self._function_name = self._normalize_function_name(
            function_name
        )

        (
            self._function_start_line,
            self._function_end_line,
        ) = self._normalize_function_range(
            function_start_line=function_start_line,
            function_end_line=function_end_line,
        )

        self._output_directory = (
            self._normalize_output_directory(
                output_directory
            )
        )

        self._coverage_service = self._normalize_service(
            coverage_service
        )

        self._overwrite = self._validate_overwrite(
            overwrite
        )

        self._timeout_seconds = self._validate_timeout(
            timeout_seconds
        )

        self._selected_scenarios: list[Scenario] = []
        self._last_coverage_result: (
            CoverageResult | FunctionCoverageResult | None
        ) = None

    @property
    def source_file(self) -> Path:
        """Coverage ölçümü yapılan kaynak dosyayı döndürür."""
        return self._source_file

    @property
    def module_path(self) -> str:
        """Kaynak dosyanın Python modül yolunu döndürür."""
        return self._module_path

    @property
    def function_name(self) -> str:
        """Test edilen fonksiyonun adını döndürür."""
        return self._function_name

    @property
    def function_start_line(self) -> int | None:
        """Hedef fonksiyonun başlangıç satırını döndürür."""
        return self._function_start_line

    @property
    def function_end_line(self) -> int | None:
        """Hedef fonksiyonun bitiş satırını döndürür."""
        return self._function_end_line

    @property
    def uses_function_coverage(self) -> bool:
        """Transition fonksiyon bazlı coverage kullanıyorsa True döndürür."""
        return (
            self._function_start_line is not None
            and self._function_end_line is not None
        )

    @property
    def output_directory(self) -> Path:
        """Üretilen test paketinin yazıldığı klasörü döndürür."""
        return self._output_directory

    @property
    def selected_scenarios(self) -> tuple[Scenario, ...]:
        """
        Episode içerisinde şu ana kadar seçilmiş senaryoları
        seçim sırasına göre döndürür.
        """
        return tuple(self._selected_scenarios)

    @property
    def selected_scenario_count(self) -> int:
        """Episode içerisinde seçilmiş senaryo sayısını döndürür."""
        return len(self._selected_scenarios)

    @property
    def last_coverage_result(
        self,
    ) -> CoverageResult | FunctionCoverageResult | None:
        """
        Son başarılı coverage ölçüm sonucunu döndürür.

        Henüz başarılı bir ölçüm yapılmadıysa None döner.
        """
        return self._last_coverage_result

    @property
    def last_function_coverage(
        self,
    ) -> FunctionCoverageResult | None:
        """
        Son başarılı sonuç fonksiyon bazlıysa onu döndürür.
        """
        if isinstance(
            self._last_coverage_result,
            FunctionCoverageResult,
        ):
            return self._last_coverage_result

        return None

    @property
    def last_file_coverage(
        self,
    ) -> CoverageResult | None:
        """
        Son başarılı ölçüme ait dosya geneli coverage sonucunu döndürür.

        Fonksiyon bazlı ölçümde FunctionCoverageResult içerisindeki
        file_coverage alanı kullanılır.
        """
        if isinstance(
            self._last_coverage_result,
            FunctionCoverageResult,
        ):
            return self._last_coverage_result.file_coverage

        if isinstance(
            self._last_coverage_result,
            CoverageResult,
        ):
            return self._last_coverage_result

        return None

    def __call__(
        self,
        state: CoverageState,
        scenario: Scenario,
    ) -> CoverageState:
        """
        Yeni senaryoyu episode paketine ekler ve kümülatif
        coverage sonucundan yeni state oluşturur.

        Coverage işlemi başarısız olursa eklenen senaryo listeden
        çıkarılarak transition önceki durumuna geri döndürülür.
        """
        self._validate_state(state)
        self._validate_scenario(scenario)

        if scenario in self._selected_scenarios:
            raise ValueError(
                "Scenario episode içerisinde daha önce seçildi: "
                f"{scenario.scenario_id}"
            )

        self._selected_scenarios.append(
            scenario
        )

        try:
            suite_result = (
                self._coverage_service.measure_scenarios(
                    source_file=self._source_file,
                    module_path=self._module_path,
                    function_name=self._function_name,
                    scenarios=self.selected_scenarios,
                    output_directory=self._output_directory,
                    function_start_line=self._function_start_line,
                    function_end_line=self._function_end_line,
                    overwrite=self._overwrite,
                    timeout_seconds=self._timeout_seconds,
                )
            )
        except Exception:
            self._selected_scenarios.pop()
            raise

        if not suite_result.success:
            self._selected_scenarios.pop()

            raise RuntimeError(
                "Seçilen senaryo paketinin coverage işlemi "
                "başarısız oldu."
            )

        self._last_coverage_result = suite_result.coverage

        return CoverageStateMapper.map(
            coverage_result=suite_result.coverage,
            executed_tests=state.executed_tests + 1,
        )

    def reset(self) -> None:
        """
        Episode boyunca biriktirilen bütün senaryoları temizler.

        Q-Table, RL ajanı ve son başarılı coverage sonucu bu işlemden
        etkilenmez. Yalnızca geçerli episode içerisinde seçilen
        senaryolar temizlenir.
        """
        self._selected_scenarios.clear()

    @staticmethod
    def _normalize_source_file(
        source_file: str | Path,
    ) -> Path:
        """Kaynak dosya yolunu doğrular."""
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
    def _normalize_function_range(
        *,
        function_start_line: int | None,
        function_end_line: int | None,
    ) -> tuple[int | None, int | None]:
        """
        Fonksiyon başlangıç ve bitiş satırlarını doğrular.

        İki değer de None ise eski dosya bazlı coverage davranışı
        korunur. Fonksiyon bazlı coverage için iki değer birlikte
        verilmelidir.
        """
        if (
            function_start_line is None
            and function_end_line is None
        ):
            return None, None

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

        return Path(output_directory)

    @staticmethod
    def _normalize_service(
        service: ScenarioSuiteCoverageService | None,
    ) -> ScenarioSuiteCoverageService:
        """Kümülatif coverage servisini doğrular."""
        if service is None:
            return ScenarioSuiteCoverageService()

        if not isinstance(
            service,
            ScenarioSuiteCoverageService,
        ):
            raise TypeError(
                "coverage_service bir "
                "ScenarioSuiteCoverageService örneği olmalıdır."
            )

        return service

    @staticmethod
    def _validate_overwrite(
        overwrite: bool,
    ) -> bool:
        """Dosya üzerine yazma seçeneğini doğrular."""
        if not isinstance(overwrite, bool):
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
                (int, float),
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
    def _validate_state(
        state: CoverageState,
    ) -> None:
        """Mevcut RL durumunu doğrular."""
        if not isinstance(state, CoverageState):
            raise TypeError(
                "state bir CoverageState örneği olmalıdır."
            )

    @staticmethod
    def _validate_scenario(
        scenario: Scenario,
    ) -> None:
        """Seçilen test senaryosunu doğrular."""
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "scenario bir Scenario örneği olmalıdır."
            )

    @staticmethod
    def _is_valid_identifier(
        value: str,
    ) -> bool:
        """Geçerli Python tanımlayıcısını kontrol eder."""
        return (
            value.isidentifier()
            and not keyword.iskeyword(value)
        )