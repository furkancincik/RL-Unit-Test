from __future__ import annotations

import keyword
from pathlib import Path

from generator.scenario_generator import Scenario
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

    Akış:

        CoverageState + Yeni Scenario
                    ↓
        Seçilmiş Scenario listesine ekleme
                    ↓
        ScenarioSuiteCoverageService
                    ↓
        Kümülatif CoverageResult
                    ↓
        CoverageStateMapper
                    ↓
        Yeni CoverageState

    Bu sınıf aksiyon seçmez ve reward hesaplamaz. Yalnızca
    episode içerisindeki seçilmiş senaryoları yöneterek gerçek
    coverage tabanlı durum geçişini gerçekleştirir.
    """

    __slots__ = (
        "_source_file",
        "_module_path",
        "_function_name",
        "_output_directory",
        "_coverage_service",
        "_overwrite",
        "_timeout_seconds",
        "_selected_scenarios",
    )

    def __init__(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        output_directory: str | Path,
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

    def __call__(
        self,
        state: CoverageState,
        scenario: Scenario,
    ) -> CoverageState:
        """
        Yeni senaryoyu episode paketine ekler ve kümülatif
        coverage sonucundan yeni state oluşturur.

        Args:
            state:
                Yeni senaryo seçilmeden önceki RL durumu.

            scenario:
                RL ajanı tarafından seçilmiş test senaryosu.

        Returns:
            Şimdiye kadar seçilmiş bütün senaryoların gerçek
            coverage sonucundan oluşturulan yeni CoverageState.

        Raises:
            TypeError:
                state veya scenario geçersiz türdeyse.

            ValueError:
                Aynı senaryo episode içerisinde tekrar seçilirse.

            RuntimeError:
                Pytest veya coverage işlemi başarısız olursa.
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
                    overwrite=self._overwrite,
                    timeout_seconds=self._timeout_seconds,
                )
            )
        except Exception:
            # Coverage işlemi tamamlanamazsa seçimi geri alır.
            self._selected_scenarios.pop()
            raise

        if not suite_result.success:
            self._selected_scenarios.pop()

            raise RuntimeError(
                "Seçilen senaryo paketinin coverage işlemi "
                "başarısız oldu."
            )

        return CoverageStateMapper.map(
            coverage_result=suite_result.coverage,
            executed_tests=state.executed_tests + 1,
        )

    def reset(self) -> None:
        """
        Episode boyunca biriktirilen bütün senaryoları temizler.

        Yeni bir episode başlamadan önce çağrılmalıdır.
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

        normalized_function_name = function_name.strip()

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