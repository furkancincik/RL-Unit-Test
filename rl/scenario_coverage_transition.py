from __future__ import annotations

from pathlib import Path

from generator.scenario_generator import Scenario
from rl.coverage_state import CoverageState
from rl.coverage_state_mapper import CoverageStateMapper
from services.scenario_coverage_service import (
    ScenarioCoverageService,
)


class ScenarioCoverageTransition:
    """
    Seçilen test senaryosunu gerçek coverage hattında çalıştırarak
    yeni bir CoverageState oluşturan geçiş bileşenidir.

    Akış:

        CoverageState + Scenario
                    ↓
        ScenarioCoverageService
                    ↓
        ScenarioCoverageResult
                    ↓
        CoverageStateMapper
                    ↓
        Yeni CoverageState

    Bu sınıf reward hesaplamaz ve aksiyon seçmez. Yalnızca seçilen
    senaryonun uygulanması sonucunda ortamın yeni durumunu üretir.
    """

    __slots__ = (
        "_source_file",
        "_module_path",
        "_function_name",
        "_output_directory",
        "_scenario_coverage_service",
        "_overwrite",
        "_timeout_seconds",
    )

    def __init__(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        output_directory: str | Path,
        scenario_coverage_service: ScenarioCoverageService | None = None,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Geçiş bileşeninin yapılandırmasını hazırlar.

        Args:
            source_file:
                Coverage ölçümü yapılacak Python kaynak dosyası.

            module_path:
                Kaynak dosyanın Python import yolu.

            function_name:
                Test edilecek fonksiyonun adı.

            output_directory:
                Seçilen senaryoya ait pytest dosyasının yazılacağı klasör.

            scenario_coverage_service:
                Senaryoyu pytest ve coverage hattında çalıştıran servis.

            overwrite:
                Aynı isimli test dosyasının üzerine yazılmasına izin verir.

            timeout_seconds:
                Coverage işleminin azami çalışma süresi.
        """
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
        self._scenario_coverage_service = (
            self._normalize_service(
                scenario_coverage_service
            )
        )
        self._overwrite = self._validate_overwrite(
            overwrite
        )
        self._timeout_seconds = self._validate_timeout(
            timeout_seconds
        )

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
        """Üretilen pytest dosyalarının klasörünü döndürür."""
        return self._output_directory

    def __call__(
        self,
        state: CoverageState,
        scenario: Scenario,
    ) -> CoverageState:
        """
        Senaryoyu gerçek coverage hattında çalıştırır ve yeni state üretir.

        Args:
            state:
                Senaryo çalıştırılmadan önceki coverage durumu.

            scenario:
                RL ajanının seçtiği test senaryosu.

        Returns:
            Gerçek coverage sonucu kullanılarak oluşturulan yeni
            CoverageState.

        Raises:
            TypeError:
                state veya scenario geçersiz türdeyse.

            RuntimeError:
                Coverage altında çalıştırılan pytest başarısız olursa.
        """
        self._validate_state(state)
        self._validate_scenario(scenario)

        scenario_result = (
            self._scenario_coverage_service.measure_scenario(
                source_file=self._source_file,
                module_path=self._module_path,
                function_name=self._function_name,
                scenario=scenario,
                output_directory=self._output_directory,
                overwrite=self._overwrite,
                timeout_seconds=self._timeout_seconds,
            )
        )

        if not scenario_result.success:
            raise RuntimeError(
                "Seçilen senaryonun coverage işlemi "
                "başarısız oldu: "
                f"{scenario.scenario_id}"
            )

        return CoverageStateMapper.map(
            coverage_result=scenario_result.coverage,
            executed_tests=state.executed_tests + 1,
        )

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

        parts = normalized_module_path.split(".")

        if any(
            not part.isidentifier()
            for part in parts
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

        normalized_function_name = (
            function_name.strip()
        )

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

        return Path(output_directory)

    @staticmethod
    def _normalize_service(
        service: ScenarioCoverageService | None,
    ) -> ScenarioCoverageService:
        """ScenarioCoverageService bağımlılığını doğrular."""
        if service is None:
            return ScenarioCoverageService()

        if not isinstance(
            service,
            ScenarioCoverageService,
        ):
            raise TypeError(
                "scenario_coverage_service bir "
                "ScenarioCoverageService örneği olmalıdır."
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
        if not isinstance(
            state,
            CoverageState,
        ):
            raise TypeError(
                "state bir CoverageState örneği olmalıdır."
            )

    @staticmethod
    def _validate_scenario(
        scenario: Scenario,
    ) -> None:
        """Seçilen test senaryosunu doğrular."""
        if not isinstance(
            scenario,
            Scenario,
        ):
            raise TypeError(
                "scenario bir Scenario örneği olmalıdır."
            )