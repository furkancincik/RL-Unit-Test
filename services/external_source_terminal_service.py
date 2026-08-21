from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalExecutionPolicy,
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    InlinePythonSource,
    LocalProjectDirectory,
    PublicGitHubRepository,
    UploadedPythonFile,
)
from services.external_source_analysis_service import (
    ExternalSourceAnalysisService,
    ExternalSourceAnalysisValidationError,
)


class ExternalSourceInteractiveValidationError(ValueError):
    """Terminal girdisi güvenli bir external-source isteğine çevrilemedi."""


class ExternalSourceTerminalAdapter:
    """External source production API'sini state tutmadan terminale bağlar."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        service_factory: Callable[[], ExternalSourceAnalysisService] = (
            ExternalSourceAnalysisService
        ),
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._service_factory = service_factory

    def run_menu(self) -> None:
        while True:
            self._print_submenu()
            choice = self._input("\nSeçiminiz: ").strip()
            if choice == "0":
                return
            if choice not in {"1", "2", "3", "4"}:
                self._output("Geçersiz dış kaynak seçimi. Lütfen 0, 1, 2, 3 veya 4 girin.")
                continue
            try:
                source = self._source(choice)
                policy = self._execution_policy()
                if policy is None:
                    self._output("Dış kaynak analizi iptal edildi.")
                    continue
                configuration = self._configuration(policy)
                request = ExternalSourceAnalysisRequest(
                    source=source,
                    execution_policy=policy,
                    configuration=configuration,
                )
                result = self._service_factory().run(request)
                self._output(self.format_result(result))
            except ExternalSourceInteractiveValidationError as error:
                self._output(f"Dış kaynak doğrulama hatası: {error}")
            except ExternalSourceAnalysisValidationError as error:
                self._output(f"Dış kaynak doğrulama hatası: {error}")
            except KeyboardInterrupt:
                self._output("Dış kaynak analizi güvenli biçimde iptal edildi.")

    def _source(self, choice: str) -> object:
        if choice == "1":
            return self._inline_source()
        if choice == "2":
            return self._uploaded_file()
        if choice == "3":
            return self._local_directory()
        repository_url = self._input("Public GitHub repository URL: ").strip()
        if not repository_url:
            raise ExternalSourceInteractiveValidationError("GitHub URL boş olamaz.")
        return PublicGitHubRepository(repository_url=repository_url)

    def _inline_source(self) -> InlinePythonSource:
        self._output(
            "Python kodunu girin. Girişi tek başına __END__ yazan satırla tamamlayın."
        )
        lines: list[str] = []
        while True:
            line = self._input("")
            if line == "__END__":
                break
            lines.append(line)
        source_text = "\n".join(lines)
        if not source_text.strip():
            raise ExternalSourceInteractiveValidationError("Python kaynak kodu boş olamaz.")
        try:
            return InlinePythonSource(source_text=source_text)
        except (TypeError, ValueError) as error:
            raise ExternalSourceInteractiveValidationError(str(error)) from error

    def _uploaded_file(self) -> UploadedPythonFile:
        path = self._normalized_path(self._input("Python dosya yolu: ").strip())
        if path.suffix.lower() != ".py":
            raise ExternalSourceInteractiveValidationError(
                "Python dosyası .py uzantılı olmalıdır."
            )
        if not path.is_file():
            raise ExternalSourceInteractiveValidationError(
                "Python dosyası bulunamadı veya normal dosya değil."
            )
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ExternalSourceInteractiveValidationError(
                "Python dosyası okunamadı."
            ) from error
        return UploadedPythonFile(original_filename=path.name, file_bytes=payload)

    def _local_directory(self) -> LocalProjectDirectory:
        path = self._normalized_path(self._input("Yerel proje klasörü: ").strip())
        if not path.is_dir():
            raise ExternalSourceInteractiveValidationError(
                "Yerel proje yolu mevcut bir klasör olmalıdır."
            )
        return LocalProjectDirectory(directory=path)

    @staticmethod
    def _normalized_path(raw_value: str) -> Path:
        if not raw_value:
            raise ExternalSourceInteractiveValidationError("Kaynak yolu boş olamaz.")
        return Path(raw_value).expanduser().resolve()

    def _execution_policy(self) -> ExternalExecutionPolicy | None:
        self._output("\nANALİZ POLİTİKASI")
        self._output("1. Güvenli statik keşif (önerilen)")
        self._output("2. Güvenilir dinamik analiz")
        self._output("0. İptal")
        choice = self._input("Analiz modu [1]: ").strip() or "1"
        if choice == "0":
            return None
        if choice == "1":
            return ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY
        if choice != "2":
            raise ExternalSourceInteractiveValidationError(
                "Analiz politikası 0, 1 veya 2 olmalıdır."
            )
        self._output("UYARI: Kaynak kod çalıştırılacaktır.")
        self._output("Timeout bir sandbox değildir.")
        self._output("Yalnız güvenilen kodlarda kullanılmalıdır.")
        confirmation = self._input("Devam etmek için EVET yazın: ").strip()
        if confirmation != "EVET":
            return None
        return ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS

    def _configuration(
        self,
        policy: ExternalExecutionPolicy,
    ) -> ExternalAnalysisConfiguration:
        suggested_output = (Path("output") / "external_source_analysis").resolve()
        output_value = self._input(f"Output root [{suggested_output}]: ").strip()
        output_root = Path(output_value).expanduser().resolve() if output_value else suggested_output
        defaults = ExternalAnalysisConfiguration(output_root=output_root)
        module_selection = self._module_selection()
        maximum_modules = self._positive_int(
            "Maximum module count", defaults.maximum_selected_modules
        )
        maximum_functions = self._positive_int(
            "Maximum function count", defaults.maximum_functions_per_module
        )
        if policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
            return ExternalAnalysisConfiguration(
                output_root=output_root,
                module_selection=module_selection,
                maximum_selected_modules=maximum_modules,
                maximum_functions_per_module=maximum_functions,
            )

        pipeline_timeout = self._positive_float(
            "Function pipeline timeout",
            defaults.per_function_pipeline_timeout_seconds,
        )
        episode_count = self._positive_int("Episode count", defaults.episode_count)
        random_seed = self._integer("Random seed", defaults.random_seed)
        greedy = self._boolean("Greedy minimization", defaults.run_greedy_baseline)
        comparison = self._boolean(
            "RL-greedy comparison", defaults.run_strategy_comparison
        )
        return ExternalAnalysisConfiguration(
            output_root=output_root,
            module_selection=module_selection,
            maximum_selected_modules=maximum_modules,
            maximum_functions_per_module=maximum_functions,
            episode_count=episode_count,
            random_seed=random_seed,
            per_function_pipeline_timeout_seconds=pipeline_timeout,
            run_greedy_baseline=greedy or comparison,
            run_strategy_comparison=comparison,
        )

    def _module_selection(self) -> ExternalModuleSelection:
        self._output("\nMODÜL SEÇİMİ")
        self._output("1. Tüm uygun modüller (limitli)")
        self._output("2. Explicit relative path")
        self._output("3. Explicit module name")
        choice = self._input("Module selection [1]: ").strip() or "1"
        modes = {
            "1": ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT,
            "2": ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS,
            "3": ExternalModuleSelectionMode.EXPLICIT_MODULE_NAMES,
        }
        if choice not in modes:
            raise ExternalSourceInteractiveValidationError(
                "Module selection 1, 2 veya 3 olmalıdır."
            )
        mode = modes[choice]
        if mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            return ExternalModuleSelection(mode=mode)
        raw_values = self._input("Virgülle ayrılmış seçimler: ")
        values = tuple(
            dict.fromkeys(value.strip() for value in raw_values.split(",") if value.strip())
        )
        if not values:
            raise ExternalSourceInteractiveValidationError(
                "Explicit module selection boş olamaz."
            )
        try:
            return ExternalModuleSelection(mode=mode, values=values)
        except (TypeError, ValueError) as error:
            raise ExternalSourceInteractiveValidationError(str(error)) from error

    def _positive_int(self, label: str, default: int) -> int:
        raw_value = self._input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ExternalSourceInteractiveValidationError(
                f"{label} pozitif tam sayı olmalıdır."
            ) from error
        if value <= 0:
            raise ExternalSourceInteractiveValidationError(
                f"{label} pozitif tam sayı olmalıdır."
            )
        return value

    def _positive_float(self, label: str, default: float | None) -> float:
        raw_value = self._input(f"{label} [{default}]: ").strip()
        if not raw_value and default is not None:
            return float(default)
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ExternalSourceInteractiveValidationError(
                f"{label} pozitif sayı olmalıdır."
            ) from error
        if not 0.0 < value < float("inf"):
            raise ExternalSourceInteractiveValidationError(
                f"{label} pozitif sayı olmalıdır."
            )
        return value

    def _integer(self, label: str, default: int | None) -> int | None:
        raw_value = self._input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            return int(raw_value)
        except ValueError as error:
            raise ExternalSourceInteractiveValidationError(
                f"{label} tam sayı olmalıdır."
            ) from error

    def _boolean(self, label: str, default: bool) -> bool:
        default_label = "e" if default else "h"
        raw_value = self._input(f"{label} [e/h, {default_label}]: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in {"e", "evet"}:
            return True
        if raw_value in {"h", "hayır", "hayir"}:
            return False
        raise ExternalSourceInteractiveValidationError(f"{label} için e veya h girilmelidir.")

    @staticmethod
    def format_result(result: ExternalSourceAnalysisResult) -> str:
        if not isinstance(result, ExternalSourceAnalysisResult):
            raise TypeError("result ExternalSourceAnalysisResult olmalıdır.")
        module_counts: dict[str, int] = {}
        functions: list[object] = []
        for module in result.module_results:
            module_counts[module.status.value] = module_counts.get(module.status.value, 0) + 1
            functions.extend(
                getattr(module.project_result, "function_results", ())
                if module.project_result is not None
                else ()
            )
        unsupported_modules = module_counts.get("UNSUPPORTED", 0)
        eligible_modules = max(0, result.discovered_module_count - unsupported_modules)
        lines = [
            "=" * 65,
            "DIŞ KAYNAK ANALİZ ÖZETİ",
            "=" * 65,
            f"Kaynak türü          : {result.source_kind.value}",
            f"Acquisition durumu   : {result.acquisition_status}",
            f"Analiz politikası    : {result.execution_policy.value}",
            f"Genel durum          : {result.status.value}",
            f"Keşfedilen modül     : {result.discovered_module_count}",
            f"Uygun modül          : {eligible_modules}",
            f"Uygunsuz modül       : {unsupported_modules}",
            f"Seçilen modül        : {result.selected_module_count}",
            f"Modül durumları      : {module_counts}",
            f"Issue kategorileri   : {', '.join(result.issues) if result.issues else 'Yok'}",
            f"Keşfedilen fonksiyon : {result.discovered_function_count}",
            f"Çalıştırılan fonksiyon: {result.analyzed_function_count}",
            f"SKIPPED_LIMIT        : {result.limit_skipped_function_count}",
        ]
        for function in functions:
            diagnostic = getattr(function, "diagnostic", None)
            comparison = getattr(function, "strategy_comparison", None)
            lines.extend(
                (
                    f"Fonksiyon            : {function.target.qualified_name}",
                    f"  Durum              : {function.status.value}",
                    f"  Scenario           : {ExternalSourceTerminalAdapter._measured(getattr(function, 'scenario_count', None))}",
                    f"  RL test            : {ExternalSourceTerminalAdapter._measured(getattr(function, 'rl_test_count', None))}",
                    f"  Greedy seçilen     : {ExternalSourceTerminalAdapter._measured(getattr(comparison, 'greedy_selected_count', None))}",
                    f"  RL seçilen         : {ExternalSourceTerminalAdapter._measured(getattr(comparison, 'best_rl_executed_test_count', None))}",
                    f"  Winner             : {getattr(getattr(comparison, 'winner', None), 'value', 'Ölçülmedi')}",
                    f"  Line coverage      : {ExternalSourceTerminalAdapter._percentage(getattr(diagnostic, 'line_coverage_percent', None))}",
                    f"  Branch coverage    : {ExternalSourceTerminalAdapter._percentage(getattr(diagnostic, 'branch_coverage_percent', None))}",
                )
            )
        lines.extend(
            (
                "Project coverage      : Ölçülmedi",
                f"Artifact output root  : {result.output_root}",
                f"JSON raporu           : {result.report_path}",
                f"Cleanup durumu        : {result.cleanup_status.value}",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _percentage(value: float | None) -> str:
        return "Ölçülmedi" if value is None else f"%{value:.2f}"

    @staticmethod
    def _measured(value: object | None) -> str:
        return "Ölçülmedi" if value is None else str(value)

    def _print_submenu(self) -> None:
        self._output("\nDIŞ KAYNAK ANALİZİ")
        self._output("1. Python kodu yapıştır")
        self._output("2. Python dosyası seç")
        self._output("3. Yerel proje klasörü seç")
        self._output("4. Public GitHub URL gir")
        self._output("0. Ana menüye dön")
