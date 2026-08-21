from __future__ import annotations

import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from analyzer.python_analyzer import FunctionInfo, PythonAnalyzer
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineRunStatus,
)
from models.project_analysis_result import (
    FunctionAnalysisResult,
    FunctionRunStatus,
    FunctionSelectionMode,
    FunctionTarget,
    ProjectAnalysisResult,
    ProjectRunStatus,
)
from services.project_analysis_report_service import ProjectAnalysisReportWriter
from services.real_rl_training_service import (
    RealRLTrainingResult,
    RealRLTrainingService,
)


class SourceAnalysisValidationError(ValueError):
    """Kullanıcının source orchestration isteği geçersiz olduğunda oluşur."""


TrainingServiceFactory = Callable[[], RealRLTrainingService]


class SourceAnalysisOrchestrator:
    """Keşfedilen fonksiyonları bağımsız production pipeline'larda çalıştırır."""

    def __init__(
        self,
        *,
        analyzer: PythonAnalyzer | None = None,
        training_service_factory: TrainingServiceFactory | None = None,
        report_writer: ProjectAnalysisReportWriter | None = None,
    ) -> None:
        self._analyzer = analyzer or PythonAnalyzer()
        self._training_service_factory = (
            training_service_factory or RealRLTrainingService
        )
        self._report_writer = report_writer or ProjectAnalysisReportWriter()

    def run(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        output_root: str | Path,
        function_name: str | None,
        all_functions: bool,
        max_visits_per_node: int = 3,
        episode_count: int = 3,
        epsilon: float = 0.0,
        learning_rate: float = 0.5,
        discount_factor: float = 0.9,
        random_seed: int | None = 42,
        overwrite: bool = True,
        timeout_seconds: float = 30.0,
        per_function_timeout_seconds: float | None = None,
        maximum_functions: int | None = None,
        import_root: str | Path | None = None,
        run_greedy_baseline: bool = False,
        run_strategy_comparison: bool = False,
        comparison_timeout_seconds: float | None = None,
    ) -> ProjectAnalysisResult:
        started_at = time.perf_counter()
        normalized_source = self._normalize_source_file(source_file)
        normalized_module = self._normalize_module_path(module_path)
        normalized_output = self._normalize_output_root(output_root)
        normalized_import_root = self._normalize_import_root(
            import_root,
            normalized_source,
        )
        self._validate_selection(function_name, all_functions)
        self._validate_maximum_functions(maximum_functions)
        self._validate_optional_timeout(per_function_timeout_seconds)
        self._validate_optional_timeout(comparison_timeout_seconds)
        if not isinstance(run_greedy_baseline, bool):
            raise SourceAnalysisValidationError(
                "run_greedy_baseline bool olmalıdır."
            )
        if not isinstance(run_strategy_comparison, bool):
            raise SourceAnalysisValidationError(
                "run_strategy_comparison bool olmalıdır."
            )

        analysis = self._analyzer.analyze_file(normalized_source)
        discovered_targets = self._resolve_duplicate_targets(
            tuple(self._to_target(function) for function in analysis.functions)
        )
        selected_targets, selection_mode = self._select_targets(
            discovered_targets,
            function_name=function_name,
            all_functions=all_functions,
        )
        limit_skipped_ordinals = self._limit_skipped_ordinals(
            selected_targets,
            maximum_functions,
        )
        report_path = normalized_output / "project_analysis_report.json"
        function_results: list[FunctionAnalysisResult] = []

        for ordinal, target in enumerate(selected_targets, start=1):
            function_output = self._function_output_directory(
                normalized_output, target.name, ordinal
            )
            if not target.is_supported:
                function_results.append(
                    FunctionAnalysisResult(
                        target=target,
                        status=FunctionRunStatus.UNSUPPORTED,
                        diagnostic=None,
                        output_directory=function_output,
                        skip_reason=target.unsupported_reason,
                    )
                )
                continue

            if ordinal in limit_skipped_ordinals:
                function_results.append(
                    FunctionAnalysisResult(
                        target=target,
                        status=FunctionRunStatus.SKIPPED_LIMIT,
                        diagnostic=None,
                        output_directory=function_output,
                        skip_reason="FUNCTION_LIMIT_EXCEEDED",
                    )
                )
                continue

            service = self._training_service_factory()
            pipeline_result = service.run_with_diagnostics(
                source_file=normalized_source,
                module_path=normalized_module,
                function_name=target.name,
                output_directory=function_output,
                max_visits_per_node=max_visits_per_node,
                episode_count=episode_count,
                epsilon=epsilon,
                epsilon_decay_rate=None,
                minimum_epsilon=0.0,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                random_seed=random_seed,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
                pipeline_timeout_seconds=per_function_timeout_seconds,
                import_root=normalized_import_root,
                run_greedy_baseline=run_greedy_baseline,
                run_strategy_comparison=run_strategy_comparison,
                comparison_timeout_seconds=comparison_timeout_seconds,
            )
            diagnostic = self._extract_diagnostic(pipeline_result)
            function_results.append(
                FunctionAnalysisResult(
                    target=target,
                    status=self._function_status(diagnostic.status),
                    diagnostic=diagnostic,
                    output_directory=function_output,
                    artifact_paths=self._artifact_paths(function_output),
                    strategy_comparison=(
                        pipeline_result.strategy_comparison_result
                        if isinstance(pipeline_result, RealRLTrainingResult)
                        else None
                    ),
                )
            )

        statuses = tuple(item.status for item in function_results)
        project_result = ProjectAnalysisResult(
            source_file=normalized_source,
            module_path=normalized_module,
            selection_mode=selection_mode,
            discovered_targets=discovered_targets,
            function_results=tuple(function_results),
            total_duration_seconds=max(0.0, time.perf_counter() - started_at),
            status=ProjectAnalysisResult.derive_status(statuses),
            output_root=normalized_output,
            report_path=report_path,
        )
        self._report_writer.write(project_result)
        return project_result

    @staticmethod
    def _to_target(function: FunctionInfo) -> FunctionTarget:
        return FunctionTarget(
            name=function.name,
            qualified_name=function.qualified_name or function.name,
            start_line=function.line_number,
            end_line=function.end_line_number or function.line_number,
            parameters=tuple(function.parameters),
            return_annotation=function.return_annotation,
            is_async=function.is_async,
            is_nested=function.is_nested,
            is_method=function.is_method,
            is_supported=function.is_supported,
            unsupported_reason=function.unsupported_reason,
        )

    @staticmethod
    def _resolve_duplicate_targets(
        targets: tuple[FunctionTarget, ...],
    ) -> tuple[FunctionTarget, ...]:
        supported_name_counts = {
            target.name: sum(
                candidate.is_supported and candidate.name == target.name
                for candidate in targets
            )
            for target in targets
            if target.is_supported
        }
        return tuple(
            replace(
                target,
                is_supported=False,
                unsupported_reason=(
                    "Duplicate top-level function names are ambiguous."
                ),
            )
            if target.is_supported and supported_name_counts[target.name] > 1
            else target
            for target in targets
        )

    @staticmethod
    def _select_targets(
        targets: tuple[FunctionTarget, ...],
        *,
        function_name: str | None,
        all_functions: bool,
    ) -> tuple[tuple[FunctionTarget, ...], FunctionSelectionMode]:
        if all_functions:
            if not targets:
                raise SourceAnalysisValidationError(
                    "Kaynak dosyada fonksiyon bulunamadı."
                )
            return targets, FunctionSelectionMode.ALL

        assert function_name is not None
        matching = tuple(target for target in targets if target.name == function_name)
        supported = tuple(target for target in matching if target.is_supported)
        if len(supported) == 1:
            return supported, FunctionSelectionMode.SINGLE
        if len(supported) > 1:
            raise SourceAnalysisValidationError(
                f"Birden fazla desteklenen {function_name!r} hedefi bulundu."
            )
        if matching:
            return (matching[0],), FunctionSelectionMode.SINGLE
        available = ", ".join(target.qualified_name for target in targets) or "Yok"
        raise SourceAnalysisValidationError(
            f"Fonksiyon bulunamadı: {function_name}. Kullanılabilir hedefler: {available}"
        )

    @staticmethod
    def _extract_diagnostic(
        result: RealRLTrainingResult | PipelineDiagnosticResult,
    ) -> PipelineDiagnosticResult:
        if isinstance(result, PipelineDiagnosticResult):
            return result
        if isinstance(result, RealRLTrainingResult) and result.diagnostic is not None:
            return result.diagnostic
        raise RuntimeError("Production function sonucu diagnostic içermiyor.")

    @staticmethod
    def _function_status(status: PipelineRunStatus) -> FunctionRunStatus:
        return FunctionRunStatus(status.value)

    @staticmethod
    def _artifact_paths(output_directory: Path) -> tuple[Path, ...]:
        if not output_directory.is_dir():
            return ()
        return tuple(
            sorted(
                (path.resolve() for path in output_directory.rglob("*") if path.is_file()),
                key=str,
            )
        )

    @staticmethod
    def _function_output_directory(
        output_root: Path,
        function_name: str,
        ordinal: int,
    ) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", function_name).strip("_")
        if not safe_name:
            safe_name = "function"
        candidate = (output_root / f"{ordinal:03d}_{safe_name}").resolve()
        if not candidate.is_relative_to(output_root):
            raise RuntimeError("Fonksiyon output yolu root dışına çıktı.")
        return candidate

    @staticmethod
    def _normalize_source_file(value: str | Path) -> Path:
        if not isinstance(value, (str, Path)):
            raise SourceAnalysisValidationError("source_file path olmalıdır.")
        path = Path(value)
        if path.suffix.lower() != ".py":
            raise SourceAnalysisValidationError("source_file bir .py dosyası olmalıdır.")
        if not path.is_file():
            raise SourceAnalysisValidationError(f"Kaynak dosya bulunamadı: {path}")
        return path.resolve()

    @staticmethod
    def _normalize_module_path(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SourceAnalysisValidationError("module_path boş olamaz.")
        normalized = value.strip()
        if any(not part.isidentifier() for part in normalized.split(".")):
            raise SourceAnalysisValidationError(f"Geçersiz module path: {normalized}")
        return normalized

    @staticmethod
    def _normalize_output_root(value: str | Path) -> Path:
        if not isinstance(value, (str, Path)):
            raise SourceAnalysisValidationError("output_root path olmalıdır.")
        if isinstance(value, str) and not value.strip():
            raise SourceAnalysisValidationError("output_root boş olamaz.")
        return Path(value).resolve()

    @staticmethod
    def _normalize_import_root(
        value: str | Path | None,
        source_file: Path,
    ) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise SourceAnalysisValidationError("import_root path olmalıdır.")
        if isinstance(value, str) and not value.strip():
            raise SourceAnalysisValidationError("import_root boş olamaz.")
        path = Path(value).resolve()
        if not path.is_dir():
            raise SourceAnalysisValidationError(
                "import_root var olan bir klasör olmalıdır."
            )
        if not source_file.is_relative_to(path):
            raise SourceAnalysisValidationError(
                "source_file import_root dışında olamaz."
            )
        return path

    @staticmethod
    def _validate_maximum_functions(value: int | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SourceAnalysisValidationError(
                "maximum_functions pozitif bir tam sayı olmalıdır."
            )

    @staticmethod
    def _limit_skipped_ordinals(
        targets: tuple[FunctionTarget, ...],
        maximum_functions: int | None,
    ) -> frozenset[int]:
        if maximum_functions is None:
            return frozenset()
        eligible_count = 0
        skipped: set[int] = set()
        for ordinal, target in enumerate(targets, start=1):
            if not target.is_supported:
                continue
            if eligible_count >= maximum_functions:
                skipped.add(ordinal)
                continue
            eligible_count += 1
        return frozenset(skipped)

    @staticmethod
    def _validate_selection(
        function_name: str | None,
        all_functions: bool,
    ) -> None:
        if not isinstance(all_functions, bool):
            raise SourceAnalysisValidationError("all_functions bool olmalıdır.")
        if function_name is not None:
            if not isinstance(function_name, str) or not function_name.strip():
                raise SourceAnalysisValidationError("function_name boş olamaz.")
            if not function_name.isidentifier():
                raise SourceAnalysisValidationError("function_name geçersiz.")
        if all_functions and function_name is not None:
            raise SourceAnalysisValidationError(
                "function_name ve all_functions aynı anda kullanılamaz."
            )
        if not all_functions and function_name is None:
            raise SourceAnalysisValidationError(
                "function_name veya all_functions seçilmelidir."
            )

    @staticmethod
    def _validate_optional_timeout(value: float | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SourceAnalysisValidationError(
                "per_function_timeout_seconds sayısal olmalıdır."
            )
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SourceAnalysisValidationError(
                "per_function_timeout_seconds pozitif ve sonlu olmalıdır."
            )
