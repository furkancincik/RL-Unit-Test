from __future__ import annotations

import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from analyzer.python_analyzer import FunctionInfo, PythonAnalyzer
from models.coverage_result import FunctionCoverageResult
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
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
    validate_qualified_target_name,
)
from models.project_coverage_result import ProjectTestCandidate
from services.project_analysis_report_service import ProjectAnalysisReportWriter
from services.project_deadline import ProjectDeadline
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
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._analyzer = analyzer or PythonAnalyzer()
        self._training_service_factory = (
            training_service_factory or RealRLTrainingService
        )
        self._report_writer = report_writer or ProjectAnalysisReportWriter()
        self._clock = clock

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
        relative_module_path: str | None = None,
        target_selection: TargetSelection | None = None,
        project_deadline: ProjectDeadline | None = None,
        allow_safe_object_setup: bool = True,
    ) -> ProjectAnalysisResult:
        started_at = self._clock()
        if project_deadline is not None and not isinstance(
            project_deadline, ProjectDeadline
        ):
            raise TypeError("project_deadline ProjectDeadline olmalıdır.")
        normalized_source = self._normalize_source_file(source_file)
        normalized_module = self._normalize_module_path(module_path)
        normalized_output = self._normalize_output_root(output_root)
        normalized_import_root = self._normalize_import_root(
            import_root,
            normalized_source,
        )
        normalized_relative_module_path = self._normalize_relative_module_path(
            relative_module_path,
            source_file=normalized_source,
            import_root=normalized_import_root,
        )
        normalized_selection = self._normalize_target_selection(
            module_identity=normalized_module,
            function_name=function_name,
            all_functions=all_functions,
            target_selection=target_selection,
        )
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
        if not isinstance(allow_safe_object_setup, bool):
            raise SourceAnalysisValidationError(
                "allow_safe_object_setup bool olmalıdır."
            )

        analysis = self._analyzer.analyze_file(normalized_source)
        discovered_targets = self._resolve_duplicate_targets(
            tuple(self._to_target(function) for function in analysis.functions)
        )
        selected_targets, selection_mode, selected_names = self._select_targets(
            discovered_targets,
            module_identity=normalized_module,
            target_selection=normalized_selection,
        )
        report_path = normalized_output / "project_analysis_report.json"
        function_results: list[FunctionAnalysisResult] = []
        coverage_candidates: list[ProjectTestCandidate] = []
        started_eligible_count = 0

        for ordinal, target in enumerate(selected_targets, start=1):
            function_output = self._function_output_directory(
                normalized_output, target.qualified_name, ordinal
            )
            if target.qualified_name not in selected_names:
                function_results.append(
                    FunctionAnalysisResult(
                        target=target,
                        status=FunctionRunStatus.SKIPPED_SELECTION,
                        diagnostic=None,
                        output_directory=function_output,
                        skip_reason="TARGET_NOT_SELECTED",
                    )
                )
                continue
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

            if project_deadline is not None and project_deadline.exceeded():
                function_results.append(
                    FunctionAnalysisResult(
                        target=target,
                        status=FunctionRunStatus.SKIPPED_DEADLINE,
                        diagnostic=None,
                        output_directory=function_output,
                        skip_reason="PROJECT_DEADLINE_EXCEEDED",
                    )
                )
                continue

            if (
                maximum_functions is not None
                and started_eligible_count >= maximum_functions
            ):
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

            started_eligible_count += 1
            function_timeout = per_function_timeout_seconds
            pytest_timeout = timeout_seconds
            comparison_timeout = comparison_timeout_seconds
            if project_deadline is not None:
                remaining = project_deadline.remaining_seconds()
                if remaining is not None:
                    pytest_timeout = min(float(timeout_seconds), remaining)
                    function_timeout = (
                        remaining
                        if per_function_timeout_seconds is None
                        else min(float(per_function_timeout_seconds), remaining)
                    )
                    comparison_timeout = (
                        remaining
                        if comparison_timeout_seconds is None
                        else min(float(comparison_timeout_seconds), remaining)
                    )
            service = self._training_service_factory()
            pipeline_result = service.run_with_diagnostics(
                source_file=normalized_source,
                module_path=normalized_module,
                function_name=target.qualified_name,
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
                timeout_seconds=pytest_timeout,
                pipeline_timeout_seconds=function_timeout,
                import_root=normalized_import_root,
                run_greedy_baseline=run_greedy_baseline,
                run_strategy_comparison=run_strategy_comparison,
                comparison_timeout_seconds=comparison_timeout,
                allow_safe_object_setup=allow_safe_object_setup,
            )
            diagnostic = self._extract_diagnostic(pipeline_result)
            if isinstance(pipeline_result, RealRLTrainingResult):
                coverage_candidates.extend(
                    self._project_test_candidates(
                        pipeline_result=pipeline_result,
                        target=target,
                        source_file=normalized_source,
                        import_root=(
                            normalized_import_root or Path.cwd().resolve()
                        ),
                        relative_module_path=normalized_relative_module_path,
                        initial_order=len(coverage_candidates),
                    )
                )
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
                    scenario_pool_coverage=(
                        pipeline_result.scenario_pool_coverage_result
                        if isinstance(pipeline_result, RealRLTrainingResult)
                        else None
                    ),
                    minimization_result=(
                        pipeline_result.minimization_result
                        if isinstance(pipeline_result, RealRLTrainingResult)
                        else None
                    ),
                    best_rl_coverage=(
                        pipeline_result.final_coverage_result
                        if isinstance(pipeline_result, RealRLTrainingResult)
                        and isinstance(
                            pipeline_result.final_coverage_result,
                            FunctionCoverageResult,
                        )
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
            total_duration_seconds=max(0.0, self._clock() - started_at),
            status=ProjectAnalysisResult.derive_status(statuses),
            output_root=normalized_output,
            report_path=report_path,
            coverage_candidates=tuple(coverage_candidates),
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
            class_name=function.class_name,
            constructor_parameters=tuple(function.constructor_parameters),
            constructor_parameter_types=tuple(
                sorted(function.constructor_parameter_types.items())
            ),
        )

    @staticmethod
    def _resolve_duplicate_targets(
        targets: tuple[FunctionTarget, ...],
    ) -> tuple[FunctionTarget, ...]:
        supported_name_counts = {
            target.qualified_name: sum(
                candidate.is_supported
                and candidate.qualified_name == target.qualified_name
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
            if (
                target.is_supported
                and supported_name_counts[target.qualified_name] > 1
            )
            else target
            for target in targets
        )

    @staticmethod
    def _select_targets(
        targets: tuple[FunctionTarget, ...],
        *,
        module_identity: str,
        target_selection: TargetSelection,
    ) -> tuple[
        tuple[FunctionTarget, ...],
        FunctionSelectionMode,
        frozenset[str],
    ]:
        if target_selection.mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            if not targets:
                raise SourceAnalysisValidationError(
                    "Kaynak dosyada fonksiyon bulunamadı."
                )
            return (
                targets,
                FunctionSelectionMode.ALL_ELIGIBLE_WITH_LIMIT,
                frozenset(target.qualified_name for target in targets),
            )

        requested = target_selection.for_module(module_identity)
        available = frozenset(target.qualified_name for target in targets)
        missing = tuple(name for name in requested if name not in available)
        if missing:
            raise SourceAnalysisValidationError(
                f"Fonksiyon bulunamadı: {missing[0]}."
            )
        selected = tuple(
            target
            for name in requested
            for target in targets
            if target.qualified_name == name
        )
        selected_ids = {id(target) for target in selected}
        ordered = (
            *selected,
            *(target for target in targets if id(target) not in selected_ids),
        )
        return (
            tuple(ordered),
            FunctionSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
            frozenset(requested),
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
        selected_names: frozenset[str],
    ) -> frozenset[int]:
        if maximum_functions is None:
            return frozenset()
        eligible_count = 0
        skipped: set[int] = set()
        for ordinal, target in enumerate(targets, start=1):
            if (
                target.qualified_name not in selected_names
                or not target.is_supported
            ):
                continue
            if eligible_count >= maximum_functions:
                skipped.add(ordinal)
                continue
            eligible_count += 1
        return frozenset(skipped)

    @staticmethod
    def _normalize_target_selection(
        *,
        module_identity: str,
        function_name: str | None,
        all_functions: bool,
        target_selection: TargetSelection | None,
    ) -> TargetSelection:
        if not isinstance(all_functions, bool):
            raise SourceAnalysisValidationError("all_functions bool olmalıdır.")
        if target_selection is not None:
            if not isinstance(target_selection, TargetSelection):
                raise SourceAnalysisValidationError(
                    "target_selection geçersiz."
                )
            if function_name is not None or all_functions:
                raise SourceAnalysisValidationError(
                    "Legacy function selection ve target_selection birlikte kullanılamaz."
                )
            return target_selection
        if function_name is not None:
            try:
                validate_qualified_target_name(function_name)
            except (TypeError, ValueError) as error:
                raise SourceAnalysisValidationError(
                    "function_name geçersiz."
                ) from error
        if all_functions and function_name is not None:
            raise SourceAnalysisValidationError(
                "function_name ve all_functions aynı anda kullanılamaz."
            )
        if not all_functions and function_name is None:
            raise SourceAnalysisValidationError(
                "function_name veya all_functions seçilmelidir."
            )
        if all_functions:
            return TargetSelection()
        assert function_name is not None
        return TargetSelection(
            TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
            (QualifiedTargetSelector(module_identity, function_name),),
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

    @staticmethod
    def _normalize_relative_module_path(
        value: str | None,
        *,
        source_file: Path,
        import_root: Path | None,
    ) -> str:
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise SourceAnalysisValidationError(
                    "relative_module_path boş olamaz."
                )
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") or ".." in Path(normalized).parts:
                raise SourceAnalysisValidationError(
                    "relative_module_path güvenli relative path olmalıdır."
                )
            return normalized
        if import_root is not None and source_file.is_relative_to(import_root):
            return source_file.relative_to(import_root).as_posix()
        return source_file.name

    @staticmethod
    def _project_test_candidates(
        *,
        pipeline_result: RealRLTrainingResult,
        target: FunctionTarget,
        source_file: Path,
        import_root: Path,
        relative_module_path: str,
        initial_order: int,
    ) -> tuple[ProjectTestCandidate, ...]:
        contributions = {
            item.scenario_id: item
            for item in (
                pipeline_result.minimization_result.contributions
                if pipeline_result.minimization_result is not None
                else ()
            )
        }
        candidates: list[ProjectTestCandidate] = []
        for offset, scenario in enumerate(pipeline_result.scenarios, start=1):
            contribution = contributions.get(scenario.scenario_id)
            may_reuse_target_only_contribution = scenario.setup_plan is None
            candidates.append(
                ProjectTestCandidate(
                    project_test_id=(
                        f"{relative_module_path}::{target.qualified_name}::"
                        f"{scenario.scenario_id}"
                    ),
                    relative_module_path=relative_module_path,
                    module_path=pipeline_result.module_path,
                    function_name=target.name,
                    function_start_line=target.start_line,
                    function_end_line=target.end_line,
                    source_file=source_file,
                    import_root=import_root,
                    scenario=scenario,
                    original_order=initial_order + offset,
                    precomputed_line_identities=(
                        contribution.covered_line_identities
                        if contribution is not None
                        and may_reuse_target_only_contribution
                        else None
                    ),
                    precomputed_branch_identities=(
                        contribution.covered_branch_identities
                        if contribution is not None
                        and may_reuse_target_only_contribution
                        else None
                    ),
                    precomputed_execution_success=(
                        contribution.execution_success
                        if contribution is not None
                        and may_reuse_target_only_contribution
                        else None
                    ),
                )
            )
        return tuple(candidates)
