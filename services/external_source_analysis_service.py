from __future__ import annotations

import ast
import json
import keyword
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath

from analyzer.python_source_reader import (
    PythonSourceEncodingError,
    decode_python_source_bytes,
)
from analyzer.python_analyzer import PythonAnalyzer
from models.external_source_analysis_result import (
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalModuleAnalysisResult,
    ExternalModuleSelectionMode,
    ExternalModuleStatus,
    ExternalSourceAnalysisRequest,
    ExternalSourceAnalysisResult,
    ExternalWorkspaceCleanupStatus,
    InlinePythonSource,
    LocalProjectDirectory,
    PublicGitHubRepository,
    UploadedPythonFile,
    validate_external_source_execution_policy,
)
from models.source_acquisition_result import (
    DiscoveredPythonModule,
    ResolvedSourceTarget,
    SourceAcquisitionLimits,
    SourceAcquisitionStatus,
    SourceAcquisitionRequest,
    SourceIssueCategory,
    SourceTargetKind,
)
from models.project_analysis_result import (
    FunctionRunStatus,
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)
from models.project_coverage_result import (
    ProjectCoverageResult,
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
    ProjectTestCandidate,
)
from services.project_coverage_service import ProjectCoverageService
from services.project_deadline import ProjectDeadline
from services.source_acquisition_service import SourceAcquisitionService
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator


_INLINE_WORKSPACE_PREFIX = "rl-unit-test-inline-"
_UPLOAD_WORKSPACE_PREFIX = "rl-unit-test-upload-"


def portable_upload_module_identity(source: UploadedPythonFile) -> str:
    """Upload için filesystem taşımayan canonical public module identity üretir."""
    stem = Path(source.original_filename).stem
    normalized = re.sub(r"\W", "_", stem, flags=re.UNICODE)
    if not normalized:
        normalized = "source"
    if not normalized.startswith("upload_"):
        normalized = f"upload_{normalized}"
    if not normalized.isidentifier() or keyword.iskeyword(normalized):
        raise ExternalSourceAnalysisValidationError(
            "Upload filename güvenli Python module adına dönüştürülemedi."
        )
    return normalized


class ExternalSourceAnalysisValidationError(ValueError):
    """External analysis isteği production validation politikasını ihlal etti."""


class ExternalSourceAnalysisService:
    """Harici Python kaynaklarını güvenli discovery veya açıkça trusted analiz eder."""

    def __init__(
        self,
        *,
        acquisition_service: SourceAcquisitionService | None = None,
        orchestrator_factory: Callable[[], SourceAnalysisOrchestrator] | None = None,
        project_coverage_service: ProjectCoverageService | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._acquisition_service = acquisition_service or SourceAcquisitionService()
        self._orchestrator_factory = orchestrator_factory or (
            lambda: SourceAnalysisOrchestrator(clock=self._clock)
        )
        self._project_coverage_service = (
            project_coverage_service or ProjectCoverageService()
        )
        self._clock = clock

    def run(self, request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        if not isinstance(request, ExternalSourceAnalysisRequest):
            raise TypeError("request ExternalSourceAnalysisRequest olmalıdır.")
        validate_external_source_execution_policy(
            request.source,
            request.execution_policy,
            request.configuration,
            request.pinned_github_authorization,
        )
        started = self._clock()
        deadline = ProjectDeadline(
            timeout_seconds=request.configuration.project_timeout_seconds,
            started_at=started,
            clock=self._clock,
        )
        last_completed_stage: str | None = None
        output_root = request.configuration.output_root.resolve()
        report_path = output_root / "external_source_analysis_report.json"
        if (
            request.execution_policy is ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS
            and request.configuration.per_function_pipeline_timeout_seconds is None
        ):
            raise ExternalSourceAnalysisValidationError(
                "Trusted external dynamic analysis per-function pipeline timeout gerektirir."
            )
        temporary_workspace: Path | None = None
        acquired: ResolvedSourceTarget | None = None
        cleanup_attempted = False
        cleanup_status = ExternalWorkspaceCleanupStatus.NOT_REQUIRED

        def cleanup_once(
            current_acquired: ResolvedSourceTarget | None,
            current_workspace: Path | None,
        ) -> ExternalWorkspaceCleanupStatus:
            nonlocal cleanup_attempted, cleanup_status
            if cleanup_attempted:
                return cleanup_status
            cleanup_attempted = True
            cleanup_status = self._cleanup(current_acquired, current_workspace)
            return cleanup_status

        def finalize(
            result: ExternalSourceAnalysisResult,
            *,
            acquired: ResolvedSourceTarget | None,
            temporary_workspace: Path | None,
            deadline: ProjectDeadline | None = None,
        ) -> ExternalSourceAnalysisResult:
            return self._finalize(
                result,
                acquired=acquired,
                temporary_workspace=temporary_workspace,
                deadline=deadline,
                cleanup=cleanup_once,
            )

        try:
            validation_issue = self._validate_payload(request)
            if validation_issue is not None:
                return finalize(
                    self._failed_result(
                        request, output_root, report_path, started, validation_issue
                    ),
                    acquired=None,
                    temporary_workspace=None,
                    deadline=deadline,
                )
            last_completed_stage = "REQUEST_VALIDATION"
            if deadline.exceeded():
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="SOURCE_PREPARATION",
                    ),
                    acquired=None,
                    temporary_workspace=None,
                    deadline=deadline,
                )

            source_request: SourceAcquisitionRequest
            source = request.source
            if isinstance(source, InlinePythonSource):
                temporary_workspace = Path(
                    tempfile.mkdtemp(prefix=_INLINE_WORKSPACE_PREFIX)
                ).resolve()
                source_file = temporary_workspace / "inline_source.py"
                source_file.write_text(source.source_text, encoding="utf-8")
                source_request = SourceAcquisitionRequest(
                    source_kind=SourceTargetKind.LOCAL_FILE,
                    origin=str(source_file),
                    limits=request.acquisition_limits,
                )
            elif isinstance(source, UploadedPythonFile):
                temporary_workspace = Path(
                    tempfile.mkdtemp(prefix=_UPLOAD_WORKSPACE_PREFIX)
                ).resolve()
                source_file = temporary_workspace / (
                    f"{self._portable_module_name(source)}.py"
                )
                source_file.write_bytes(source.file_bytes)
                source_request = SourceAcquisitionRequest(
                    source_kind=SourceTargetKind.LOCAL_FILE,
                    origin=str(source_file),
                    limits=request.acquisition_limits,
                )
            elif isinstance(source, LocalProjectDirectory):
                source_request = SourceAcquisitionRequest(
                    source_kind=SourceTargetKind.LOCAL_DIRECTORY,
                    origin=str(source.directory),
                    include_tests=source.include_tests,
                    limits=request.acquisition_limits,
                )
            else:
                assert isinstance(source, PublicGitHubRepository)
                source_request = SourceAcquisitionRequest(
                    source_kind=SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
                    origin=source.repository_url,
                    ref=source.ref,
                    include_tests=source.include_tests,
                    limits=request.acquisition_limits,
                )

            last_completed_stage = "SOURCE_PREPARATION"
            if deadline.exceeded():
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="SOURCE_ACQUISITION",
                    ),
                    acquired=None,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )
            source_request = replace(
                source_request,
                limits=self._acquisition_limits_for_deadline(
                    source_request,
                    deadline,
                ),
            )
            if deadline.exceeded():
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="SOURCE_ACQUISITION",
                    ),
                    acquired=None,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )
            acquired = self._acquisition_service.resolve(
                source_request,
                project_deadline=deadline,
            )
            last_completed_stage = "SOURCE_ACQUISITION"
            if not acquired.is_available:
                if deadline.exceeded():
                    return finalize(
                        self._deadline_result(
                            request,
                            output_root,
                            report_path,
                            deadline,
                            acquired=acquired,
                            last_completed_stage=last_completed_stage,
                            deadline_stage="SOURCE_ACQUISITION",
                        ),
                        acquired=acquired,
                        temporary_workspace=temporary_workspace,
                        deadline=deadline,
                    )
                categories = tuple(issue.category.value for issue in acquired.issues)
                issue = categories[0] if categories else "SOURCE_ACQUISITION_FAILED"
                return finalize(
                    self._failed_result(
                        request,
                        output_root,
                        report_path,
                        started,
                        issue,
                        acquired=acquired,
                    ),
                    acquired=acquired,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )

            self._validate_output_separation(output_root, acquired.resolved_project_root)

            if deadline.exceeded():
                inventories = self._discovery_inventories(acquired)
                module_results = self._analyze_modules(
                    request,
                    acquired,
                    output_root,
                    inventories,
                    deadline,
                )
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        acquired=acquired,
                        module_results=module_results,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="TARGET_DISCOVERY",
                    ),
                    acquired=acquired,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )

            target_inventories = self._target_inventories(
                request,
                acquired,
                deadline=deadline,
            )
            if deadline.exceeded():
                module_results = self._analyze_modules(
                    request,
                    acquired,
                    output_root,
                    target_inventories,
                    deadline,
                )
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        acquired=acquired,
                        module_results=module_results,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="TARGET_DISCOVERY",
                    ),
                    acquired=acquired,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )
            last_completed_stage = "TARGET_DISCOVERY"
            selection_issue = self._target_selection_issue(
                request,
                acquired,
                target_inventories,
            )
            if selection_issue is not None:
                return finalize(
                    self._failed_result(
                        request,
                        output_root,
                        report_path,
                        started,
                        selection_issue,
                        acquired=acquired,
                    ),
                    acquired=acquired,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )

            module_results = self._analyze_modules(
                request,
                acquired,
                output_root,
                target_inventories,
                deadline,
            )
            last_completed_stage = "FUNCTION_ANALYSIS"
            if deadline.exceeded():
                return finalize(
                    self._deadline_result(
                        request,
                        output_root,
                        report_path,
                        deadline,
                        acquired=acquired,
                        module_results=module_results,
                        last_completed_stage=last_completed_stage,
                        deadline_stage="PROJECT_COVERAGE",
                    ),
                    acquired=acquired,
                    temporary_workspace=temporary_workspace,
                    deadline=deadline,
                )
            project_coverage = self._measure_project_coverage(
                request=request,
                acquired=acquired,
                module_results=module_results,
                output_root=output_root,
                target_inventories=target_inventories,
                deadline=deadline,
            )
            last_completed_stage = "PROJECT_COVERAGE"
            status = self._derive_status(
                request.execution_policy,
                module_results,
                acquired,
            )
            if (
                status is ExternalAnalysisStatus.COMPLETED
                and project_coverage is not None
                and project_coverage.full_scenario_count > 0
                and project_coverage.status is not ProjectCoverageStatus.COMPLETED
            ):
                status = ExternalAnalysisStatus.PARTIAL
            result = ExternalSourceAnalysisResult(
                source_kind=request.source.source_kind,
                execution_policy=request.execution_policy,
                status=status,
                acquisition_status=acquired.status.value,
                repository_name=self._public_source_name(request, acquired),
                github_owner=acquired.github_owner,
                github_repository=acquired.github_repository,
                resolved_commit_sha=acquired.resolved_commit_sha,
                discovered_module_count=len(acquired.discovered_modules),
                selected_module_count=sum(
                    item.status is not ExternalModuleStatus.SKIPPED_LIMIT
                    for item in module_results
                ),
                module_results=module_results,
                output_root=output_root,
                report_path=report_path,
                duration_seconds=max(0.0, self._clock() - started),
                cleanup_status=ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
                issues=tuple(dict.fromkeys(issue.category.value for issue in acquired.issues)),
                project_coverage=project_coverage,
                project_timeout_seconds=deadline.timeout_seconds,
                project_deadline_exceeded=deadline.exceeded(),
                last_completed_stage=last_completed_stage,
                deadline_stage=(
                    "PROJECT_COVERAGE" if deadline.exceeded() else None
                ),
            )
            return finalize(
                result,
                acquired=acquired,
                temporary_workspace=temporary_workspace,
                deadline=deadline,
            )
        finally:
            cleanup_once(acquired, temporary_workspace)

    def _analyze_modules(
        self,
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
        output_root: Path,
        target_inventories: dict[str, tuple[str, ...]],
        deadline: ProjectDeadline,
    ) -> tuple[ExternalModuleAnalysisResult, ...]:
        ordered = tuple(
            sorted(acquired.discovered_modules, key=lambda item: item.relative_path.casefold())
        )
        selected = self._selected_paths(request, ordered)
        results: list[ExternalModuleAnalysisResult] = []
        for ordinal, module in enumerate(ordered, start=1):
            module_issue = self._module_issue(acquired, module)
            discovered_names = target_inventories.get(
                module.relative_path,
                module.top_level_function_names,
            )
            if module.relative_path not in selected:
                if request.configuration.module_selection.mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
                    results.append(
                        self._module_result(
                            module,
                            ExternalModuleStatus.SKIPPED_LIMIT,
                            discovered_function_count=len(discovered_names),
                            discovered_function_names=discovered_names,
                        )
                    )
                continue
            if not module.supported or module.module_path is None:
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.UNSUPPORTED,
                        issue_category=module_issue,
                        issue_message="Module static discovery sonrasında güvenle analiz edilemiyor.",
                        discovered_function_count=len(discovered_names),
                        discovered_function_names=discovered_names,
                    )
                )
                continue
            if deadline.exceeded():
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.SKIPPED_DEADLINE,
                        issue_category="PROJECT_DEADLINE_EXCEEDED",
                        issue_message="Module project deadline nedeniyle başlatılmadı.",
                        discovered_function_count=len(discovered_names),
                        discovered_function_names=discovered_names,
                    )
                )
                continue
            if request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.STATIC_ONLY,
                        discovered_function_count=len(discovered_names),
                        discovered_function_names=discovered_names,
                    )
                )
                continue

            import_root = self._import_root(acquired.resolved_project_root, module)
            source_file = (acquired.resolved_project_root / module.relative_path).resolve()
            module_output = (output_root / f"module_{ordinal:03d}").resolve()
            if not module_output.is_relative_to(output_root):
                raise RuntimeError("Module output yolu root dışına çıktı.")
            try:
                target_selection = self._internal_target_selection(
                    request,
                    module,
                )
                project_result = self._orchestrator_factory().run(
                    source_file=source_file,
                    module_path=module.module_path,
                    output_root=module_output,
                    function_name=None,
                    all_functions=(
                        target_selection.mode
                        is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
                    ),
                    target_selection=(
                        None
                        if target_selection.mode
                        is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
                        else target_selection
                    ),
                    maximum_functions=request.configuration.maximum_functions_per_module,
                    import_root=import_root,
                    max_visits_per_node=request.configuration.max_visits_per_node,
                    episode_count=request.configuration.episode_count,
                    epsilon=request.configuration.epsilon,
                    learning_rate=request.configuration.learning_rate,
                    discount_factor=request.configuration.discount_factor,
                    random_seed=request.configuration.random_seed,
                    timeout_seconds=request.configuration.pytest_coverage_timeout_seconds,
                    per_function_timeout_seconds=(
                        request.configuration.per_function_pipeline_timeout_seconds
                    ),
                    run_greedy_baseline=request.configuration.run_greedy_baseline,
                    run_strategy_comparison=request.configuration.run_strategy_comparison,
                    comparison_timeout_seconds=request.configuration.comparison_timeout_seconds,
                    relative_module_path=module.relative_path,
                    project_deadline=deadline,
                )
            except ModuleNotFoundError:
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.FAILED,
                        issue_category="DEPENDENCY_UNAVAILABLE",
                        issue_message="Module dependency import edilemedi.",
                    )
                )
                continue
            except ImportError:
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.FAILED,
                        issue_category="IMPORT_FAILED",
                        issue_message="Module güvenli import doğrulamasını geçemedi.",
                    )
                )
                continue
            status = ExternalModuleStatus(project_result.status.value)
            self._sanitize_json_artifacts(
                module_output=module_output,
                source_root=acquired.resolved_project_root,
                output_root=output_root,
                internal_module_name=module.module_path,
                public_module_name=self._public_module_name(request, module),
            )
            artifact_candidates = [project_result.report_path]
            for function_result in project_result.function_results:
                artifact_candidates.extend(function_result.artifact_paths)
            artifacts = tuple(
                dict.fromkeys(
                    Path(path).resolve()
                    for path in artifact_candidates
                    if path is not None and Path(path).is_file()
                )
            )
            results.append(
                self._module_result(
                    module,
                    status,
                    project_result=project_result,
                    artifact_paths=artifacts,
                    discovered_function_count=len(project_result.discovered_targets),
                    discovered_function_names=tuple(
                        target.qualified_name
                        for target in project_result.discovered_targets
                    ),
                )
            )
        if isinstance(request.source, InlinePythonSource):
            results = [
                replace(
                    item,
                    relative_path="inline_source.py",
                    module_name="inline_source",
                )
                for item in results
            ]
        elif isinstance(request.source, UploadedPythonFile):
            public_name = Path(request.source.original_filename).name
            portable_module_name = self._portable_module_name(request.source)
            results = [
                replace(
                    item,
                    relative_path=public_name,
                    module_name=portable_module_name,
                )
                for item in results
            ]
        return tuple(results)

    def _measure_project_coverage(
        self,
        *,
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
        module_results: tuple[ExternalModuleAnalysisResult, ...],
        output_root: Path,
        target_inventories: dict[str, tuple[str, ...]],
        deadline: ProjectDeadline,
    ) -> ProjectCoverageResult | None:
        if request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
            return None
        remaining = deadline.remaining_seconds()
        if remaining is not None and remaining <= 0.0:
            return None
        candidates: list[ProjectTestCandidate] = []
        for module in module_results:
            project = module.project_result
            project_candidates = getattr(project, "coverage_candidates", ())
            if not isinstance(project_candidates, tuple):
                project_candidates = ()
            for candidate in project_candidates:
                order = len(candidates) + 1
                public_relative_path = module.relative_path.replace("\\", "/")
                candidates.append(
                    replace(
                        candidate,
                        project_test_id=(
                            f"{public_relative_path}::"
                            f"{candidate.qualified_function_name}::"
                            f"{candidate.scenario.scenario_id}"
                        ),
                        relative_module_path=public_relative_path,
                        original_order=order,
                    )
                )
        scope = self._project_scope_summary(
            acquired=acquired,
            module_results=module_results,
            target_inventories=target_inventories,
        )
        coverage_timeout = request.configuration.pytest_coverage_timeout_seconds
        if remaining is not None:
            coverage_timeout = min(float(coverage_timeout), remaining)
        return self._project_coverage_service.measure_and_minimize(
            candidates=tuple(candidates),
            scope=scope,
            output_root=output_root,
            timeout_seconds=coverage_timeout,
        )

    @staticmethod
    def _project_scope_summary(
        *,
        acquired: ResolvedSourceTarget,
        module_results: tuple[ExternalModuleAnalysisResult, ...],
        target_inventories: dict[str, tuple[str, ...]],
    ) -> ProjectCoverageScopeSummary:
        function_results = tuple(
            function
            for module in module_results
            for function in (
                getattr(module.project_result, "function_results", ())
                if module.project_result is not None
                else ()
            )
        )
        discovered_functions = sum(
            len(target_inventories.get(module.relative_path, ()))
            for module in acquired.discovered_modules
        )
        unsupported_from_projects = sum(
            function.status is FunctionRunStatus.UNSUPPORTED
            for function in function_results
        )
        unsupported_without_project = sum(
            len(target_inventories.get(module.relative_path, ()))
            for module in acquired.discovered_modules
            if not module.supported
        )
        unsupported = unsupported_from_projects + unsupported_without_project
        analyzed = sum(
            function.status
            not in {
                FunctionRunStatus.SKIPPED,
                FunctionRunStatus.SKIPPED_SELECTION,
                FunctionRunStatus.SKIPPED_LIMIT,
                FunctionRunStatus.SKIPPED_DEADLINE,
                FunctionRunStatus.UNSUPPORTED,
            }
            for function in function_results
        )
        skipped_limit = sum(
            function.status is FunctionRunStatus.SKIPPED_LIMIT
            for function in function_results
        ) + sum(
            module.discovered_function_count
            for module in module_results
            if module.status is ExternalModuleStatus.SKIPPED_LIMIT
            and module.project_result is None
        )
        skipped_selection = sum(
            function.status is FunctionRunStatus.SKIPPED_SELECTION
            for function in function_results
        )
        skipped_deadline = sum(
            function.status is FunctionRunStatus.SKIPPED_DEADLINE
            for function in function_results
        ) + sum(
            module.discovered_function_count
            for module in module_results
            if module.status is ExternalModuleStatus.SKIPPED_DEADLINE
            and module.project_result is None
        )
        selected_modules = sum(
            module.status is not ExternalModuleStatus.SKIPPED_LIMIT
            for module in module_results
        )
        completed_modules = sum(
            module.status is ExternalModuleStatus.COMPLETED
            for module in module_results
        )
        completed = sum(
            function.status is FunctionRunStatus.COMPLETED
            for function in function_results
        )
        partial = sum(
            function.status is FunctionRunStatus.PARTIAL
            for function in function_results
        )
        failed = sum(
            function.status is FunctionRunStatus.FAILED
            for function in function_results
        ) + sum(
            module.discovered_function_count
            for module in module_results
            if module.status is ExternalModuleStatus.FAILED
            and module.project_result is None
        )
        timed_out = sum(
            function.status is FunctionRunStatus.TIMED_OUT
            for function in function_results
        ) + sum(
            module.discovered_function_count
            for module in module_results
            if module.status is ExternalModuleStatus.TIMED_OUT
            and module.project_result is None
        )
        scope_complete = (
            bool(module_results)
            and all(
                module.discovered_function_count == 0
                or module.status is ExternalModuleStatus.COMPLETED
                for module in module_results
            )
            and unsupported == 0
            and skipped_limit == 0
            and skipped_selection == 0
            and skipped_deadline == 0
            and partial == 0
            and failed == 0
            and timed_out == 0
            and completed == discovered_functions
        )
        return ProjectCoverageScopeSummary(
            discovered_module_count=len(acquired.discovered_modules),
            selected_module_count=selected_modules,
            completed_module_count=completed_modules,
            discovered_function_count=discovered_functions,
            eligible_function_count=max(0, discovered_functions - unsupported),
            analyzed_function_count=analyzed,
            completed_function_count=completed,
            partial_function_count=partial,
            failed_function_count=failed,
            timed_out_function_count=timed_out,
            unsupported_function_count=unsupported,
            skipped_limit_function_count=skipped_limit,
            scope_complete=scope_complete,
            skipped_selection_function_count=skipped_selection,
            skipped_deadline_function_count=skipped_deadline,
        )

    @staticmethod
    def _target_inventories(
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
        *,
        deadline: ProjectDeadline | None = None,
    ) -> dict[str, tuple[str, ...]]:
        inventories: dict[str, tuple[str, ...]] = {}
        analyzer = PythonAnalyzer()
        for module in acquired.discovered_modules:
            if deadline is not None and deadline.exceeded():
                inventories[module.relative_path] = module.top_level_function_names
                continue
            if not module.supported or module.module_path is None:
                inventories[module.relative_path] = module.top_level_function_names
                continue
            source_file = (
                acquired.resolved_project_root / module.relative_path
            ).resolve()
            analysis = analyzer.analyze_file(source_file)
            inventories[module.relative_path] = tuple(
                target.qualified_name for target in analysis.functions
            )
        return inventories

    @staticmethod
    def _target_selection_issue(
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
        inventories: dict[str, tuple[str, ...]],
    ) -> str | None:
        selection = request.configuration.target_selection
        if selection.mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            return None
        available = {
            (
                ExternalSourceAnalysisService._public_module_name(request, module),
                qualified_name,
            )
            for module in acquired.discovered_modules
            for qualified_name in inventories.get(module.relative_path, ())
        }
        if any(
            (selector.module_identity, selector.qualified_name) not in available
            for selector in selection.selectors
        ):
            return "UNKNOWN_TARGET_SELECTION"
        return None

    @staticmethod
    def _internal_target_selection(
        request: ExternalSourceAnalysisRequest,
        module: DiscoveredPythonModule,
    ) -> TargetSelection:
        selection = request.configuration.target_selection
        if selection.mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            return selection
        public_module = ExternalSourceAnalysisService._public_module_name(
            request,
            module,
        )
        internal_module = module.module_path
        if internal_module is None:
            raise RuntimeError("Dynamic target selection module path gerektirir.")
        return TargetSelection(
            TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
            tuple(
                QualifiedTargetSelector(
                    (
                        internal_module
                        if selector.module_identity == public_module
                        else selector.module_identity
                    ),
                    selector.qualified_name,
                )
                for selector in selection.selectors
            ),
        )

    @staticmethod
    def _selected_paths(
        request: ExternalSourceAnalysisRequest,
        modules: tuple[DiscoveredPythonModule, ...],
    ) -> frozenset[str]:
        selection = request.configuration.module_selection
        if selection.mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            eligible = tuple(item for item in modules if item.supported)
            selected_eligible = eligible[: request.configuration.maximum_selected_modules]
            return frozenset(
                item.relative_path
                for item in (*selected_eligible, *(item for item in modules if not item.supported))
            )
        if selection.mode is ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS:
            normalized: set[str] = set()
            for value in selection.values:
                path = PurePosixPath(value.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".py":
                    raise ExternalSourceAnalysisValidationError(
                        "Explicit relative module path geçersiz."
                    )
                normalized.add(path.as_posix())
            return frozenset(normalized)
        requested = frozenset(selection.values)
        if any(any(not part.isidentifier() for part in value.split(".")) for value in requested):
            raise ExternalSourceAnalysisValidationError(
                "Explicit module name geçersiz."
            )
        return frozenset(
            item.relative_path for item in modules if item.module_path in requested
        )

    @staticmethod
    def _module_issue(
        acquired: ResolvedSourceTarget,
        module: DiscoveredPythonModule,
    ) -> str | None:
        return next(
            (
                issue.category.value
                for issue in acquired.issues
                if issue.relative_path == module.relative_path
            ),
            None,
        )

    @staticmethod
    def _module_result(
        module: DiscoveredPythonModule,
        status: ExternalModuleStatus,
        *,
        project_result: object | None = None,
        issue_category: str | None = None,
        issue_message: str | None = None,
        artifact_paths: tuple[Path, ...] = (),
        discovered_function_count: int | None = None,
        discovered_function_names: tuple[str, ...] | None = None,
    ) -> ExternalModuleAnalysisResult:
        return ExternalModuleAnalysisResult(
            relative_path=module.relative_path,
            module_name=module.module_path,
            status=status,
            discovered_function_count=(
                module.top_level_function_count
                if discovered_function_count is None
                else discovered_function_count
            ),
            project_result=project_result,
            issue_category=issue_category,
            issue_message=issue_message,
            artifact_paths=artifact_paths,
            discovered_function_names=(
                module.top_level_function_names
                if discovered_function_names is None
                else discovered_function_names
            ),
        )

    @staticmethod
    def _import_root(root: Path, module: DiscoveredPythonModule) -> Path:
        if module.package_root in (None, "."):
            candidate = root.resolve()
        else:
            candidate = (root / module.package_root).resolve()
        if not candidate.is_dir() or not candidate.is_relative_to(root.resolve()):
            raise RuntimeError("Module import root project root dışında.")
        return candidate

    @staticmethod
    def _derive_status(
        policy: ExternalExecutionPolicy,
        modules: tuple[ExternalModuleAnalysisResult, ...],
        acquired: ResolvedSourceTarget,
    ) -> ExternalAnalysisStatus:
        considered = tuple(
            item.status
            for item in modules
            if item.status is not ExternalModuleStatus.SKIPPED_LIMIT
        )
        if policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
            if (
                not considered
                and acquired.source_kind
                is SourceTargetKind.PUBLIC_GITHUB_REPOSITORY
                and acquired.status is SourceAcquisitionStatus.PARTIAL
                and not acquired.discovered_modules
                and any(
                    issue.category is SourceIssueCategory.NO_PYTHON_FILES
                    for issue in acquired.issues
                )
            ):
                return ExternalAnalysisStatus.PARTIAL
            return (
                ExternalAnalysisStatus.STATIC_COMPLETED
                if any(status is ExternalModuleStatus.STATIC_ONLY for status in considered)
                else ExternalAnalysisStatus.FAILED
            )
        if not considered:
            return ExternalAnalysisStatus.FAILED
        if all(status is ExternalModuleStatus.COMPLETED for status in considered):
            return ExternalAnalysisStatus.COMPLETED
        if all(status is ExternalModuleStatus.TIMED_OUT for status in considered):
            return ExternalAnalysisStatus.TIMED_OUT
        if any(
            status in {ExternalModuleStatus.COMPLETED, ExternalModuleStatus.PARTIAL}
            for status in considered
        ):
            return ExternalAnalysisStatus.PARTIAL
        return ExternalAnalysisStatus.FAILED

    @staticmethod
    def _validate_payload(request: ExternalSourceAnalysisRequest) -> str | None:
        source = request.source
        limit = request.configuration.maximum_payload_bytes
        if isinstance(source, InlinePythonSource):
            if source.transport_size_bytes > limit:
                return "SOURCE_LIMIT_EXCEEDED"
            if source.normalization_issue is not None:
                return source.normalization_issue
            if ExternalSourceAnalysisService._has_invalid_control(
                source.source_text
            ):
                return "INVALID_CONTROL_CHARACTER"
            try:
                ast.parse(source.source_text)
            except SyntaxError:
                return "SYNTAX_ERROR"
        elif isinstance(source, UploadedPythonFile):
            filename = source.original_filename
            posix = PurePosixPath(filename)
            windows = PureWindowsPath(filename)
            if (
                Path(filename).name != filename
                or posix.name != filename
                or windows.name != filename
                or posix.is_absolute()
                or windows.is_absolute()
                or filename in {".", ".."}
                or Path(filename).suffix.lower() != ".py"
            ):
                return "INVALID_UPLOAD_FILENAME"
            if len(source.file_bytes) > limit:
                return "SOURCE_LIMIT_EXCEEDED"
            try:
                decoded = decode_python_source_bytes(source.file_bytes).text
            except PythonSourceEncodingError:
                return "UNSUPPORTED_ENCODING"
            if ExternalSourceAnalysisService._has_invalid_control(decoded):
                return "INVALID_CONTROL_CHARACTER"
            try:
                ast.parse(decoded)
            except SyntaxError:
                return "SYNTAX_ERROR"
        return None

    @staticmethod
    def _has_invalid_control(value: str) -> bool:
        return any(ord(character) < 32 and character not in "\n\r\t\f" for character in value)

    @staticmethod
    def _validate_output_separation(output_root: Path, project_root: Path) -> None:
        root = project_root.resolve()
        output = output_root.resolve()
        if root.exists() and output.is_relative_to(root):
            raise ExternalSourceAnalysisValidationError(
                "output root source project içinde olamaz."
            )

    @staticmethod
    def _acquisition_limits_for_deadline(
        source_request: SourceAcquisitionRequest,
        deadline: ProjectDeadline,
    ) -> SourceAcquisitionLimits:
        limits = source_request.limits
        if source_request.source_kind is not SourceTargetKind.PUBLIC_GITHUB_REPOSITORY:
            return limits
        remaining = deadline.remaining_seconds()
        if remaining is None or remaining <= 0.0:
            return limits
        return replace(
            limits,
            clone_timeout_seconds=min(
                float(limits.clone_timeout_seconds),
                remaining,
            ),
        )

    @staticmethod
    def _discovery_inventories(
        acquired: ResolvedSourceTarget,
    ) -> dict[str, tuple[str, ...]]:
        return {
            module.relative_path: module.top_level_function_names
            for module in acquired.discovered_modules
        }

    def _deadline_result(
        self,
        request: ExternalSourceAnalysisRequest,
        output_root: Path,
        report_path: Path,
        deadline: ProjectDeadline,
        *,
        acquired: ResolvedSourceTarget | None = None,
        module_results: tuple[ExternalModuleAnalysisResult, ...] = (),
        last_completed_stage: str | None,
        deadline_stage: str,
    ) -> ExternalSourceAnalysisResult:
        usable = any(
            item.status
            in {
                ExternalModuleStatus.STATIC_ONLY,
                ExternalModuleStatus.COMPLETED,
                ExternalModuleStatus.PARTIAL,
            }
            for item in module_results
        )
        acquisition_issues = (
            tuple(issue.category.value for issue in acquired.issues)
            if acquired is not None
            else ()
        )
        return ExternalSourceAnalysisResult(
            source_kind=request.source.source_kind,
            execution_policy=request.execution_policy,
            status=(
                ExternalAnalysisStatus.PARTIAL
                if usable
                else ExternalAnalysisStatus.TIMED_OUT
            ),
            acquisition_status=(
                acquired.status.value if acquired is not None else "NOT_STARTED"
            ),
            repository_name=(
                self._public_source_name(request, acquired)
                if acquired is not None
                else None
            ),
            github_owner=acquired.github_owner if acquired is not None else None,
            github_repository=(
                acquired.github_repository if acquired is not None else None
            ),
            resolved_commit_sha=(
                acquired.resolved_commit_sha if acquired is not None else None
            ),
            discovered_module_count=(
                len(acquired.discovered_modules) if acquired is not None else 0
            ),
            selected_module_count=sum(
                item.status is not ExternalModuleStatus.SKIPPED_LIMIT
                for item in module_results
            ),
            module_results=module_results,
            output_root=output_root,
            report_path=report_path,
            duration_seconds=deadline.elapsed_seconds(),
            cleanup_status=ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
            issues=tuple(
                dict.fromkeys(
                    (*acquisition_issues, "PROJECT_DEADLINE_EXCEEDED")
                )
            ),
            project_coverage=None,
            project_timeout_seconds=deadline.timeout_seconds,
            project_deadline_exceeded=True,
            last_completed_stage=last_completed_stage,
            deadline_stage=deadline_stage,
        )

    def _failed_result(
        self,
        request: ExternalSourceAnalysisRequest,
        output_root: Path,
        report_path: Path,
        started: float,
        issue: str,
        *,
        acquired: ResolvedSourceTarget | None = None,
        cleanup_status: ExternalWorkspaceCleanupStatus = ExternalWorkspaceCleanupStatus.NOT_REQUIRED,
    ) -> ExternalSourceAnalysisResult:
        return ExternalSourceAnalysisResult(
            source_kind=request.source.source_kind,
            execution_policy=request.execution_policy,
            status=ExternalAnalysisStatus.FAILED,
            acquisition_status=(acquired.status.value if acquired else "FAILED"),
            repository_name=(
                self._public_source_name(request, acquired) if acquired else None
            ),
            github_owner=acquired.github_owner if acquired else None,
            github_repository=acquired.github_repository if acquired else None,
            resolved_commit_sha=acquired.resolved_commit_sha if acquired else None,
            discovered_module_count=(len(acquired.discovered_modules) if acquired else 0),
            selected_module_count=0,
            module_results=(),
            output_root=output_root,
            report_path=report_path,
            duration_seconds=max(0.0, self._clock() - started),
            cleanup_status=cleanup_status,
            issues=(issue,),
            project_timeout_seconds=request.configuration.project_timeout_seconds,
        )

    def _finalize(
        self,
        result: ExternalSourceAnalysisResult,
        *,
        acquired: ResolvedSourceTarget | None,
        temporary_workspace: Path | None,
        deadline: ProjectDeadline | None = None,
        cleanup: Callable[
            [ResolvedSourceTarget | None, Path | None],
            ExternalWorkspaceCleanupStatus,
        ] | None = None,
    ) -> ExternalSourceAnalysisResult:
        # Rapor önce persistent output'a atomik yazılır; cleanup sonucu aynı
        # dosyaya ikinci bir atomik metadata güncellemesiyle yansıtılır.
        self._write_result(result)
        cleanup_status = (cleanup or self._cleanup)(acquired, temporary_workspace)
        issues = result.issues
        if cleanup_status is ExternalWorkspaceCleanupStatus.FAILED:
            issues = tuple(dict.fromkeys((*issues, "CLEANUP_FAILED")))
        deadline_exceeded = result.project_deadline_exceeded or (
            deadline is not None and deadline.exceeded()
        )
        status = result.status
        deadline_stage = result.deadline_stage
        if deadline_exceeded and not result.project_deadline_exceeded:
            deadline_stage = "REPORT_FINALIZATION_OR_CLEANUP"
            if status in {
                ExternalAnalysisStatus.COMPLETED,
                ExternalAnalysisStatus.STATIC_COMPLETED,
            }:
                status = (
                    ExternalAnalysisStatus.PARTIAL
                    if result.module_results
                    else ExternalAnalysisStatus.TIMED_OUT
                )
        if deadline_exceeded:
            issues = tuple(dict.fromkeys((*issues, "PROJECT_DEADLINE_EXCEEDED")))
        finalized = replace(
            result,
            status=(
                ExternalAnalysisStatus.FAILED
                if cleanup_status is ExternalWorkspaceCleanupStatus.FAILED
                else status
            ),
            cleanup_status=cleanup_status,
            issues=issues,
            duration_seconds=(
                deadline.elapsed_seconds() if deadline is not None else result.duration_seconds
            ),
            project_deadline_exceeded=deadline_exceeded,
            deadline_stage=deadline_stage,
        )
        return self._write_result(finalized)

    @staticmethod
    def _public_source_name(
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
    ) -> str | None:
        source = request.source
        if isinstance(source, InlinePythonSource):
            return "inline_source"
        if isinstance(source, UploadedPythonFile):
            return Path(source.original_filename).stem
        return acquired.repository_name

    @staticmethod
    def _public_module_name(
        request: ExternalSourceAnalysisRequest,
        module: DiscoveredPythonModule,
    ) -> str:
        if isinstance(request.source, InlinePythonSource):
            return "inline_source"
        if isinstance(request.source, UploadedPythonFile):
            return ExternalSourceAnalysisService._portable_module_name(
                request.source
            )
        return module.module_path or "unsupported_module"

    @staticmethod
    def _portable_module_name(source: UploadedPythonFile) -> str:
        return portable_upload_module_identity(source)

    @staticmethod
    def _sanitize_json_artifacts(
        *,
        module_output: Path,
        source_root: Path,
        output_root: Path,
        internal_module_name: str,
        public_module_name: str,
    ) -> None:
        source_values = {
            str(source_root.resolve()),
            source_root.resolve().as_posix(),
        }
        output_values = {
            str(output_root.resolve()),
            output_root.resolve().as_posix(),
        }

        def sanitize(value: object) -> object:
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if not isinstance(value, str):
                return value
            sanitized = value
            for root_value in sorted(source_values, key=len, reverse=True):
                sanitized = sanitized.replace(root_value, "<external-source>")
            for root_value in sorted(output_values, key=len, reverse=True):
                sanitized = sanitized.replace(root_value, ".")
            if internal_module_name != public_module_name:
                sanitized = sanitized.replace(internal_module_name, public_module_name)
            return sanitized

        for path in sorted(module_output.rglob("*.json")):
            resolved = path.resolve()
            if not resolved.is_relative_to(output_root.resolve()):
                raise RuntimeError("JSON artifact output root dışında.")
            payload = json.loads(path.read_text(encoding="utf-8"))
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _cleanup(
        self,
        acquired: ResolvedSourceTarget | None,
        temporary_workspace: Path | None,
    ) -> ExternalWorkspaceCleanupStatus:
        required = temporary_workspace is not None or (
            acquired is not None and acquired.cleanup_required
        )
        if not required:
            return ExternalWorkspaceCleanupStatus.NOT_REQUIRED
        try:
            if acquired is not None and acquired.cleanup_required:
                self._acquisition_service.cleanup(acquired)
            if temporary_workspace is not None and temporary_workspace.exists():
                self._remove_temporary_workspace(temporary_workspace)
        except (OSError, RuntimeError):
            return ExternalWorkspaceCleanupStatus.FAILED
        return ExternalWorkspaceCleanupStatus.COMPLETED

    @staticmethod
    def _write_result(result: ExternalSourceAnalysisResult) -> ExternalSourceAnalysisResult:
        result.output_root.mkdir(parents=True, exist_ok=True)
        temporary = result.report_path.with_name(
            f".{result.report_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, result.report_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return result

    @staticmethod
    def _remove_temporary_workspace(workspace: Path) -> None:
        normalized = workspace.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if (
            normalized.parent != temp_root
            or not normalized.name.startswith((_INLINE_WORKSPACE_PREFIX, _UPLOAD_WORKSPACE_PREFIX))
        ):
            raise RuntimeError("External temporary workspace cleanup kapsamı dışında.")

        def make_writable(function: Callable[..., object], path: str, _: object) -> None:
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(normalized, onerror=make_writable)
