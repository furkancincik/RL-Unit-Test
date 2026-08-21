from __future__ import annotations

import ast
import json
import os
import shutil
import stat
import tempfile
import time
import tokenize
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath

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
)
from models.source_acquisition_result import (
    DiscoveredPythonModule,
    ResolvedSourceTarget,
    SourceAcquisitionRequest,
    SourceTargetKind,
)
from services.source_acquisition_service import SourceAcquisitionService
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator


_INLINE_WORKSPACE_PREFIX = "rl-unit-test-inline-"
_UPLOAD_WORKSPACE_PREFIX = "rl-unit-test-upload-"


class ExternalSourceAnalysisService:
    """Harici Python kaynaklarını güvenli discovery veya açıkça trusted analiz eder."""

    def __init__(
        self,
        *,
        acquisition_service: SourceAcquisitionService | None = None,
        orchestrator_factory: Callable[[], SourceAnalysisOrchestrator] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._acquisition_service = acquisition_service or SourceAcquisitionService()
        self._orchestrator_factory = orchestrator_factory or SourceAnalysisOrchestrator
        self._clock = clock

    def run(self, request: ExternalSourceAnalysisRequest) -> ExternalSourceAnalysisResult:
        if not isinstance(request, ExternalSourceAnalysisRequest):
            raise TypeError("request ExternalSourceAnalysisRequest olmalıdır.")
        started = self._clock()
        output_root = request.configuration.output_root.resolve()
        report_path = output_root / "external_source_analysis_report.json"
        if (
            request.execution_policy is ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS
            and request.configuration.per_function_pipeline_timeout_seconds is None
        ):
            raise ValueError(
                "Trusted external dynamic analysis per-function pipeline timeout gerektirir."
            )
        temporary_workspace: Path | None = None
        acquired: ResolvedSourceTarget | None = None
        try:
            validation_issue = self._validate_payload(request)
            if validation_issue is not None:
                return self._finalize(
                    self._failed_result(
                        request, output_root, report_path, started, validation_issue
                    ),
                    acquired=None,
                    temporary_workspace=None,
                )

            source_request: SourceAcquisitionRequest
            source = request.source
            if isinstance(source, InlinePythonSource):
                temporary_workspace = Path(
                    tempfile.mkdtemp(prefix=_INLINE_WORKSPACE_PREFIX)
                ).resolve()
                source_file = temporary_workspace / f"source_{uuid.uuid4().hex}.py"
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
                source_file = temporary_workspace / f"source_{uuid.uuid4().hex}.py"
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

            acquired = self._acquisition_service.resolve(source_request)
            if not acquired.is_available:
                categories = tuple(issue.category.value for issue in acquired.issues)
                issue = categories[0] if categories else "SOURCE_ACQUISITION_FAILED"
                return self._finalize(
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
                )

            self._validate_output_separation(output_root, acquired.resolved_project_root)

            module_results = self._analyze_modules(request, acquired, output_root)
            status = self._derive_status(request.execution_policy, module_results)
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
            )
            return self._finalize(
                result,
                acquired=acquired,
                temporary_workspace=temporary_workspace,
            )
        except Exception:
            self._cleanup(acquired, temporary_workspace)
            raise

    def _analyze_modules(
        self,
        request: ExternalSourceAnalysisRequest,
        acquired: ResolvedSourceTarget,
        output_root: Path,
    ) -> tuple[ExternalModuleAnalysisResult, ...]:
        ordered = tuple(
            sorted(acquired.discovered_modules, key=lambda item: item.relative_path.casefold())
        )
        selected = self._selected_paths(request, ordered)
        results: list[ExternalModuleAnalysisResult] = []
        for ordinal, module in enumerate(ordered, start=1):
            module_issue = self._module_issue(acquired, module)
            if module.relative_path not in selected:
                if request.configuration.module_selection.mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
                    results.append(self._module_result(module, ExternalModuleStatus.SKIPPED_LIMIT))
                continue
            if not module.supported or module.module_path is None:
                results.append(
                    self._module_result(
                        module,
                        ExternalModuleStatus.UNSUPPORTED,
                        issue_category=module_issue,
                        issue_message="Module static discovery sonrasında güvenle analiz edilemiyor.",
                    )
                )
                continue
            if request.execution_policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
                results.append(self._module_result(module, ExternalModuleStatus.STATIC_ONLY))
                continue

            import_root = self._import_root(acquired.resolved_project_root, module)
            source_file = (acquired.resolved_project_root / module.relative_path).resolve()
            module_output = (output_root / f"module_{ordinal:03d}").resolve()
            if not module_output.is_relative_to(output_root):
                raise RuntimeError("Module output yolu root dışına çıktı.")
            try:
                project_result = self._orchestrator_factory().run(
                    source_file=source_file,
                    module_path=module.module_path,
                    output_root=module_output,
                    function_name=None,
                    all_functions=True,
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
            results = [
                replace(
                    item,
                    relative_path=public_name,
                    module_name=Path(public_name).stem,
                )
                for item in results
            ]
        return tuple(results)

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
                    raise ValueError("Explicit relative module path geçersiz.")
                normalized.add(path.as_posix())
            return frozenset(normalized)
        requested = frozenset(selection.values)
        if any(any(not part.isidentifier() for part in value.split(".")) for value in requested):
            raise ValueError("Explicit module name geçersiz.")
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
    ) -> ExternalAnalysisStatus:
        considered = tuple(
            item.status
            for item in modules
            if item.status is not ExternalModuleStatus.SKIPPED_LIMIT
        )
        if policy is ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY:
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
            encoded = source.source_text.encode("utf-8")
            if len(encoded) > limit:
                return "SOURCE_LIMIT_EXCEEDED"
            if ExternalSourceAnalysisService._has_invalid_control(source.source_text):
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
                readline = iter(source.file_bytes.splitlines(keepends=True)).__next__
                encoding, _ = tokenize.detect_encoding(readline)
                decoded = source.file_bytes.decode(encoding)
            except (LookupError, UnicodeError, SyntaxError, StopIteration):
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
            raise ValueError("output root source project içinde olamaz.")

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
        )

    def _finalize(
        self,
        result: ExternalSourceAnalysisResult,
        *,
        acquired: ResolvedSourceTarget | None,
        temporary_workspace: Path | None,
    ) -> ExternalSourceAnalysisResult:
        # Rapor önce persistent output'a atomik yazılır; cleanup sonucu aynı
        # dosyaya ikinci bir atomik metadata güncellemesiyle yansıtılır.
        self._write_result(result)
        cleanup_status = self._cleanup(acquired, temporary_workspace)
        issues = result.issues
        if cleanup_status is ExternalWorkspaceCleanupStatus.FAILED:
            issues = tuple(dict.fromkeys((*issues, "CLEANUP_FAILED")))
        finalized = replace(
            result,
            cleanup_status=cleanup_status,
            issues=issues,
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
            return Path(request.source.original_filename).stem
        return module.module_path or "unsupported_module"

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
        except Exception:
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
