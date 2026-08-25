from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from analyzer.python_source_reader import (
    PythonSourceEncodingError,
    read_python_source,
)
from models.source_acquisition_result import (
    DiscoveredPythonModule,
    ResolvedSourceTarget,
    SourceAcquisitionLimits,
    SourceAcquisitionRequest,
    SourceAcquisitionStatus,
    SourceDiscoveryIssue,
    SourceIssueCategory,
    SourceTargetKind,
    SourceWorkspaceOwnership,
)
from services.project_deadline import ProjectDeadline
from services.safe_filesystem_cleanup import (
    is_link_like,
    remove_workspace_tree,
)


IGNORED_SOURCE_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "tox",
        ".nox",
        "nox",
        "build",
        "dist",
        "site-packages",
        "node_modules",
        "output",
    }
)
SOURCE_WORKSPACE_PREFIX = "rl-unit-test-source-"
_GITHUB_COMPONENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})$")
_GIT_REF = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,127})$")
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class _ControlledAcquisitionError(Exception):
    def __init__(self, category: SourceIssueCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.safe_message = message


class SourceCleanupError(RuntimeError):
    """Tool-owned workspace güvenli biçimde temizlenemediğinde oluşur."""

    category = SourceIssueCategory.CLEANUP_FAILED


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


class GitSubprocessClient:
    """Git'i shell olmadan ve process-local güvenli ayarlarla çalıştırır."""

    def __init__(
        self,
        executable: str,
        runner: SubprocessRunner = subprocess.run,
    ) -> None:
        if not isinstance(executable, str) or not executable:
            raise TypeError("Git executable boş olmayan string olmalıdır.")
        self._executable = executable
        self._runner = runner

    def clone(
        self,
        *,
        url: str,
        destination: Path,
        ref: str | None,
        timeout_seconds: float,
    ) -> None:
        arguments = [
            self._executable,
            "-c",
            "credential.helper=",
            "-c",
            "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
            "-c",
            "protocol.file.allow=never",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--no-tags",
            "--no-recurse-submodules",
        ]
        if ref is not None:
            arguments.extend(("--branch", ref))
        arguments.extend(("--", url, str(destination)))
        try:
            completed = self._run(
                tuple(arguments), timeout_seconds, capture_output=False
            )
        except subprocess.TimeoutExpired as error:
            raise _ControlledAcquisitionError(
                SourceIssueCategory.CLONE_TIMEOUT,
                "Public repository clone süre sınırını aştı.",
            ) from error
        if completed.returncode != 0:
            raise _ControlledAcquisitionError(
                SourceIssueCategory.CLONE_FAILED,
                "Public repository güvenli biçimde clone edilemedi.",
            )

    def resolve_commit_sha(self, repository: Path, timeout_seconds: float) -> str:
        arguments = (
            self._executable,
            "-c",
            "credential.helper=",
            "-c",
            "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            "HEAD",
        )
        try:
            completed = self._run(
                arguments, timeout_seconds, capture_output=True
            )
        except subprocess.TimeoutExpired as error:
            raise _ControlledAcquisitionError(
                SourceIssueCategory.CLONE_TIMEOUT,
                "Repository commit doğrulaması süre sınırını aştı.",
            ) from error
        sha = (completed.stdout or "").strip()
        if completed.returncode != 0 or _COMMIT_SHA.fullmatch(sha) is None:
            raise _ControlledAcquisitionError(
                SourceIssueCategory.CLONE_FAILED,
                "Repository commit kimliği güvenli biçimde doğrulanamadı.",
            )
        return sha.lower()

    def _run(
        self,
        arguments: tuple[str, ...],
        timeout_seconds: float,
        *,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        output_arguments = (
            {"capture_output": True}
            if capture_output
            else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        )
        return self._runner(
            arguments,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
            env=self._safe_environment(),
            **output_arguments,
        )

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        allowed = (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
        )
        environment = {
            key: os.environ[key]
            for key in allowed
            if key in os.environ
        }
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GCM_INTERACTIVE": "Never",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
        )
        return environment


class SourceAcquisitionReportWriter:
    """Acquisition envanterini atomik ve güvenli JSON olarak yazar."""

    @staticmethod
    def write(result: ResolvedSourceTarget, path: str | Path) -> Path:
        if not isinstance(result, ResolvedSourceTarget):
            raise TypeError("result ResolvedSourceTarget olmalıdır.")
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


class SourceAcquisitionService:
    """Yerel veya public GitHub kaynaklarını çalıştırmadan keşfeder."""

    def __init__(
        self,
        *,
        subprocess_runner: SubprocessRunner = subprocess.run,
        git_executable: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._runner = subprocess_runner
        self._git_executable = git_executable
        self._clock = clock
        self._owned_workspaces: set[Path] = set()
        self._cleaned_workspaces: set[Path] = set()
        self._workspace_lock = threading.Lock()

    def resolve(
        self,
        request: SourceAcquisitionRequest,
        *,
        project_deadline: ProjectDeadline | None = None,
    ) -> ResolvedSourceTarget:
        if not isinstance(request, SourceAcquisitionRequest):
            raise TypeError("request SourceAcquisitionRequest olmalıdır.")
        started = self._clock()
        if project_deadline is not None and not isinstance(
            project_deadline, ProjectDeadline
        ):
            raise TypeError("project_deadline ProjectDeadline olmalıdır.")
        if request.source_kind is SourceTargetKind.PUBLIC_GITHUB_REPOSITORY:
            return self._resolve_github(request, started, project_deadline)
        return self._resolve_local(request, started)

    @contextmanager
    def acquired(
        self,
        request: SourceAcquisitionRequest,
    ) -> Iterator[ResolvedSourceTarget]:
        result = self.resolve(request)
        try:
            yield result
        finally:
            self.cleanup(result)

    def cleanup(self, result: ResolvedSourceTarget) -> bool:
        if not isinstance(result, ResolvedSourceTarget):
            raise TypeError("result ResolvedSourceTarget olmalıdır.")
        if (
            result.workspace_ownership is SourceWorkspaceOwnership.USER_OWNED
            or not result.cleanup_required
        ):
            return False
        workspace = result.resolved_project_root.parent.resolve()
        system_temp = Path(tempfile.gettempdir()).resolve()
        with self._workspace_lock:
            if workspace in self._cleaned_workspaces:
                return False
            if workspace not in self._owned_workspaces:
                raise RuntimeError("Workspace bu resolver tarafından owned değil.")
            if (
                workspace.parent != system_temp
                or not workspace.name.startswith(SOURCE_WORKSPACE_PREFIX)
            ):
                raise RuntimeError("Tool workspace güvenli cleanup kapsamı dışında.")
            try:
                self._remove_workspace_tree(workspace)
            except OSError as error:
                raise SourceCleanupError(
                    "Tool-owned workspace temizlenemedi."
                ) from error
            self._owned_workspaces.remove(workspace)
            self._cleaned_workspaces.add(workspace)
            return True

    def _resolve_local(
        self,
        request: SourceAcquisitionRequest,
        started: float,
    ) -> ResolvedSourceTarget:
        raw_origin = request.origin.strip()
        path = Path(raw_origin).resolve() if raw_origin else Path(raw_origin)
        valid = (
            path.is_file()
            if request.source_kind is SourceTargetKind.LOCAL_FILE
            else path.is_dir()
        )
        if not raw_origin or not valid or (
            request.source_kind is SourceTargetKind.LOCAL_FILE
            and path.suffix.lower() != ".py"
        ):
            return self._result(
                request=request,
                origin=str(path),
                root=path.parent if path.suffix else path,
                ownership=SourceWorkspaceOwnership.USER_OWNED,
                repository_name=path.stem if path.suffix else path.name,
                status=SourceAcquisitionStatus.FAILED,
                issues=(self._issue(SourceIssueCategory.INVALID_SOURCE, "Yerel source mevcut ve beklenen türde olmalıdır."),),
                started=started,
            )
        if request.source_kind is SourceTargetKind.LOCAL_FILE:
            project_root = path.parent.parent if path.name == "__init__.py" else path.parent
            candidates = (path,)
        else:
            project_root = path
            limit_issue = self._repository_limit_issue(project_root, request.limits)
            if limit_issue is not None:
                return self._result(
                    request=request,
                    origin=str(path),
                    root=project_root,
                    ownership=SourceWorkspaceOwnership.USER_OWNED,
                    repository_name=path.name,
                    status=SourceAcquisitionStatus.LIMIT_EXCEEDED,
                    issues=(limit_issue,),
                    started=started,
                )
            candidates, traversal_issues = self._discover_candidates(
                project_root, request.include_tests, request.limits
            )
            if self._has_limit_issue(traversal_issues):
                return self._result(
                    request=request,
                    origin=str(path),
                    root=project_root,
                    ownership=SourceWorkspaceOwnership.USER_OWNED,
                    repository_name=path.name,
                    status=SourceAcquisitionStatus.LIMIT_EXCEEDED,
                    issues=traversal_issues,
                    started=started,
                )
            return self._analyze_result(
                request=request,
                origin=str(path),
                root=project_root,
                ownership=SourceWorkspaceOwnership.USER_OWNED,
                repository_name=path.name,
                candidates=candidates,
                initial_issues=traversal_issues,
                started=started,
            )
        return self._analyze_result(
            request=request,
            origin=str(path),
            root=project_root,
            ownership=SourceWorkspaceOwnership.USER_OWNED,
            repository_name=path.stem,
            candidates=candidates,
            initial_issues=(),
            started=started,
        )

    def _resolve_github(
        self,
        request: SourceAcquisitionRequest,
        started: float,
        project_deadline: ProjectDeadline | None,
    ) -> ResolvedSourceTarget:
        try:
            normalized, owner, repository = self._validate_github(
                request.origin, request.ref
            )
        except _ControlledAcquisitionError as error:
            return self._result(
                request=request,
                origin=request.origin.strip(),
                root=Path(),
                ownership=SourceWorkspaceOwnership.TOOL_TEMPORARY,
                repository_name=None,
                github_owner=None,
                github_repository=None,
                status=SourceAcquisitionStatus.FAILED,
                issues=(self._issue(error.category, error.safe_message),),
                started=started,
            )
        executable = self._git_executable or shutil.which("git")
        if executable is None:
            return self._result(
                request=request,
                origin=normalized,
                root=Path(),
                ownership=SourceWorkspaceOwnership.TOOL_TEMPORARY,
                repository_name=repository,
                github_owner=owner,
                github_repository=repository,
                status=SourceAcquisitionStatus.FAILED,
                issues=(self._issue(SourceIssueCategory.GIT_NOT_AVAILABLE, "Git executable bulunamadı."),),
                started=started,
            )
        workspace = Path(tempfile.mkdtemp(prefix=SOURCE_WORKSPACE_PREFIX)).resolve()
        repository_root = (workspace / "repository").resolve()
        if not repository_root.is_relative_to(workspace):
            shutil.rmtree(workspace)
            raise RuntimeError("Repository workspace root dışına çıktı.")
        with self._workspace_lock:
            self._owned_workspaces.add(workspace)
        client = GitSubprocessClient(executable, self._runner)
        try:
            clone_timeout = self._deadline_timeout(
                float(request.limits.clone_timeout_seconds),
                project_deadline,
            )
            client.clone(
                url=normalized,
                destination=repository_root,
                ref=request.ref,
                timeout_seconds=clone_timeout,
            )
            commit_timeout = self._deadline_timeout(
                float(request.limits.clone_timeout_seconds),
                project_deadline,
            )
            commit_sha = client.resolve_commit_sha(
                repository_root, commit_timeout
            )
            limit_issue = self._repository_limit_issue(repository_root, request.limits)
            if limit_issue is not None:
                return self._failed_temporary_result(
                    request, normalized, repository_root, repository, owner,
                    SourceAcquisitionStatus.LIMIT_EXCEEDED, (limit_issue,), started
                )
            candidates, traversal_issues = self._discover_candidates(
                repository_root, request.include_tests, request.limits
            )
            if self._has_limit_issue(traversal_issues):
                return self._failed_temporary_result(
                    request, normalized, repository_root, repository, owner,
                    SourceAcquisitionStatus.LIMIT_EXCEEDED, traversal_issues, started,
                    resolved_commit_sha=commit_sha,
                )
            return self._analyze_result(
                request=request,
                origin=normalized,
                root=repository_root,
                ownership=SourceWorkspaceOwnership.TOOL_TEMPORARY,
                repository_name=repository,
                candidates=candidates,
                initial_issues=traversal_issues,
                started=started,
                github_owner=owner,
                github_repository=repository,
                resolved_commit_sha=commit_sha,
                cleanup_required=True,
            )
        except _ControlledAcquisitionError as error:
            return self._failed_temporary_result(
                request, normalized, repository_root, repository, owner,
                SourceAcquisitionStatus.FAILED,
                (self._issue(error.category, error.safe_message),), started,
            )
        except Exception:
            self._remove_owned_workspace(workspace)
            raise

    @staticmethod
    def _deadline_timeout(
        configured_timeout: float,
        project_deadline: ProjectDeadline | None,
    ) -> float:
        if project_deadline is None:
            return configured_timeout
        remaining = project_deadline.remaining_seconds()
        if remaining is None:
            return configured_timeout
        if remaining <= 0.0:
            raise _ControlledAcquisitionError(
                SourceIssueCategory.CLONE_TIMEOUT,
                "Project deadline public repository acquisition sırasında aşıldı.",
            )
        return min(configured_timeout, remaining)

    def _failed_temporary_result(
        self,
        request: SourceAcquisitionRequest,
        origin: str,
        root: Path,
        repository: str,
        owner: str,
        status: SourceAcquisitionStatus,
        issues: tuple[SourceDiscoveryIssue, ...],
        started: float,
        resolved_commit_sha: str | None = None,
    ) -> ResolvedSourceTarget:
        workspace = root.parent.resolve()
        self._remove_owned_workspace(workspace)
        return self._result(
            request=request,
            origin=origin,
            root=root,
            ownership=SourceWorkspaceOwnership.TOOL_TEMPORARY,
            repository_name=repository,
            github_owner=owner,
            github_repository=repository,
            resolved_commit_sha=resolved_commit_sha,
            status=status,
            issues=issues,
            started=started,
            cleanup_required=False,
        )

    def _remove_owned_workspace(self, workspace: Path) -> None:
        system_temp = Path(tempfile.gettempdir()).resolve()
        if (
            workspace.parent != system_temp
            or not workspace.name.startswith(SOURCE_WORKSPACE_PREFIX)
        ):
            raise RuntimeError("Partial workspace güvenli cleanup kapsamı dışında.")
        shutil.rmtree(workspace, ignore_errors=False)
        with self._workspace_lock:
            self._owned_workspaces.discard(workspace)
            self._cleaned_workspaces.add(workspace)

    @staticmethod
    def _validate_github(origin: str, ref: str | None) -> tuple[str, str, str]:
        if not origin or any(ord(character) < 32 for character in origin):
            raise _ControlledAcquisitionError(SourceIssueCategory.INVALID_GITHUB_URL, "GitHub URL geçersiz.")
        parsed = urlsplit(origin.strip())
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname is None
            or parsed.hostname.lower() != "github.com"
            or parsed.netloc.lower() != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or unquote(parsed.path) != parsed.path
        ):
            raise _ControlledAcquisitionError(SourceIssueCategory.INVALID_GITHUB_URL, "Yalnız exact github.com HTTPS repository URL kabul edilir.")
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 2:
            raise _ControlledAcquisitionError(SourceIssueCategory.INVALID_GITHUB_URL, "GitHub URL owner/repository içermelidir.")
        owner, repository = parts
        if repository.lower().endswith(".git"):
            repository = repository[:-4]
        if (
            not owner
            or not repository
            or owner in {".", ".."}
            or repository in {".", ".."}
            or _GITHUB_COMPONENT.fullmatch(owner) is None
            or _GITHUB_COMPONENT.fullmatch(repository) is None
        ):
            raise _ControlledAcquisitionError(SourceIssueCategory.INVALID_GITHUB_URL, "GitHub owner/repository geçersiz.")
        if ref is not None and not SourceAcquisitionService._valid_ref(ref):
            raise _ControlledAcquisitionError(SourceIssueCategory.INVALID_GITHUB_URL, "Git ref güvenli allowlist politikasına uymuyor.")
        return f"https://github.com/{owner}/{repository}", owner, repository

    @staticmethod
    def _valid_ref(ref: str) -> bool:
        return bool(
            ref
            and _GIT_REF.fullmatch(ref)
            and ".." not in ref
            and "//" not in ref
            and "@{" not in ref
            and not ref.endswith(("/", ".", ".lock"))
            and not ref.startswith(("/", "."))
        )

    def _analyze_result(
        self,
        *,
        request: SourceAcquisitionRequest,
        origin: str,
        root: Path,
        ownership: SourceWorkspaceOwnership,
        repository_name: str,
        candidates: tuple[Path, ...],
        initial_issues: tuple[SourceDiscoveryIssue, ...],
        started: float,
        github_owner: str | None = None,
        github_repository: str | None = None,
        resolved_commit_sha: str | None = None,
        cleanup_required: bool = False,
    ) -> ResolvedSourceTarget:
        issues = list(initial_issues)
        modules: list[DiscoveredPythonModule] = []
        total_bytes = 0
        for path in candidates:
            size = path.stat().st_size
            if size > request.limits.maximum_single_file_bytes:
                issues.append(self._issue(SourceIssueCategory.FILE_LIMIT_EXCEEDED, "Python dosyası tekil boyut sınırını aştı.", self._relative(root, path)))
                return self._result(
                    request=request, origin=origin, root=root, ownership=ownership,
                    repository_name=repository_name, github_owner=github_owner,
                    github_repository=github_repository, resolved_commit_sha=resolved_commit_sha,
                    status=SourceAcquisitionStatus.LIMIT_EXCEEDED, modules=tuple(modules),
                    issues=tuple(issues), total_bytes=total_bytes, started=started,
                    cleanup_required=cleanup_required,
                )
            if total_bytes + size > request.limits.maximum_total_python_bytes:
                issues.append(self._issue(SourceIssueCategory.FILE_LIMIT_EXCEEDED, "Toplam Python byte sınırı aşıldı.", self._relative(root, path)))
                return self._result(
                    request=request, origin=origin, root=root, ownership=ownership,
                    repository_name=repository_name, github_owner=github_owner,
                    github_repository=github_repository, resolved_commit_sha=resolved_commit_sha,
                    status=SourceAcquisitionStatus.LIMIT_EXCEEDED, modules=tuple(modules),
                    issues=tuple(issues), total_bytes=total_bytes, started=started,
                    cleanup_required=cleanup_required,
                )
            total_bytes += size
            module, module_issues = self._analyze_file(root, path, size)
            modules.append(module)
            issues.extend(module_issues)
        if not modules:
            issues.append(self._issue(SourceIssueCategory.NO_PYTHON_FILES, "Keşfedilebilir Python dosyası bulunamadı."))
            status = (
                SourceAcquisitionStatus.PARTIAL
                if request.source_kind
                is SourceTargetKind.PUBLIC_GITHUB_REPOSITORY
                and resolved_commit_sha is not None
                else SourceAcquisitionStatus.FAILED
            )
        elif issues:
            status = SourceAcquisitionStatus.PARTIAL
        else:
            status = SourceAcquisitionStatus.COMPLETED
        return self._result(
            request=request, origin=origin, root=root, ownership=ownership,
            repository_name=repository_name, github_owner=github_owner,
            github_repository=github_repository, resolved_commit_sha=resolved_commit_sha,
            status=status, modules=tuple(modules), issues=tuple(issues),
            total_bytes=total_bytes, started=started, cleanup_required=cleanup_required,
        )

    def _analyze_file(
        self, root: Path, path: Path, size: int
    ) -> tuple[DiscoveredPythonModule, tuple[SourceDiscoveryIssue, ...]]:
        relative = self._relative(root, path)
        issues: list[SourceDiscoveryIssue] = []
        source: str | None = None
        encoding: str | None = None
        try:
            decoded = read_python_source(path)
            encoding = decoded.encoding
            source = decoded.text
        except PythonSourceEncodingError:
            issues.append(self._issue(SourceIssueCategory.UNSUPPORTED_ENCODING, "Python source encoding çözümlenemedi.", relative))
        syntax_valid = False
        function_count = 0
        function_names: tuple[str, ...] = ()
        if source is not None:
            try:
                tree = ast.parse(source, filename=relative)
                syntax_valid = True
                function_names = tuple(
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                function_count = len(function_names)
            except SyntaxError:
                issues.append(self._issue(SourceIssueCategory.SYNTAX_ERROR, "Python syntax validation başarısız.", relative))
        candidates, roots = self._module_candidates(root, path)
        module_path = candidates[0] if len(candidates) == 1 else None
        package_root = roots[0] if len(candidates) == 1 else None
        if not candidates:
            issues.append(self._issue(SourceIssueCategory.MODULE_PATH_UNRESOLVED, "Import edilebilir module path güvenle çıkarılamadı.", relative))
        elif len(candidates) > 1:
            issues.append(self._issue(SourceIssueCategory.MODULE_PATH_AMBIGUOUS, "Birden fazla geçerli module root bulundu.", relative))
        supported = syntax_valid and module_path is not None
        return DiscoveredPythonModule(
            relative_path=relative,
            file_size_bytes=size,
            encoding=encoding,
            syntax_valid=syntax_valid,
            top_level_function_count=function_count,
            module_path=module_path,
            module_path_candidates=candidates,
            package_root=package_root,
            supported=supported,
            top_level_function_names=function_names,
        ), tuple(issues)

    @staticmethod
    def _module_candidates(root: Path, path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidate_roots = [(root, ".")]
        src_root = root / "src"
        if src_root.is_dir():
            candidate_roots.append((src_root, "src"))
        found: dict[str, str] = {}
        for package_root, label in candidate_roots:
            try:
                relative = path.relative_to(package_root)
            except ValueError:
                continue
            parts = list(relative.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            if not parts or any(not part.isidentifier() for part in parts):
                continue
            parent_parts = parts[:-1]
            if path.name == "__init__.py":
                parent_parts = parts
            if any(
                not (package_root.joinpath(*parts[:index], "__init__.py")).is_file()
                for index in range(1, len(parent_parts) + 1)
            ):
                continue
            found[".".join(parts)] = label
        ordered = tuple(sorted(found, key=str.casefold))
        return ordered, tuple(found[value] for value in ordered)

    def _discover_candidates(
        self,
        root: Path,
        include_tests: bool,
        limits: SourceAcquisitionLimits,
    ) -> tuple[tuple[Path, ...], tuple[SourceDiscoveryIssue, ...]]:
        resolved_root = root.resolve()
        candidates: list[Path] = []
        issues: list[SourceDiscoveryIssue] = []
        stack = [resolved_root]
        stop = False
        while stack and not stop:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name.casefold(), reverse=True)
            except OSError:
                issues.append(self._issue(SourceIssueCategory.PATH_OUTSIDE_ROOT, "Directory güvenli biçimde taranamadı.", self._relative(resolved_root, directory)))
                continue
            for entry in entries:
                candidate = Path(entry.path)
                relative_path = Path(os.path.relpath(candidate, resolved_root))
                relative = relative_path.as_posix()
                if self._is_link_like(candidate):
                    issues.append(self._issue(SourceIssueCategory.SYMLINK_SKIPPED, "Symlink/junction source discovery dışında bırakıldı.", relative))
                    continue
                try:
                    resolved_candidate = candidate.resolve()
                except OSError:
                    issues.append(self._issue(SourceIssueCategory.PATH_OUTSIDE_ROOT, "Path güvenli biçimde resolve edilemedi.", relative))
                    continue
                if not resolved_candidate.is_relative_to(resolved_root):
                    issues.append(self._issue(SourceIssueCategory.PATH_OUTSIDE_ROOT, "Path project root dışına çıktı.", relative))
                    continue
                depth = len(relative_path.parts)
                if depth > limits.maximum_path_depth:
                    issues.append(self._issue(SourceIssueCategory.PATH_DEPTH_EXCEEDED, "Maximum path depth aşıldı.", relative))
                    stop = True
                    break
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in IGNORED_SOURCE_DIRECTORIES:
                        continue
                    if not include_tests and entry.name.lower() in {"test", "tests"}:
                        continue
                    stack.append(resolved_candidate)
                    continue
                if not entry.is_file(follow_symlinks=False) or candidate.suffix.lower() != ".py":
                    continue
                if not include_tests and self._is_test_file(candidate.name):
                    continue
                if len(candidates) >= limits.maximum_python_file_count:
                    issues.append(self._issue(SourceIssueCategory.FILE_LIMIT_EXCEEDED, "Maximum Python file count aşıldı.", relative))
                    stop = True
                    break
                candidates.append(resolved_candidate)
        return tuple(sorted(candidates, key=lambda value: self._relative(resolved_root, value).casefold())), tuple(issues)

    @staticmethod
    def _repository_limit_issue(root: Path, limits: SourceAcquisitionLimits) -> SourceDiscoveryIssue | None:
        total = 0
        stack = [root]
        while stack:
            directory = stack.pop()
            for entry in os.scandir(directory):
                candidate = Path(entry.path)
                if SourceAcquisitionService._is_link_like(candidate):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(candidate)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    if total > limits.maximum_repository_bytes:
                        return SourceAcquisitionService._issue(
                            SourceIssueCategory.REPOSITORY_LIMIT_EXCEEDED,
                            "Repository toplam byte sınırını aştı.",
                        )
        return None

    @staticmethod
    def _is_link_like(path: Path) -> bool:
        return is_link_like(path)

    @staticmethod
    def _remove_workspace_tree(workspace: Path) -> None:
        remove_workspace_tree(workspace, rmtree=shutil.rmtree)

    @staticmethod
    def _is_test_file(name: str) -> bool:
        lower = name.lower()
        return lower.startswith("test_") or lower.endswith("_test.py")

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _has_limit_issue(issues: tuple[SourceDiscoveryIssue, ...]) -> bool:
        return any(
            issue.category in {
                SourceIssueCategory.FILE_LIMIT_EXCEEDED,
                SourceIssueCategory.PATH_DEPTH_EXCEEDED,
                SourceIssueCategory.REPOSITORY_LIMIT_EXCEEDED,
            }
            for issue in issues
        )

    @staticmethod
    def _issue(
        category: SourceIssueCategory,
        message: str,
        relative_path: str | None = None,
    ) -> SourceDiscoveryIssue:
        return SourceDiscoveryIssue(category, message, relative_path)

    def _result(
        self,
        *,
        request: SourceAcquisitionRequest,
        origin: str,
        root: Path,
        ownership: SourceWorkspaceOwnership,
        repository_name: str | None,
        status: SourceAcquisitionStatus,
        issues: tuple[SourceDiscoveryIssue, ...],
        started: float,
        github_owner: str | None = None,
        github_repository: str | None = None,
        resolved_commit_sha: str | None = None,
        modules: tuple[DiscoveredPythonModule, ...] = (),
        total_bytes: int = 0,
        cleanup_required: bool = False,
    ) -> ResolvedSourceTarget:
        return ResolvedSourceTarget(
            source_kind=request.source_kind,
            normalized_origin=origin,
            resolved_project_root=root,
            workspace_ownership=ownership,
            repository_name=repository_name,
            github_owner=github_owner,
            github_repository=github_repository,
            requested_ref=request.ref,
            resolved_commit_sha=resolved_commit_sha,
            status=status,
            discovered_modules=modules,
            issues=issues,
            total_scanned_bytes=total_bytes,
            duration_seconds=max(0.0, self._clock() - started),
            cleanup_required=cleanup_required,
            include_tests=request.include_tests,
            limits=request.limits,
        )
