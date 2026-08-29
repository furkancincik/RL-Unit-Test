from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from models.source_acquisition_result import (
    SourceAcquisitionLimits,
    SourceAcquisitionRequest,
    SourceAcquisitionStatus,
    SourceIssueCategory,
    SourceTargetKind,
    SourceWorkspaceOwnership,
)
from services.source_acquisition_service import (
    GitSubprocessClient,
    SourceAcquisitionReportWriter,
    SourceCleanupError,
    SourceAcquisitionService,
)
from services.project_deadline import ProjectDeadline


def _request(kind: SourceTargetKind, origin: str | Path, **kwargs: object) -> SourceAcquisitionRequest:
    return SourceAcquisitionRequest(source_kind=kind, origin=str(origin), **kwargs)


def _categories(result: object) -> set[SourceIssueCategory]:
    return {issue.category for issue in result.issues}  # type: ignore[attr-defined]


def test_local_python_file_resolves_without_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = tmp_path / "module.py"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef target():\n    return 1\n",
        encoding="utf-8",
    )
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_FILE, source))
    assert result.status is SourceAcquisitionStatus.COMPLETED
    assert result.discovered_modules[0].module_path == "module"
    assert result.discovered_modules[0].top_level_function_count == 1
    assert not marker.exists()
    assert result.workspace_ownership is SourceWorkspaceOwnership.USER_OWNED


@pytest.mark.parametrize(
    ("name", "category"),
    (("missing.py", SourceIssueCategory.INVALID_SOURCE), ("notes.txt", SourceIssueCategory.INVALID_SOURCE)),
)
def test_invalid_local_source_is_controlled(tmp_path: Path, name: str, category: SourceIssueCategory) -> None:
    path = tmp_path / name
    if path.suffix != ".py":
        path.write_text("text", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_FILE, path))
    assert result.status is SourceAcquisitionStatus.FAILED
    assert category in _categories(result)


def test_syntax_and_encoding_issues_do_not_block_valid_files(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("def good():\n    return 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "encoded.py").write_bytes("# coding: latin-1\nvalue = 'caf\xe9'\n".encode("latin-1"))
    (tmp_path / "bad_encoding.py").write_bytes(b"# coding: missing-codec\nvalue=1\n")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    modules = {item.relative_path: item for item in result.discovered_modules}
    assert modules["good.py"].syntax_valid is True
    assert modules["broken.py"].syntax_valid is False
    assert modules["encoded.py"].encoding == "iso-8859-1"
    assert SourceIssueCategory.SYNTAX_ERROR in _categories(result)
    assert SourceIssueCategory.UNSUPPORTED_ENCODING in _categories(result)


def test_directory_discovery_is_deterministic_and_ignores_policy(tmp_path: Path) -> None:
    for relative in ("z.py", "a.py", "package/__init__.py", "package/service.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")
    for relative in (".venv/ignored.py", ".git/ignored.py", "node_modules/ignored.py", "output/ignored.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    paths = tuple(item.relative_path for item in result.discovered_modules)
    assert paths == tuple(sorted(paths, key=str.casefold))
    assert paths == ("a.py", "package/__init__.py", "package/service.py", "z.py")


def test_include_tests_policy_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value=1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_module.py").write_text("value=1\n", encoding="utf-8")
    excluded = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    included = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path, include_tests=True))
    assert [item.relative_path for item in excluded.discovered_modules] == ["module.py"]
    assert [item.relative_path for item in included.discovered_modules] == ["module.py", "tests/test_module.py"]


def test_empty_directory_reports_no_python_files(tmp_path: Path) -> None:
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    assert result.status is SourceAcquisitionStatus.FAILED
    assert SourceIssueCategory.NO_PYTHON_FILES in _categories(result)


def test_module_path_inference_supports_packages_init_and_src_layout(tmp_path: Path) -> None:
    paths = (
        "module.py",
        "package/__init__.py",
        "package/service.py",
        "src/second_package/__init__.py",
        "src/second_package/helper.py",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value=1\n", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    values = {item.relative_path: item.module_path for item in result.discovered_modules}
    assert values == {
        "module.py": "module",
        "package/__init__.py": "package",
        "package/service.py": "package.service",
        "src/second_package/__init__.py": "second_package",
        "src/second_package/helper.py": "second_package.helper",
    }


def test_real_local_project_acceptance_preserves_user_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    files = {
        "module.py": "value = 1\n",
        "package/__init__.py": "",
        "package/service.py": "def service():\n    return 1\n",
        "src/second_package/__init__.py": "",
        "src/second_package/helper.py": "def helper():\n    return 2\n",
        "tests/test_module.py": "def test_module():\n    assert True\n",
        ".venv/ignored.py": "raise RuntimeError('must not execute')\n",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    service = SourceAcquisitionService()

    production = service.resolve(
        _request(SourceTargetKind.LOCAL_DIRECTORY, project)
    )
    with_tests = service.resolve(
        _request(SourceTargetKind.LOCAL_DIRECTORY, project, include_tests=True)
    )

    modules = {item.relative_path: item.module_path for item in production.discovered_modules}
    assert modules == {
        "module.py": "module",
        "package/__init__.py": "package",
        "package/service.py": "package.service",
        "src/second_package/__init__.py": "second_package",
        "src/second_package/helper.py": "second_package.helper",
    }
    assert "tests/test_module.py" not in modules
    assert "tests/test_module.py" in {
        item.relative_path for item in with_tests.discovered_modules
    }
    assert all(".venv" not in item.relative_path for item in with_tests.discovered_modules)
    assert service.cleanup(production) is False
    assert (project / ".venv" / "ignored.py").is_file()


def test_invalid_and_ambiguous_module_layouts_are_reported(tmp_path: Path) -> None:
    invalid = tmp_path / "bad-name" / "module.py"
    invalid.parent.mkdir()
    invalid.write_text("value=1\n", encoding="utf-8")
    src = tmp_path / "src"
    package = src / "pkg"
    package.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "service.py").write_text("", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path))
    modules = {item.relative_path: item for item in result.discovered_modules}
    assert modules["bad-name/module.py"].module_path is None
    assert SourceIssueCategory.MODULE_PATH_UNRESOLVED in _categories(result)
    assert modules["src/pkg/service.py"].module_path is None
    assert set(modules["src/pkg/service.py"].module_path_candidates) == {"src.pkg.service", "pkg.service"}
    assert SourceIssueCategory.MODULE_PATH_AMBIGUOUS in _categories(result)


def test_symlink_is_skipped_and_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.py").write_text("value=1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("secret=1\n", encoding="utf-8")
    link = root / "escape.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable.")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, root))
    assert [item.relative_path for item in result.discovered_modules] == ["inside.py"]
    assert SourceIssueCategory.SYMLINK_SKIPPED in _categories(result)


@pytest.mark.parametrize(
    ("limits", "category"),
    (
        (SourceAcquisitionLimits(maximum_path_depth=1), SourceIssueCategory.PATH_DEPTH_EXCEEDED),
        (SourceAcquisitionLimits(maximum_python_file_count=1), SourceIssueCategory.FILE_LIMIT_EXCEEDED),
        (SourceAcquisitionLimits(maximum_single_file_bytes=4), SourceIssueCategory.FILE_LIMIT_EXCEEDED),
        (SourceAcquisitionLimits(maximum_total_python_bytes=5), SourceIssueCategory.FILE_LIMIT_EXCEEDED),
        (SourceAcquisitionLimits(maximum_repository_bytes=5), SourceIssueCategory.REPOSITORY_LIMIT_EXCEEDED),
    ),
)
def test_resource_limits_stop_discovery(tmp_path: Path, limits: SourceAcquisitionLimits, category: SourceIssueCategory) -> None:
    deep = tmp_path / "one" / "two"
    deep.mkdir(parents=True)
    (tmp_path / "first.py").write_text("value=1\n", encoding="utf-8")
    (deep / "second.py").write_text("value=2\n", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_DIRECTORY, tmp_path, limits=limits))
    assert category in _categories(result)
    assert result.status is SourceAcquisitionStatus.LIMIT_EXCEEDED


@pytest.mark.parametrize(
    "url",
    (
        "http://github.com/owner/repo",
        "git://github.com/owner/repo",
        "ssh://git@github.com/owner/repo",
        "git@github.com:owner/repo.git",
        "file:///tmp/repo",
        "https://localhost/owner/repo",
        "https://127.0.0.1/owner/repo",
        "https://github.com.evil.example/owner/repo",
        "https://user@github.com/owner/repo",
        "https://github.com/owner/repo?ref=main",
        "https://github.com/owner/repo#main",
        "https://github.com/owner/repo/extra",
        "https://github.com/owner/%2e%2e",
        "https://github.com/../repo",
        "https://github.com/owner/bad repo",
    ),
)
def test_invalid_github_urls_are_rejected_without_git(url: str) -> None:
    runner = Mock()
    result = SourceAcquisitionService(subprocess_runner=runner).resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, url)
    )
    assert result.status is SourceAcquisitionStatus.FAILED
    assert SourceIssueCategory.INVALID_GITHUB_URL in _categories(result)
    runner.assert_not_called()


@pytest.mark.parametrize("ref", ("../main", "main;calc", "refs//heads/main", "main@{1}", "bad ref", ".lock"))
def test_invalid_refs_are_rejected(ref: str) -> None:
    result = SourceAcquisitionService(subprocess_runner=Mock()).resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo", ref=ref)
    )
    assert SourceIssueCategory.INVALID_GITHUB_URL in _categories(result)


def test_public_github_validation_is_reusable_before_acquisition() -> None:
    normalized, owner, repository = (
        SourceAcquisitionService.validate_public_github_repository(
            "https://github.com/Owner/Repo.git",
            "feature/safe-ref",
        )
    )

    assert normalized == "https://github.com/Owner/Repo"
    assert owner == "Owner"
    assert repository == "Repo"
    with pytest.raises(ValueError, match="Git ref"):
        SourceAcquisitionService.validate_public_github_repository(
            "https://github.com/owner/repo",
            "../main",
        )


def test_git_missing_is_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.source_acquisition_service.shutil.which", lambda _: None)
    result = SourceAcquisitionService().resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo.git")
    )
    assert result.normalized_origin == "https://github.com/owner/repo"
    assert SourceIssueCategory.GIT_NOT_AVAILABLE in _categories(result)


def test_clone_policy_uses_safe_arguments_and_environment(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((tuple(arguments), kwargs))
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "cloned", "")
        return subprocess.CompletedProcess(arguments, 0, "a" * 40 + "\n", "")

    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    result = service.resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/Owner/Repo.git", ref="main")
    )
    clone, kwargs = calls[0]
    assert clone[0] == "git"
    assert "--depth" in clone and clone[clone.index("--depth") + 1] == "1"
    assert "--single-branch" in clone and "--no-tags" in clone and "--no-recurse-submodules" in clone
    assert "--branch" in clone and clone[clone.index("--branch") + 1] == "main"
    assert kwargs["shell"] is False
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert "capture_output" not in kwargs
    environment = kwargs["env"]
    assert environment["GIT_TERMINAL_PROMPT"] == "0"  # type: ignore[index]
    assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"  # type: ignore[index]
    assert "GITHUB_TOKEN" not in environment  # type: ignore[operator]
    assert result.resolved_commit_sha == "a" * 40
    assert not (result.resolved_project_root / ".git" / "modules").exists()
    assert result.resolved_project_root.parent != tmp_path
    service.cleanup(result)


def test_exact_commit_ref_is_checked_out_without_branch_interpretation() -> None:
    commit = "c" * 40
    calls: list[tuple[str, ...]] = []

    def runner(
        arguments: tuple[str, ...], **_: object
    ) -> subprocess.CompletedProcess[str]:
        call = tuple(arguments)
        calls.append(call)
        if "clone" in call:
            destination = Path(call[-1])
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(call, 0, "", "")
        if "rev-parse" in call:
            return subprocess.CompletedProcess(call, 0, commit + "\n", "")
        return subprocess.CompletedProcess(call, 0, "", "")

    service = SourceAcquisitionService(
        subprocess_runner=runner,
        git_executable="git",
    )
    result = service.resolve(
        _request(
            SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
            "https://github.com/owner/repo",
            ref=commit,
        )
    )

    clone = next(call for call in calls if "clone" in call)
    fetch = next(call for call in calls if "fetch" in call)
    checkout = next(call for call in calls if "checkout" in call)
    assert "--branch" not in clone
    assert fetch[-1] == commit
    assert checkout[-1] == commit
    assert "--detach" in checkout
    assert result.resolved_commit_sha == commit
    service.cleanup(result)


def test_successful_github_clone_without_python_files_is_not_acquisition_failure() -> None:
    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "README").write_text("public fixture\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "e" * 40, "")

    service = SourceAcquisitionService(
        subprocess_runner=runner,
        git_executable="git",
    )
    result = service.resolve(
        _request(
            SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
            "https://github.com/owner/repo",
        )
    )

    assert result.status is SourceAcquisitionStatus.PARTIAL
    assert result.resolved_commit_sha == "e" * 40
    assert result.python_file_count == 0
    assert SourceIssueCategory.NO_PYTHON_FILES in _categories(result)
    assert result.is_available is True
    assert service.cleanup(result) is True


def test_static_discovery_retains_safe_top_level_function_names(tmp_path: Path) -> None:
    source = tmp_path / "inventory.py"
    source.write_text(
        "def calculate_total(price, quantity):\n    return price * quantity\n\n"
        "async def classify_stock(stock):\n    return stock\n",
        encoding="utf-8",
    )

    result = SourceAcquisitionService().resolve(
        _request(SourceTargetKind.LOCAL_FILE, source)
    )

    assert result.discovered_modules[0].top_level_function_names == (
        "calculate_total",
        "classify_stock",
    )
    assert "return price" not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("error", "category"),
    ((subprocess.TimeoutExpired("git", 1), SourceIssueCategory.CLONE_TIMEOUT),),
)
def test_clone_timeout_cleans_partial_workspace(error: Exception, category: SourceIssueCategory) -> None:
    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise error
    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    result = service.resolve(_request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo"))
    assert category in _categories(result)
    assert result.cleanup_required is False
    assert not result.resolved_project_root.exists()


def test_clone_failure_is_sanitized_and_cleaned() -> None:
    runner = Mock(return_value=subprocess.CompletedProcess(("git",), 128, "", "https://token@host failed"))
    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    result = service.resolve(_request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo"))
    serialized = json.dumps(result.to_dict())
    assert SourceIssueCategory.CLONE_FAILED in _categories(result)
    assert "token" not in serialized
    assert not result.resolved_project_root.exists()


def test_cleanup_is_owned_idempotent_and_never_deletes_user_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value=1\n", encoding="utf-8")
    service = SourceAcquisitionService()
    local = service.resolve(_request(SourceTargetKind.LOCAL_FILE, source))
    assert service.cleanup(local) is False
    assert source.exists()


def test_acquired_context_cleans_tool_workspace() -> None:
    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "c" * 40, "")

    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    with service.acquired(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo")
    ) as result:
        root = result.resolved_project_root
        assert root.exists()
        assert result.is_available is True
    assert not root.exists()
    assert result.is_available is False


def test_cleanup_rejects_broad_or_unowned_target(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value=1\n", encoding="utf-8")
    result = SourceAcquisitionService().resolve(_request(SourceTargetKind.LOCAL_FILE, source))
    forged = object.__new__(type(result))
    for field in result.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(result, field))
    object.__setattr__(forged, "workspace_ownership", SourceWorkspaceOwnership.TOOL_TEMPORARY)
    object.__setattr__(forged, "cleanup_required", True)
    with pytest.raises(RuntimeError, match="owned"):
        SourceAcquisitionService().cleanup(forged)
    assert tmp_path.exists()


def test_cleanup_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "d" * 40, "")
    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    result = service.resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo")
    )
    original = shutil.rmtree

    def fail(path: Path, *args: object, **kwargs: object) -> None:
        raise OSError("locked")

    monkeypatch.setattr("services.source_acquisition_service.shutil.rmtree", fail)
    with pytest.raises(SourceCleanupError) as captured:
        service.cleanup(result)
    assert captured.value.category is SourceIssueCategory.CLEANUP_FAILED
    monkeypatch.setattr("services.source_acquisition_service.shutil.rmtree", original)
    assert service.cleanup(result) is True


def test_cleanup_removes_read_only_git_pack_file() -> None:
    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "f" * 40, "")

    service = SourceAcquisitionService(
        subprocess_runner=runner,
        git_executable="git",
    )
    result = service.resolve(
        _request(
            SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
            "https://github.com/owner/repo",
        )
    )
    pack_file = result.resolved_project_root / ".git" / "objects" / "pack" / "fixture.idx"
    pack_file.parent.mkdir(parents=True)
    pack_file.write_bytes(b"fixture")
    pack_file.chmod(stat.S_IREAD)

    try:
        assert service.cleanup(result) is True
        assert not result.resolved_project_root.parent.exists()
    finally:
        if pack_file.exists():
            pack_file.chmod(stat.S_IWRITE)
            shutil.rmtree(result.resolved_project_root.parent, ignore_errors=True)


def test_two_resolutions_and_parallel_workspaces_are_isolated() -> None:
    def runner(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(arguments[-1])
        if "clone" in arguments:
            destination.mkdir(parents=True)
            (destination / "module.py").write_text("value=1\n", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "b" * 40, "")
    service = SourceAcquisitionService(subprocess_runner=runner, git_executable="git")
    request = _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(service.resolve, (request, request)))
    assert first.resolved_project_root != second.resolved_project_root
    assert service.cleanup(first) is True
    assert second.resolved_project_root.exists()
    assert service.cleanup(first) is False
    assert service.cleanup(second) is True


def test_unexpected_runner_exception_propagates() -> None:
    runner = Mock(side_effect=RuntimeError("unexpected"))
    with pytest.raises(RuntimeError, match="unexpected"):
        SourceAcquisitionService(subprocess_runner=runner, git_executable="git").resolve(
            _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, "https://github.com/owner/repo")
        )


def test_atomic_inventory_report_contains_no_raw_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    secret_source = "secret_literal = 'do-not-serialize'\n"
    source.write_text(secret_source, encoding="utf-8")
    result = SourceAcquisitionService().resolve(
        _request(SourceTargetKind.LOCAL_FILE, source)
    )
    report = SourceAcquisitionReportWriter.write(
        result, tmp_path / "reports" / "source_inventory.json"
    )
    payload = report.read_text(encoding="utf-8")
    assert "do-not-serialize" not in payload
    assert not tuple(report.parent.glob(".*.tmp"))
    assert json.loads(payload)["python_file_count"] == 1


def test_real_temporary_git_repository_commit_resolution(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git executable is unavailable.")
    repository = tmp_path / "repository"
    repository.mkdir()
    commands = (
        (git, "init", str(repository)),
        (git, "-C", str(repository), "config", "user.email", "fixture@example.invalid"),
        (git, "-C", str(repository), "config", "user.name", "Fixture"),
    )
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, shell=False)
    (repository / "module.py").write_text("value=1\n", encoding="utf-8")
    subprocess.run((git, "-C", str(repository), "add", "--", "module.py"), check=True, capture_output=True, shell=False)
    subprocess.run((git, "-C", str(repository), "commit", "-m", "fixture"), check=True, capture_output=True, shell=False)

    sha = GitSubprocessClient(git).resolve_commit_sha(repository, 10.0)

    assert len(sha) == 40
    assert all(character in "0123456789abcdef" for character in sha)


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_windows_drive_path_is_not_accepted_as_github_origin() -> None:
    result = SourceAcquisitionService(subprocess_runner=Mock()).resolve(
        _request(SourceTargetKind.PUBLIC_GITHUB_REPOSITORY, r"C:\\repository")
    )
    assert SourceIssueCategory.INVALID_GITHUB_URL in _categories(result)


def test_project_deadline_is_recomputed_before_each_git_subprocess() -> None:
    class Clock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    calls: list[tuple[str, ...]] = []

    def runner(arguments: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        destination = Path(arguments[-1])
        destination.mkdir(parents=True)
        clock.value = 2.0
        return subprocess.CompletedProcess(arguments, 0, "", "")

    deadline = ProjectDeadline.start(1.0, clock=clock)
    service = SourceAcquisitionService(
        subprocess_runner=runner,
        git_executable="git",
        clock=clock,
    )

    result = service.resolve(
        _request(
            SourceTargetKind.PUBLIC_GITHUB_REPOSITORY,
            "https://github.com/owner/repository",
        ),
        project_deadline=deadline,
    )

    assert len(calls) == 1
    assert SourceIssueCategory.CLONE_TIMEOUT in _categories(result)
    assert result.cleanup_required is False
    assert not result.resolved_project_root.exists()
