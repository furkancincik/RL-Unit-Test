from __future__ import annotations

import shutil
import stat
from collections.abc import Callable
from pathlib import Path


def is_link_like(path: Path) -> bool:
    """Symlink, junction ve Windows reparse-point girişlerini tanır."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    except FileNotFoundError:
        return False
    except OSError:
        return True


def remove_workspace_tree(
    workspace: Path,
    *,
    rmtree: Callable[..., None] = shutil.rmtree,
) -> None:
    """Doğrulanmış workspace'i containment korumalı chmod callback'iyle siler."""
    resolved_workspace = workspace.resolve()

    def make_writable_and_retry(
        function: Callable[[str], object],
        path: str,
        error_info: tuple[type[BaseException], BaseException, object],
    ) -> None:
        error = error_info[1]
        candidate = Path(path)
        try:
            resolved_candidate = candidate.resolve(strict=False)
        except OSError:
            raise error
        if (
            not resolved_candidate.is_relative_to(resolved_workspace)
            or is_link_like(candidate)
        ):
            raise error
        try:
            candidate.chmod(stat.S_IWRITE)
            function(path)
        except OSError:
            raise error

    rmtree(resolved_workspace, onerror=make_writable_and_retry)
