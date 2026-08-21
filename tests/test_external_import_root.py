from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from services.coverage_service import CoverageService


def test_coverage_subprocess_uses_only_validated_import_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project" / "module.py"
    source.parent.mkdir()
    source.write_text("value=1\n", encoding="utf-8")
    test_file = tmp_path / "test_module.py"
    test_file.write_text("def test_value():\n assert True\n", encoding="utf-8")
    completed = subprocess.CompletedProcess(("pytest",), 0, "", "")

    with patch("services.coverage_service.subprocess.run", return_value=completed) as runner:
        with patch.object(CoverageService, "_read_report", return_value={"files": {str(source): {"summary": {}, "executed_lines": [], "missing_lines": [], "executed_branches": [], "missing_branches": []}}}):
            try:
                CoverageService().measure(source, test_file, import_root=source.parent)
            except (KeyError, ValueError, RuntimeError):
                pass

    first = runner.call_args_list[0].kwargs
    assert first["cwd"] == source.parent.resolve()
    assert first["env"]["PYTHONPATH"] == str(source.parent.resolve())
    assert os.environ.get("PYTHONPATH") != first["env"]["PYTHONPATH"] or source.parent == Path(os.environ.get("PYTHONPATH", ""))
