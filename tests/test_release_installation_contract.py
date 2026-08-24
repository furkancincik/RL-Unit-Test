from __future__ import annotations

from pathlib import Path


def _requirements() -> dict[str, str]:
    entries = {}
    for raw_line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", maxsplit=1)[0].casefold()
        entries[name] = line
    return entries


def test_clean_install_includes_production_http_and_execution_dependencies() -> None:
    requirements = _requirements()

    assert "httpx" in requirements
    assert "httpx2" not in requirements
    assert "pytest" in requirements
    assert "coverage" in requirements


def test_runtime_report_package_is_not_excluded_from_release_archive() -> None:
    ignored_patterns = {
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "reports/" not in ignored_patterns
    assert Path("reports/__init__.py").is_file()
    assert Path("reports/json_reporter.py").is_file()
