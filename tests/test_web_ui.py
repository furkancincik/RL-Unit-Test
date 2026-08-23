from __future__ import annotations

import re
import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.app import create_app
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def _client(tmp_path: Path) -> TestClient:
    service = Mock(spec=AnalysisJobService)
    service.settings = AnalysisJobSettings(output_root=tmp_path)
    service.purge_expired.return_value = 0
    service.capacity.return_value = (0, 0, 22)
    return TestClient(create_app(job_service=service))


def test_root_serves_local_production_web_ui(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "RL Unit Test" in response.text
    assert "Maksimum coverage, minimum test" in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'href="/static/styles.css"' in response.text
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", response.text, re.IGNORECASE)
    assert not re.search(
        r"(?:src|href)=[\"']https?://", response.text, re.IGNORECASE
    )


def test_static_assets_are_available_and_contained(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        css = client.get("/static/styles.css")
        javascript = client.get("/static/app.js")
        missing = client.get("/static/missing.js")
        traversal = client.get("/static/%2e%2e/README.md")

    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert javascript.status_code == 200
    assert "javascript" in javascript.headers["content-type"]
    assert missing.status_code == 404
    assert traversal.status_code == 404


def test_docs_openapi_and_security_headers_are_preserved(tmp_path: Path) -> None:
    expected_headers = {
        "content-security-policy",
        "x-content-type-options",
        "referrer-policy",
        "x-frame-options",
        "permissions-policy",
    }
    client = _client(tmp_path)
    with client:
        responses = [
            client.get("/"),
            client.get("/docs"),
            client.get("/redoc"),
            client.get("/openapi.json"),
            client.get("/api/v1/health"),
        ]

    assert all(response.status_code == 200 for response in responses)
    for response in responses:
        assert expected_headers <= set(response.headers)
        csp = response.headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "unsafe-eval" not in csp
    assert not any(
        middleware.cls.__name__ == "CORSMiddleware"
        for middleware in client.app.user_middleware
    )


def test_web_ui_contains_separate_sources_and_accessible_controls(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        html = client.get("/").text

    assert 'data-source-tab="inline"' in html
    assert 'data-source-tab="upload"' in html
    assert 'data-source-tab="github"' in html
    assert 'id="inline-source"' in html
    assert 'id="python-file"' in html
    assert 'id="github-url"' in html
    assert 'id="trusted-acknowledgement"' in html
    assert 'id="artifact-list"' in html
    assert 'id="result-description"' in html
    assert 'id="result-info"' in html
    assert 'aria-live="polite"' in html
    assert "Yerel proje" not in html
    assert "webkitdirectory" not in html
    assert re.search(r'<label[^>]+for="inline-source"', html)
    assert re.search(r'<label[^>]+for="github-url"', html)
    assert re.search(
        r'<input[^>]+id="greedy-minimization"[^>]+checked',
        html,
    )


def test_importing_server_does_not_start_uvicorn() -> None:
    sys.modules.pop("api.server", None)

    module = importlib.import_module("api.server")

    assert callable(module.main)
