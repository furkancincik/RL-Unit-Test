from __future__ import annotations

import ast
import codecs
import json
import tokenize
from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from analyzer.python_source_reader import (
    PythonSourceEncodingError,
    read_python_source,
)
from cfg.control_flow_graph import ControlFlowGraphBuilder
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalAnalysisStatus,
    ExternalExecutionPolicy,
    ExternalSourceAnalysisRequest,
    UploadedPythonFile,
)
from models.source_acquisition_result import (
    SourceAcquisitionRequest,
    SourceAcquisitionStatus,
    SourceIssueCategory,
    SourceTargetKind,
)
from services.external_source_analysis_service import ExternalSourceAnalysisService
from services.source_acquisition_service import SourceAcquisitionService


def _acquire(source_file: Path):
    return SourceAcquisitionService().resolve(
        SourceAcquisitionRequest(
            source_kind=SourceTargetKind.LOCAL_FILE,
            origin=str(source_file),
        )
    )


def _categories(result: object) -> set[SourceIssueCategory]:
    return {issue.category for issue in result.issues}  # type: ignore[attr-defined]


def test_utf8_bom_has_equivalent_acquisition_analyzer_and_ast_results(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "renamed_bom_source.py"
    source_file.write_bytes(
        codecs.BOM_UTF8
        + b"def renamed_target(value: int) -> int:\n    return value + 1\n"
    )

    acquired = _acquire(source_file)
    analyzed = PythonAnalyzer().analyze_file(source_file)
    with tokenize.open(source_file) as stream:
        parsed = ast.parse(stream.read())

    assert acquired.status is SourceAcquisitionStatus.COMPLETED
    assert acquired.discovered_modules[0].encoding == "utf-8-sig"
    assert acquired.discovered_modules[0].top_level_function_names == (
        "renamed_target",
    )
    assert [item.name for item in analyzed.functions] == ["renamed_target"]
    assert [node.name for node in parsed.body if isinstance(node, ast.FunctionDef)] == [
        "renamed_target"
    ]


def test_plain_utf8_source_remains_supported(tmp_path: Path) -> None:
    source_file = tmp_path / "plain_source.py"
    source_file.write_bytes(
        "def plain_target(label: str) -> str:\n    return label\n".encode("utf-8")
    )

    acquired = _acquire(source_file)
    analyzed = PythonAnalyzer().analyze_file(source_file)

    assert acquired.status is SourceAcquisitionStatus.COMPLETED
    assert analyzed.functions[0].name == "plain_target"


def test_pep263_non_utf8_cookie_is_used_by_analyzer(tmp_path: Path) -> None:
    source_file = tmp_path / "legacy_encoded.py"
    source_file.write_bytes(
        "# -*- coding: cp1254 -*-\ndef localized() -> str:\n    return 'şube'\n".encode(
            "cp1254"
        )
    )

    acquired = _acquire(source_file)
    analyzed = PythonAnalyzer().analyze_file(source_file)

    assert acquired.status is SourceAcquisitionStatus.COMPLETED
    assert acquired.discovered_modules[0].encoding == "cp1254"
    assert analyzed.functions[0].name == "localized"


def test_crlf_and_lf_preserve_equivalent_line_metadata(tmp_path: Path) -> None:
    lf_file = tmp_path / "line_feed.py"
    crlf_file = tmp_path / "carriage_return.py"
    source = "# heading\ndef line_target(value: int) -> int:\n    if value:\n        return 1\n    return 0\n"
    lf_file.write_bytes(source.encode("utf-8"))
    crlf_file.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))

    lf_result = PythonAnalyzer().analyze_file(lf_file).functions[0]
    crlf_result = PythonAnalyzer().analyze_file(crlf_file).functions[0]
    lf_graph = ControlFlowGraphBuilder().build_from_file(lf_file)[0]
    crlf_graph = ControlFlowGraphBuilder().build_from_file(crlf_file)[0]

    assert (lf_result.line_number, lf_result.end_line_number) == (
        crlf_result.line_number,
        crlf_result.end_line_number,
    )
    assert [node.line_number for node in lf_graph.nodes] == [
        node.line_number for node in crlf_graph.nodes
    ]


@pytest.mark.parametrize(
    "payload",
    (
        codecs.BOM_UTF8 + b"# coding: latin-1\ndef target():\n    return 1\n",
        b"# coding: codec-that-does-not-exist\ndef target():\n    return 1\n",
        b"# coding: utf-8\ndef target():\n    return '\xff'\n",
    ),
)
def test_invalid_source_encoding_is_controlled(
    tmp_path: Path,
    payload: bytes,
) -> None:
    source_file = tmp_path / "invalid_encoding.py"
    source_file.write_bytes(payload)

    acquired = _acquire(source_file)

    with pytest.raises(PythonSourceEncodingError, match="encoding"):
        PythonAnalyzer().analyze_file(source_file)
    assert acquired.status is SourceAcquisitionStatus.PARTIAL
    assert SourceIssueCategory.UNSUPPORTED_ENCODING in _categories(acquired)


def test_real_syntax_and_null_byte_remain_syntax_errors(tmp_path: Path) -> None:
    for name, payload in (
        ("broken_syntax.py", b"def broken(:\n"),
        ("null_byte.py", b"def target():\n    return 1\x00\n"),
    ):
        source_file = tmp_path / name
        source_file.write_bytes(payload)

        acquired = _acquire(source_file)

        with pytest.raises(SyntaxError):
            PythonAnalyzer().analyze_file(source_file)
        assert SourceIssueCategory.SYNTAX_ERROR in _categories(acquired)


def test_bom_source_uses_same_analyzer_cfg_and_external_static_semantics(
    tmp_path: Path,
) -> None:
    source = (
        codecs.BOM_UTF8
        + b"def parity_target(flag: bool) -> int:\n"
        b"    if flag:\n        return 7\n"
        b"    return 3\n"
    )
    source_file = tmp_path / "direct_parity.py"
    source_file.write_bytes(source)

    analyzed = PythonAnalyzer().analyze_file(source_file)
    graphs = ControlFlowGraphBuilder().build_from_file(source_file)
    external = ExternalSourceAnalysisService().run(
        ExternalSourceAnalysisRequest(
            UploadedPythonFile("uploaded_parity.py", source),
            ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
            ExternalAnalysisConfiguration(output_root=tmp_path / "output"),
        )
    )

    assert analyzed.functions[0].name == "parity_target"
    assert graphs[0].function_name == "parity_target"
    assert external.status is ExternalAnalysisStatus.STATIC_COMPLETED
    assert external.module_results[0].discovered_function_names == (
        "parity_target",
    )
    serialized = json.dumps(external.to_dict())
    assert "def parity_target" not in serialized
    assert source.hex() not in serialized


@pytest.mark.parametrize("error_type", (AssertionError, TypeError, RuntimeError))
def test_unexpected_source_reader_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    source_file = tmp_path / "propagation.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    def fail(_path: object) -> object:
        raise error_type("unexpected reader failure")

    monkeypatch.setattr("analyzer.python_source_reader.tokenize.open", fail)

    with pytest.raises(error_type, match="unexpected reader failure"):
        read_python_source(source_file)
