from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer


def test_analyze_file_returns_correct_counts() -> None:
    analyzer = PythonAnalyzer()

    sample_file = Path("datasets/sample_code.py")
    result = analyzer.analyze_file(sample_file)

    assert result.file_name == "sample_code.py"
    assert result.function_count == 1
    assert result.class_count == 0
    assert result.if_count == 2
    assert result.for_count == 0
    assert result.while_count == 0
    assert result.try_count == 0
    assert result.return_count == 3

    assert len(result.functions) == 1

    function = result.functions[0]

    assert function.name == "calculate_score"
    assert function.parameters == ["score"]
    assert function.line_number == 1
    assert function.is_async is False
    assert function.return_count == 3
    assert function.branch_count == 2
    assert function.cyclomatic_complexity == 3
    assert function.risk_level == "Low"


def test_function_discovery_preserves_source_order_and_scope_metadata(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "targets.py"
    source_file.write_text(
        """
def first(value: int) -> int:
    return value

async def asynchronous() -> None:
    return None

def outer() -> int:
    def nested() -> int:
        return 1
    return nested()

class Handler:
    def method(self) -> int:
        return 1

def last(*, enabled: bool) -> bool:
    return enabled
""",
        encoding="utf-8",
    )

    functions = PythonAnalyzer().analyze_file(source_file).functions

    assert [function.qualified_name for function in functions] == [
        "first",
        "asynchronous",
        "outer",
        "outer.nested",
        "Handler.method",
        "last",
    ]
    assert functions[0].return_annotation == "int"
    assert functions[0].is_supported is True
    assert functions[1].is_async is True
    assert functions[1].is_supported is False
    assert functions[3].is_nested is True
    assert functions[3].is_supported is False
    assert functions[4].is_method is True
    assert functions[4].is_supported is False
    assert functions[5].parameters == ["enabled"]
    assert functions[5].is_supported is True


def test_analyze_file_raises_error_when_file_does_not_exist() -> None:
    analyzer = PythonAnalyzer()

    with pytest.raises(FileNotFoundError):
        analyzer.analyze_file("datasets/olmayan_dosya.py")


def test_analyze_file_rejects_non_python_files(
    tmp_path: Path,
) -> None:
    analyzer = PythonAnalyzer()

    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "Ã¶rnek iÃ§erik",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        analyzer.analyze_file(text_file)


def test_analyze_file_raises_error_for_invalid_python_syntax(
    tmp_path: Path,
) -> None:
    analyzer = PythonAnalyzer()

    invalid_file = tmp_path / "invalid_code.py"
    invalid_file.write_text(
        "def broken_function(\n",
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError):
        analyzer.analyze_file(invalid_file)


@pytest.mark.parametrize(
    ("complexity", "expected_risk"),
    [
        (1, "Low"),
        (5, "Low"),
        (6, "Medium"),
        (10, "Medium"),
        (11, "High"),
        (20, "High"),
    ],
)
def test_determine_risk_level(
    complexity: int,
    expected_risk: str,
) -> None:
    result = PythonAnalyzer._determine_risk_level(complexity)

    assert result == expected_risk

