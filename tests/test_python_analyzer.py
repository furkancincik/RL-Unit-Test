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
        "örnek içerik",
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