from __future__ import annotations

import ast
from pathlib import Path

import pytest

from services.generation_service import (
    GeneratedTestArtifact,
    GenerationService,
)

SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"


def test_generate_for_file_creates_test_artifact(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    assert len(artifacts) == 1

    artifact = artifacts[0]

    assert isinstance(
        artifact,
        GeneratedTestArtifact,
    )
    assert artifact.function_name == "calculate_score"
    assert artifact.scenario_count == 3
    assert artifact.output_path.exists()


def test_generate_for_file_creates_expected_file_name(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    assert artifacts[0].output_path == (
        tmp_path / "test_calculate_score.py"
    )


def test_generated_file_contains_target_import(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    assert (
        "from datasets.sample_code "
        "import calculate_score"
    ) in generated_source


def test_generated_file_contains_three_real_tests(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    assert generated_source.count(
        "def test_calculate_score_"
    ) == 3

    assert generated_source.count(
        "result = calculate_score("
    ) == 3


def test_generated_file_does_not_contain_placeholder_test(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    assert "assert callable" not in generated_source
    assert "_target_function" not in generated_source
    assert "TODO" not in generated_source


def test_generated_file_contains_all_expected_results(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    assert (
        "assert result == 'Başarılı'"
        in generated_source
    )
    assert (
        "assert result == 'Orta'"
        in generated_source
    )
    assert (
        "assert result == 'Başarısız'"
        in generated_source
    )


def test_generated_file_uses_inputs_matching_paths(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    syntax_tree = ast.parse(generated_source)

    generated_scores: list[int | float] = []

    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        if node.func.id != "calculate_score":
            continue

        for keyword_argument in node.keywords:
            if keyword_argument.arg != "score":
                continue

            value = ast.literal_eval(
                keyword_argument.value
            )

            generated_scores.append(value)

    assert len(generated_scores) == 3

    assert any(
        score >= 85
        for score in generated_scores
    )

    assert any(
        50 <= score < 85
        for score in generated_scores
    )

    assert any(
        score < 50
        for score in generated_scores
    )


def test_generated_file_has_valid_python_syntax(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
        overwrite=True,
    )

    generated_source = artifacts[
        0
    ].output_path.read_text(
        encoding="utf-8",
    )

    ast.parse(generated_source)


def test_generate_for_file_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    missing_file = (
        tmp_path / "missing_source.py"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        service.generate_for_file(
            source_file=missing_file,
            module_path=MODULE_PATH,
            output_directory=tmp_path,
            overwrite=True,
        )


def test_generate_for_file_rejects_non_python_file(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    invalid_file = tmp_path / "source.txt"

    invalid_file.write_text(
        "not a Python file",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="uzantısı .py olmalıdır",
    ):
        service.generate_for_file(
            source_file=invalid_file,
            module_path=MODULE_PATH,
            output_directory=tmp_path,
            overwrite=True,
        )


def test_generate_for_file_rejects_invalid_module_path(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        service.generate_for_file(
            source_file=SOURCE_FILE,
            module_path="datasets/sample_code",
            output_directory=tmp_path,
            overwrite=True,
        )


def test_generate_for_file_rejects_empty_module_path(
    tmp_path: Path,
) -> None:
    service = GenerationService()

    with pytest.raises(
        ValueError,
        match="Modül yolu boş olamaz",
    ):
        service.generate_for_file(
            source_file=SOURCE_FILE,
            module_path=" ",
            output_directory=tmp_path,
            overwrite=True,
        )


def test_generate_for_file_rejects_empty_output_directory() -> None:
    service = GenerationService()

    with pytest.raises(
        ValueError,
        match="Çıktı klasörü boş olamaz",
    ):
        service.generate_for_file(
            source_file=SOURCE_FILE,
            module_path=MODULE_PATH,
            output_directory=" ",
            overwrite=True,
        )