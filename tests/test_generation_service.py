import ast
from pathlib import Path
from typing import Final

import pytest

from services.generation_service import (
    GeneratedTestArtifact,
    GenerationService,
)


SOURCE_FILE: Final[str] = "datasets/sample_code.py"
MODULE_PATH: Final[str] = "datasets.sample_code"


def test_generate_for_file_creates_pytest_file(
    tmp_path: Path,
) -> None:
    """Servisin gerÃ§ek bir pytest dosyasÄ± oluÅŸturduÄŸunu doÄŸrular."""
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].output_path.exists()
    assert artifacts[0].output_path.is_file()


def test_generate_for_file_returns_artifact_metadata(
    tmp_path: Path,
) -> None:
    """Ãœretim sonucunun doÄŸru metadata bilgilerini taÅŸÄ±dÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    artifact = artifacts[0]

    assert isinstance(artifact, GeneratedTestArtifact)
    assert artifact.function_name == "calculate_score"
    assert artifact.scenario_count == 3
    assert artifact.output_path == (
        tmp_path / "test_calculate_score.py"
    )


def test_generate_for_file_creates_valid_python_code(
    tmp_path: Path,
) -> None:
    """Servisin oluÅŸturduÄŸu dosyanÄ±n geÃ§erli Python kodu olduÄŸunu doÄŸrular."""
    service = GenerationService()

    artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=tmp_path,
    )

    generated_code = artifacts[0].output_path.read_text(
        encoding="utf-8"
    )

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)
    assert (
        "from datasets.sample_code import calculate_score"
        in generated_code
    )


def test_generate_for_file_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    """Bulunamayan kaynak dosyanÄ±n reddedildiÄŸini doÄŸrular."""
    service = GenerationService()

    missing_source = tmp_path / "missing_source.py"

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadÄ±",
    ):
        service.generate_for_file(
            source_file=missing_source,
            module_path=MODULE_PATH,
            output_directory=tmp_path / "generated",
        )


def test_generate_for_file_rejects_non_python_source(
    tmp_path: Path,
) -> None:
    """Python dÄ±ÅŸÄ±ndaki kaynak dosyalarÄ±n reddedildiÄŸini doÄŸrular."""
    service = GenerationService()

    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "sample content",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Kaynak dosyanÄ±n uzantÄ±sÄ± \.py olmalÄ±dÄ±r",
    ):
        service.generate_for_file(
            source_file=text_file,
            module_path=MODULE_PATH,
            output_directory=tmp_path / "generated",
        )


def test_generate_for_file_rejects_file_without_functions(
    tmp_path: Path,
) -> None:
    """Fonksiyon iÃ§ermeyen Python dosyasÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    service = GenerationService()

    source_file = tmp_path / "constants.py"
    source_file.write_text(
        "APPLICATION_NAME = 'RL Unit Test'\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="analiz edilebilir fonksiyon bulunamadÄ±",
    ):
        service.generate_for_file(
            source_file=source_file,
            module_path="constants",
            output_directory=tmp_path / "generated",
        )


def test_generate_for_file_protects_existing_output(
    tmp_path: Path,
) -> None:
    """Mevcut test dosyasÄ±nÄ±n varsayÄ±lan olarak korunmasÄ±nÄ± doÄŸrular."""
    service = GenerationService()
    output_directory = tmp_path / "generated"

    service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=output_directory,
    )

    with pytest.raises(
        FileExistsError,
        match="Ã‡Ä±ktÄ± dosyasÄ± zaten mevcut",
    ):
        service.generate_for_file(
            source_file=SOURCE_FILE,
            module_path=MODULE_PATH,
            output_directory=output_directory,
        )


def test_generate_for_file_overwrites_output_when_enabled(
    tmp_path: Path,
) -> None:
    """Overwrite etkinleÅŸtirildiÄŸinde mevcut testin yenilendiÄŸini doÄŸrular."""
    service = GenerationService()
    output_directory = tmp_path / "generated"

    first_artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=output_directory,
    )

    first_output = first_artifacts[0].output_path
    first_output.write_text(
        "temporary_content = True\n",
        encoding="utf-8",
    )

    second_artifacts = service.generate_for_file(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        output_directory=output_directory,
        overwrite=True,
    )

    regenerated_content = second_artifacts[0].output_path.read_text(
        encoding="utf-8"
    )

    assert "temporary_content" not in regenerated_content
    assert "def test_calculate_score_" in regenerated_content

