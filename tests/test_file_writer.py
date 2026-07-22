from pathlib import Path

import pytest

from generator.file_writer import GeneratedTestFileWriter


VALID_SOURCE_CODE = (
    '"""Otomatik oluÅŸturulmuÅŸ test dosyasÄ±."""\n'
    "\n"
    "\n"
    "def test_example() -> None:\n"
    "    assert True\n"
)


def test_write_creates_python_file(
    tmp_path: Path,
) -> None:
    """GeÃ§erli kaynak kodun Python dosyasÄ±na yazÄ±ldÄ±ÄŸÄ±nÄ± doÄŸrular."""
    writer = GeneratedTestFileWriter()
    output_path = tmp_path / "test_generated.py"

    result_path = writer.write(
        source_code=VALID_SOURCE_CODE,
        output_path=output_path,
    )

    assert result_path == output_path
    assert result_path.exists()
    assert result_path.is_file()


def test_write_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """Eksik Ã¼st klasÃ¶rlerin otomatik oluÅŸturulduÄŸunu doÄŸrular."""
    writer = GeneratedTestFileWriter()

    output_path = (
        tmp_path
        / "tests"
        / "generated"
        / "test_example.py"
    )

    result_path = writer.write(
        source_code=VALID_SOURCE_CODE,
        output_path=output_path,
    )

    assert result_path.exists()
    assert result_path.parent.exists()


def test_write_preserves_utf8_content(
    tmp_path: Path,
) -> None:
    """TÃ¼rkÃ§e karakterlerin ve kaynak iÃ§eriÄŸin korunduÄŸunu doÄŸrular."""
    writer = GeneratedTestFileWriter()
    output_path = tmp_path / "test_turkish.py"

    source_code = (
        '"""TÃ¼rkÃ§e aÃ§Ä±klama: Ã§, ÄŸ, Ä±, Ã¶, ÅŸ, Ã¼."""\n'
        "\n"
        "\n"
        "def test_turkish_characters() -> None:\n"
        '    message = "BaÅŸarÄ±lÄ±"\n'
        '    assert message == "BaÅŸarÄ±lÄ±"\n'
    )

    writer.write(
        source_code=source_code,
        output_path=output_path,
    )

    saved_content = output_path.read_text(
        encoding="utf-8",
    )

    assert saved_content == source_code


@pytest.mark.parametrize(
    "source_code",
    [
        "",
        "   \n\t",
    ],
)
def test_write_rejects_empty_source_code(
    tmp_path: Path,
    source_code: str,
) -> None:
    """BoÅŸ veya yalnÄ±zca boÅŸluk iÃ§eren kaynak kodun reddedildiÄŸini doÄŸrular."""
    writer = GeneratedTestFileWriter()

    with pytest.raises(
        ValueError,
        match="Kaynak kod boÅŸ olamaz",
    ):
        writer.write(
            source_code=source_code,
            output_path=tmp_path / "test_empty.py",
        )


def test_write_rejects_invalid_python_syntax(
    tmp_path: Path,
) -> None:
    """GeÃ§ersiz Python sÃ¶zdiziminin dosyaya yazÄ±lmadÄ±ÄŸÄ±nÄ± doÄŸrular."""
    writer = GeneratedTestFileWriter()
    output_path = tmp_path / "test_invalid.py"

    with pytest.raises(
        ValueError,
        match=(
            "Kaynak kod geÃ§erli Python "
            "sÃ¶zdizimine sahip deÄŸil"
        ),
    ):
        writer.write(
            source_code="def broken_function(:\n    pass\n",
            output_path=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "file_name",
    [
        "generated_test.txt",
        "generated_test.json",
    ],
)
def test_write_rejects_non_python_extension(
    tmp_path: Path,
    file_name: str,
) -> None:
    """Python dÄ±ÅŸÄ±ndaki dosya uzantÄ±larÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    writer = GeneratedTestFileWriter()

    with pytest.raises(
        ValueError,
        match=r"Ã‡Ä±ktÄ± dosyasÄ±nÄ±n uzantÄ±sÄ± \.py olmalÄ±dÄ±r",
    ):
        writer.write(
            source_code=VALID_SOURCE_CODE,
            output_path=tmp_path / file_name,
        )


def test_write_rejects_existing_file_without_overwrite(
    tmp_path: Path,
) -> None:
    """Mevcut dosyanÄ±n varsayÄ±lan olarak korunmasÄ±nÄ± doÄŸrular."""
    writer = GeneratedTestFileWriter()
    output_path = tmp_path / "test_existing.py"

    output_path.write_text(
        "original_content = True\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Ã‡Ä±ktÄ± dosyasÄ± zaten mevcut",
    ):
        writer.write(
            source_code=VALID_SOURCE_CODE,
            output_path=output_path,
        )

    assert (
        output_path.read_text(encoding="utf-8")
        == "original_content = True\n"
    )


def test_write_overwrites_existing_file_when_enabled(
    tmp_path: Path,
) -> None:
    """Overwrite etkinleÅŸtirildiÄŸinde mevcut dosyanÄ±n gÃ¼ncellendiÄŸini doÄŸrular."""
    writer = GeneratedTestFileWriter()
    output_path = tmp_path / "test_existing.py"

    output_path.write_text(
        "original_content = True\n",
        encoding="utf-8",
    )

    result_path = writer.write(
        source_code=VALID_SOURCE_CODE,
        output_path=output_path,
        overwrite=True,
    )

    assert result_path == output_path
    assert (
        output_path.read_text(encoding="utf-8")
        == VALID_SOURCE_CODE
    )

