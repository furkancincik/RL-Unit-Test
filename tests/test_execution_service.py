from pathlib import Path

import pytest

from services.execution_service import (
    ExecutionResult,
    ExecutionService,
    ExecutionTimeoutError,
)


def create_test_file(
    directory: Path,
    source_code: str,
    file_name: str = "test_generated.py",
) -> Path:
    """
    ExecutionService testleri iÃ§in geÃ§ici pytest dosyasÄ± oluÅŸturur.

    Args:
        directory: DosyanÄ±n oluÅŸturulacaÄŸÄ± klasÃ¶r.
        source_code: Test dosyasÄ±nÄ±n Python kaynak kodu.
        file_name: OluÅŸturulacak dosyanÄ±n adÄ±.

    Returns:
        OluÅŸturulan pytest dosyasÄ±nÄ±n yolu.
    """
    test_file = directory / file_name
    test_file.write_text(
        source_code,
        encoding="utf-8",
    )

    return test_file


def test_execute_returns_successful_result(
    tmp_path: Path,
) -> None:
    """BaÅŸarÄ±lÄ± pytest dosyasÄ±nÄ±n doÄŸru sonuÃ§ dÃ¶ndÃ¼rdÃ¼ÄŸÃ¼nÃ¼ doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_success() -> None:\n"
            "    assert 2 + 2 == 4\n"
        ),
    )

    result = service.execute(test_file)

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.exit_code == 0
    assert result.passed_count == 1
    assert result.failed_count == 0
    assert result.error_count == 0
    assert result.total_test_count == 1


def test_execute_returns_failed_result(
    tmp_path: Path,
) -> None:
    """BaÅŸarÄ±sÄ±z pytest testinin sonuÃ§ bilgilerinin Ã§Ä±karÄ±ldÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_failure() -> None:\n"
            "    assert 2 + 2 == 5\n"
        ),
    )

    result = service.execute(test_file)

    assert result.success is False
    assert result.exit_code != 0
    assert result.passed_count == 0
    assert result.failed_count == 1
    assert result.total_test_count == 1


def test_execute_counts_multiple_test_results(
    tmp_path: Path,
) -> None:
    """Birden fazla pytest sonucunun doÄŸru sayÄ±ldÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_first_success() -> None:\n"
            "    assert True\n"
            "\n"
            "\n"
            "def test_second_success() -> None:\n"
            "    assert 10 > 5\n"
            "\n"
            "\n"
            "def test_failure() -> None:\n"
            "    assert False\n"
        ),
    )

    result = service.execute(test_file)

    assert result.success is False
    assert result.passed_count == 2
    assert result.failed_count == 1
    assert result.total_test_count == 3


def test_execute_captures_pytest_output(
    tmp_path: Path,
) -> None:
    """Pytest standart Ã§Ä±ktÄ±sÄ±nÄ±n sonuÃ§ nesnesinde saklandÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_output() -> None:\n"
            "    assert True\n"
        ),
    )

    result = service.execute(test_file)

    assert "1 passed" in result.stdout
    assert result.stderr == ""
    assert result.duration_seconds >= 0
    assert str(test_file) in result.command


def test_execute_supports_extra_pytest_arguments(
    tmp_path: Path,
) -> None:
    """Ek pytest argÃ¼manlarÄ±nÄ±n komuta eklendiÄŸini doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_sample() -> None:\n"
            "    assert True\n"
        ),
    )

    result = service.execute(
        test_file,
        extra_args=("--disable-warnings",),
    )

    assert "--disable-warnings" in result.command
    assert result.success is True


def test_execute_rejects_missing_test_file(
    tmp_path: Path,
) -> None:
    """Bulunamayan test dosyasÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    service = ExecutionService()

    missing_file = tmp_path / "test_missing.py"

    with pytest.raises(
        FileNotFoundError,
        match="Test dosyasÄ± bulunamadÄ±",
    ):
        service.execute(missing_file)


def test_execute_rejects_non_python_file(
    tmp_path: Path,
) -> None:
    """Python dÄ±ÅŸÄ±ndaki test dosyalarÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    service = ExecutionService()

    text_file = tmp_path / "test_sample.txt"
    text_file.write_text(
        "sample",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Test dosyasÄ±nÄ±n uzantÄ±sÄ± \.py olmalÄ±dÄ±r",
    ):
        service.execute(text_file)


def test_execute_rejects_directory_path(
    tmp_path: Path,
) -> None:
    """Dosya yerine klasÃ¶r verilmesinin reddedildiÄŸini doÄŸrular."""
    service = ExecutionService()

    directory = tmp_path / "test_directory.py"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="Test dosyasÄ± yolu bir dosya olmalÄ±dÄ±r",
    ):
        service.execute(directory)


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_execute_rejects_invalid_timeout(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    """SÄ±fÄ±r ve negatif zaman aÅŸÄ±mÄ± deÄŸerlerinin reddedildiÄŸini doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_sample() -> None:\n"
            "    assert True\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Zaman aÅŸÄ±mÄ± deÄŸeri sÄ±fÄ±rdan bÃ¼yÃ¼k olmalÄ±dÄ±r",
    ):
        service.execute(
            test_file,
            timeout_seconds=timeout_seconds,
        )


def test_execute_raises_timeout_error(
    tmp_path: Path,
) -> None:
    """Uzun sÃ¼ren pytest iÅŸleminin zaman aÅŸÄ±mÄ±na uÄŸradÄ±ÄŸÄ±nÄ± doÄŸrular."""
    service = ExecutionService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "import time\n"
            "\n"
            "\n"
            "def test_slow_operation() -> None:\n"
            "    time.sleep(2)\n"
            "    assert True\n"
        ),
    )

    with pytest.raises(
        ExecutionTimeoutError,
        match="Pytest Ã§alÄ±ÅŸtÄ±rma iÅŸlemi zaman aÅŸÄ±mÄ±na uÄŸradÄ±",
    ):
        service.execute(
            test_file,
            timeout_seconds=0.1,
        )

