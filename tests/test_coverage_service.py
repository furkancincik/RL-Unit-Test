from pathlib import Path
from typing import Final

import pytest

from models.coverage_result import CoverageResult
from services.coverage_service import (
    CoverageExecutionTimeoutError,
    CoverageService,
)


SOURCE_FILE: Final[Path] = Path(
    "datasets/sample_code.py"
)


def create_test_file(
    directory: Path,
    source_code: str,
    file_name: str = "test_coverage_target.py",
) -> Path:
    """Coverage testlerinde kullanılacak pytest dosyasını oluşturur."""
    test_file = directory / file_name
    test_file.write_text(
        source_code,
        encoding="utf-8",
    )

    return test_file


def test_measure_returns_coverage_result(
    tmp_path: Path,
) -> None:
    """Coverage ölçümünün sonuç modeli döndürdüğünü doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "from datasets.sample_code import calculate_score\n"
            "\n"
            "\n"
            "def test_high_score() -> None:\n"
            '    assert calculate_score(90) == "Başarılı"\n'
        ),
    )

    result = service.measure(
        source_file=SOURCE_FILE,
        test_file=test_file,
    )

    assert isinstance(result, CoverageResult)
    assert result.source_file == SOURCE_FILE.resolve()
    assert result.test_file == test_file.resolve()


def test_measure_reports_full_coverage(
    tmp_path: Path,
) -> None:
    """Bütün karar yollarını kullanan testlerin tam coverage sağladığını doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "from datasets.sample_code import calculate_score\n"
            "\n"
            "\n"
            "def test_high_score() -> None:\n"
            '    assert calculate_score(90) == "Başarılı"\n'
            "\n"
            "\n"
            "def test_medium_score() -> None:\n"
            '    assert calculate_score(60) == "Orta"\n'
            "\n"
            "\n"
            "def test_low_score() -> None:\n"
            '    assert calculate_score(20) == "Başarısız"\n'
        ),
    )

    result = service.measure(
        source_file=SOURCE_FILE,
        test_file=test_file,
    )

    assert result.success is True
    assert result.line_coverage_percent == 100.0
    assert result.branch_coverage_percent == 100.0
    assert result.missing_line_count == 0
    assert result.missing_branch_count == 0
    assert result.has_full_coverage is True


def test_measure_reports_partial_coverage(
    tmp_path: Path,
) -> None:
    """Tek karar yolunu kullanan testin kısmi coverage verdiğini doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "from datasets.sample_code import calculate_score\n"
            "\n"
            "\n"
            "def test_high_score() -> None:\n"
            '    assert calculate_score(90) == "Başarılı"\n'
        ),
    )

    result = service.measure(
        source_file=SOURCE_FILE,
        test_file=test_file,
    )

    assert result.success is True
    assert result.line_coverage_percent < 100.0
    assert result.branch_coverage_percent < 100.0
    assert result.missing_line_count > 0
    assert result.missing_branch_count > 0
    assert result.has_full_coverage is False


def test_measure_preserves_failed_test_exit_code(
    tmp_path: Path,
) -> None:
    """Başarısız pytest işleminin çıkış kodunun korunduğunu doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "from datasets.sample_code import calculate_score\n"
            "\n"
            "\n"
            "def test_failure() -> None:\n"
            '    assert calculate_score(90) == "Yanlış"\n'
        ),
    )

    result = service.measure(
        source_file=SOURCE_FILE,
        test_file=test_file,
    )

    assert result.test_exit_code != 0
    assert result.success is False
    assert result.total_line_count > 0


def test_measure_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    """Bulunamayan kaynak dosyanın reddedildiğini doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code="def test_sample() -> None:\n    assert True\n",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Kaynak dosya bulunamadı",
    ):
        service.measure(
            source_file=tmp_path / "missing.py",
            test_file=test_file,
        )


def test_measure_rejects_missing_test_file(
    tmp_path: Path,
) -> None:
    """Bulunamayan pytest dosyasının reddedildiğini doğrular."""
    service = CoverageService()

    with pytest.raises(
        FileNotFoundError,
        match="Test dosyası bulunamadı",
    ):
        service.measure(
            source_file=SOURCE_FILE,
            test_file=tmp_path / "test_missing.py",
        )


@pytest.mark.parametrize(
    ("file_name", "field_name"),
    [
        ("source.txt", "Kaynak dosya"),
        ("test_sample.json", "Test dosyası"),
    ],
)
def test_measure_rejects_non_python_files(
    tmp_path: Path,
    file_name: str,
    field_name: str,
) -> None:
    """Python dışındaki kaynak ve test dosyalarının reddedildiğini doğrular."""
    service = CoverageService()

    non_python_file = tmp_path / file_name
    non_python_file.write_text(
        "sample",
        encoding="utf-8",
    )

    if field_name == "Kaynak dosya":
        source_file = non_python_file
        test_file = create_test_file(
            directory=tmp_path,
            source_code=(
                "def test_sample() -> None:\n"
                "    assert True\n"
            ),
        )
    else:
        source_file = SOURCE_FILE
        test_file = non_python_file

    with pytest.raises(
        ValueError,
        match=rf"{field_name} uzantısı \.py olmalıdır",
    ):
        service.measure(
            source_file=source_file,
            test_file=test_file,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_measure_rejects_invalid_timeout(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    """Sıfır ve negatif timeout değerlerinin reddedildiğini doğrular."""
    service = CoverageService()

    test_file = create_test_file(
        directory=tmp_path,
        source_code=(
            "def test_sample() -> None:\n"
            "    assert True\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Coverage zaman aşımı sıfırdan büyük olmalıdır",
    ):
        service.measure(
            source_file=SOURCE_FILE,
            test_file=test_file,
            timeout_seconds=timeout_seconds,
        )


def test_measure_raises_timeout_error(
    tmp_path: Path,
) -> None:
    """Uzun süren coverage işleminin zaman aşımını doğrular."""
    service = CoverageService()

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
        CoverageExecutionTimeoutError,
        match="Coverage ölçüm işlemi zaman aşımına uğradı",
    ):
        service.measure(
            source_file=SOURCE_FILE,
            test_file=test_file,
            timeout_seconds=0.1,
        )