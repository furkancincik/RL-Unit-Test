from pathlib import Path

import pytest

from models.coverage_result import CoverageResult


def create_coverage_result(
    *,
    line_coverage_percent: float = 100.0,
    branch_coverage_percent: float = 100.0,
    covered_line_count: int = 10,
    missing_line_count: int = 0,
    total_line_count: int = 10,
    covered_branch_count: int = 4,
    missing_branch_count: int = 0,
    total_branch_count: int = 4,
    test_exit_code: int = 0,
    duration_seconds: float = 0.25,
) -> CoverageResult:
    """
    CoverageResult testlerinde kullanılacak sonuç nesnesi oluşturur.

    Returns:
        Verilen alanlarla hazırlanmış CoverageResult nesnesi.
    """
    return CoverageResult(
        source_file=Path("datasets/sample_code.py"),
        test_file=Path(
            "output/generated_tests/test_calculate_score.py"
        ),
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=branch_coverage_percent,
        covered_line_count=covered_line_count,
        missing_line_count=missing_line_count,
        total_line_count=total_line_count,
        covered_branch_count=covered_branch_count,
        missing_branch_count=missing_branch_count,
        total_branch_count=total_branch_count,
        test_exit_code=test_exit_code,
        duration_seconds=duration_seconds,
    )


def test_coverage_result_reports_success() -> None:
    """Sıfır çıkış kodunun başarılı sonuç olarak değerlendirildiğini doğrular."""
    result = create_coverage_result()

    assert result.success is True


def test_coverage_result_reports_failed_execution() -> None:
    """Sıfır dışındaki çıkış kodunun başarısız sayıldığını doğrular."""
    result = create_coverage_result(
        test_exit_code=1,
    )

    assert result.success is False


def test_coverage_result_reports_full_coverage() -> None:
    """Tam satır ve branch coverage değerlerinin doğru raporlandığını doğrular."""
    result = create_coverage_result()

    assert result.has_full_line_coverage is True
    assert result.has_full_branch_coverage is True
    assert result.has_full_coverage is True


def test_coverage_result_reports_partial_coverage() -> None:
    """Eksik satır ve branch bulunan sonucun tam coverage olmadığını doğrular."""
    result = create_coverage_result(
        line_coverage_percent=80.0,
        branch_coverage_percent=75.0,
        covered_line_count=8,
        missing_line_count=2,
        total_line_count=10,
        covered_branch_count=3,
        missing_branch_count=1,
        total_branch_count=4,
    )

    assert result.has_full_line_coverage is False
    assert result.has_full_branch_coverage is False
    assert result.has_full_coverage is False


@pytest.mark.parametrize(
    "percentage",
    [
        -0.1,
        100.1,
        150.0,
    ],
)
def test_coverage_result_rejects_invalid_line_percentage(
    percentage: float,
) -> None:
    """0–100 aralığı dışındaki satır coverage değerlerini reddeder."""
    with pytest.raises(
        ValueError,
        match="Satır coverage yüzdesi 0 ile 100 arasında olmalıdır",
    ):
        create_coverage_result(
            line_coverage_percent=percentage,
        )


@pytest.mark.parametrize(
    "percentage",
    [
        -1.0,
        101.0,
    ],
)
def test_coverage_result_rejects_invalid_branch_percentage(
    percentage: float,
) -> None:
    """0–100 aralığı dışındaki branch coverage değerlerini reddeder."""
    with pytest.raises(
        ValueError,
        match="Branch coverage yüzdesi 0 ile 100 arasında olmalıdır",
    ):
        create_coverage_result(
            branch_coverage_percent=percentage,
        )


def test_coverage_result_rejects_negative_count() -> None:
    """Negatif coverage sayaçlarının reddedildiğini doğrular."""
    with pytest.raises(
        ValueError,
        match="Eksik satır sayısı negatif olamaz",
    ):
        create_coverage_result(
            covered_line_count=11,
            missing_line_count=-1,
        )


def test_coverage_result_rejects_inconsistent_line_counts() -> None:
    """Satır toplamları tutarsız olduğunda hata üretildiğini doğrular."""
    with pytest.raises(
        ValueError,
        match=(
            "Çalıştırılan ve eksik satırların toplamı, "
            "toplam satır sayısına eşit olmalıdır"
        ),
    ):
        create_coverage_result(
            covered_line_count=8,
            missing_line_count=1,
            total_line_count=10,
        )


def test_coverage_result_rejects_inconsistent_branch_counts() -> None:
    """Branch toplamları tutarsız olduğunda hata üretildiğini doğrular."""
    with pytest.raises(
        ValueError,
        match=(
            "Çalıştırılan ve eksik branchlerin toplamı, "
            "toplam branch sayısına eşit olmalıdır"
        ),
    ):
        create_coverage_result(
            covered_branch_count=2,
            missing_branch_count=1,
            total_branch_count=4,
        )


def test_coverage_result_rejects_negative_duration() -> None:
    """Negatif çalışma süresinin reddedildiğini doğrular."""
    with pytest.raises(
        ValueError,
        match="Coverage çalışma süresi negatif olamaz",
    ):
        create_coverage_result(
            duration_seconds=-0.1,
        )