from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """
    Bir coverage ölçüm işleminin sonucunu temsil eder.

    Attributes:
        source_file: Coverage ölçümü yapılan kaynak dosyanın yolu.
        test_file: Coverage ölçümünde çalıştırılan pytest dosyasının yolu.
        line_coverage_percent: Satır coverage yüzdesi.
        branch_coverage_percent: Branch coverage yüzdesi.
        covered_line_count: Çalıştırılan satır sayısı.
        missing_line_count: Çalıştırılmayan satır sayısı.
        total_line_count: Ölçülebilir toplam satır sayısı.
        covered_branch_count: Çalıştırılan branch sayısı.
        missing_branch_count: Çalıştırılmayan branch sayısı.
        total_branch_count: Ölçülebilir toplam branch sayısı.
        test_exit_code: Pytest işleminin çıkış kodu.
        duration_seconds: Coverage işleminin çalışma süresi.
    """

    source_file: Path
    test_file: Path
    line_coverage_percent: float
    branch_coverage_percent: float
    covered_line_count: int
    missing_line_count: int
    total_line_count: int
    covered_branch_count: int
    missing_branch_count: int
    total_branch_count: int
    test_exit_code: int
    duration_seconds: float

    def __post_init__(self) -> None:
        """Coverage sonucundaki alanların geçerliliğini doğrular."""
        self._validate_percentage(
            field_name="Satır coverage yüzdesi",
            value=self.line_coverage_percent,
        )
        self._validate_percentage(
            field_name="Branch coverage yüzdesi",
            value=self.branch_coverage_percent,
        )

        count_fields = {
            "Çalıştırılan satır sayısı": self.covered_line_count,
            "Eksik satır sayısı": self.missing_line_count,
            "Toplam satır sayısı": self.total_line_count,
            "Çalıştırılan branch sayısı": self.covered_branch_count,
            "Eksik branch sayısı": self.missing_branch_count,
            "Toplam branch sayısı": self.total_branch_count,
        }

        for field_name, value in count_fields.items():
            self._validate_non_negative_integer(
                field_name=field_name,
                value=value,
            )

        if self.duration_seconds < 0:
            raise ValueError(
                "Coverage çalışma süresi negatif olamaz."
            )

        if (
            self.covered_line_count + self.missing_line_count
            != self.total_line_count
        ):
            raise ValueError(
                "Çalıştırılan ve eksik satırların toplamı, "
                "toplam satır sayısına eşit olmalıdır."
            )

        if (
            self.covered_branch_count + self.missing_branch_count
            != self.total_branch_count
        ):
            raise ValueError(
                "Çalıştırılan ve eksik branchlerin toplamı, "
                "toplam branch sayısına eşit olmalıdır."
            )

    @property
    def success(self) -> bool:
        """
        Coverage işleminin başarılı olup olmadığını döndürür.

        Returns:
            Pytest çıkış kodu sıfırsa True.
        """
        return self.test_exit_code == 0

    @property
    def has_full_line_coverage(self) -> bool:
        """
        Satır coverage oranının yüzde 100 olup olmadığını döndürür.

        Returns:
            Eksik satır yoksa ve coverage yüzde 100 ise True.
        """
        return (
            self.line_coverage_percent == 100.0
            and self.missing_line_count == 0
        )

    @property
    def has_full_branch_coverage(self) -> bool:
        """
        Branch coverage oranının yüzde 100 olup olmadığını döndürür.

        Returns:
            Eksik branch yoksa ve coverage yüzde 100 ise True.
        """
        return (
            self.branch_coverage_percent == 100.0
            and self.missing_branch_count == 0
        )

    @property
    def has_full_coverage(self) -> bool:
        """
        Hem satır hem branch coverage oranının yüzde 100 olduğunu döndürür.

        Returns:
            Satır ve branch coverage değerlerinin ikisi de tam ise True.
        """
        return (
            self.has_full_line_coverage
            and self.has_full_branch_coverage
        )

    @staticmethod
    def _validate_percentage(
        field_name: str,
        value: float,
    ) -> None:
        """
        Yüzdelik coverage değerini doğrular.

        Args:
            field_name: Doğrulanan alanın adı.
            value: Coverage yüzdesi.

        Raises:
            TypeError: Değer sayısal değilse.
            ValueError: Değer 0–100 aralığında değilse.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} sayısal olmalıdır."
            )

        if not 0 <= float(value) <= 100:
            raise ValueError(
                f"{field_name} 0 ile 100 arasında olmalıdır."
            )

    @staticmethod
    def _validate_non_negative_integer(
        field_name: str,
        value: int,
    ) -> None:
        """
        Negatif olmayan tam sayı alanını doğrular.

        Args:
            field_name: Doğrulanan alanın adı.
            value: Doğrulanacak değer.

        Raises:
            TypeError: Değer tam sayı değilse.
            ValueError: Değer negatifse.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} tam sayı olmalıdır."
            )

        if value < 0:
            raise ValueError(
                f"{field_name} negatif olamaz."
            )