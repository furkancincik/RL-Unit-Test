from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """
    Bir dosya bazlı coverage ölçüm işleminin sonucunu temsil eder.

    Attributes:
        source_file:
            Coverage ölçümü yapılan kaynak dosyanın yolu.

        test_file:
            Coverage ölçümünde çalıştırılan pytest dosyasının yolu.

        line_coverage_percent:
            Dosyanın satır coverage yüzdesi.

        branch_coverage_percent:
            Dosyanın branch coverage yüzdesi.

        covered_line_count:
            Çalıştırılan satır sayısı.

        missing_line_count:
            Çalıştırılmayan satır sayısı.

        total_line_count:
            Ölçülebilir toplam satır sayısı.

        covered_branch_count:
            Çalıştırılan branch sayısı.

        missing_branch_count:
            Çalıştırılmayan branch sayısı.

        total_branch_count:
            Ölçülebilir toplam branch sayısı.

        test_exit_code:
            Pytest işleminin çıkış kodu.

        duration_seconds:
            Coverage işleminin çalışma süresi.
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

        self._validate_exit_code(
            self.test_exit_code,
        )

        self._validate_duration(
            self.duration_seconds,
        )

        if (
            self.covered_line_count
            + self.missing_line_count
            != self.total_line_count
        ):
            raise ValueError(
                "Çalıştırılan ve eksik satırların toplamı, "
                "toplam satır sayısına eşit olmalıdır."
            )

        if (
            self.covered_branch_count
            + self.missing_branch_count
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
        """
        return (
            self.line_coverage_percent == 100.0
            and self.missing_line_count == 0
        )

    @property
    def has_full_branch_coverage(self) -> bool:
        """
        Branch coverage oranının yüzde 100 olup olmadığını döndürür.
        """
        return (
            self.branch_coverage_percent == 100.0
            and self.missing_branch_count == 0
        )

    @property
    def has_full_coverage(self) -> bool:
        """
        Hem satır hem branch coverage oranının yüzde 100
        olduğunu döndürür.
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
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{field_name} sayısal olmalıdır."
            )

        normalized_value = float(value)

        if not 0.0 <= normalized_value <= 100.0:
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
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} tam sayı olmalıdır."
            )

        if value < 0:
            raise ValueError(
                f"{field_name} negatif olamaz."
            )

    @staticmethod
    def _validate_exit_code(
        value: int,
    ) -> None:
        """Pytest çıkış kodunu doğrular."""
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                "Test çıkış kodu tam sayı olmalıdır."
            )

    @staticmethod
    def _validate_duration(
        value: float,
    ) -> None:
        """Coverage çalışma süresini doğrular."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                "Coverage çalışma süresi sayısal olmalıdır."
            )

        if float(value) < 0.0:
            raise ValueError(
                "Coverage çalışma süresi negatif olamaz."
            )


@dataclass(frozen=True, slots=True)
class FunctionCoverageResult:
    """
    Belirli bir Python fonksiyonuna ait coverage sonucunu temsil eder.

    Dosya genelindeki coverage oranından farklı olarak yalnızca hedef
    fonksiyonun başlangıç ve bitiş satırları arasındaki çalıştırılabilir
    satırlar ve branchler değerlendirilir.

    Attributes:
        source_file:
            Fonksiyonun bulunduğu Python kaynak dosyası.

        test_file:
            Coverage ölçümünde çalıştırılan pytest dosyası.

        function_name:
            Coverage ölçümü yapılan fonksiyonun adı.

        start_line:
            Fonksiyonun kaynak dosyadaki başlangıç satırı.

        end_line:
            Fonksiyonun kaynak dosyadaki bitiş satırı.

        line_coverage_percent:
            Fonksiyonun satır coverage yüzdesi.

        branch_coverage_percent:
            Fonksiyonun branch coverage yüzdesi.

        covered_lines:
            Fonksiyon içerisinde çalıştırılan satır numaraları.

        missing_lines:
            Fonksiyon içerisinde çalıştırılmayan satır numaraları.

        covered_branch_count:
            Fonksiyon içerisinde çalıştırılan branch sayısı.

        missing_branch_count:
            Fonksiyon içerisinde çalıştırılmayan branch sayısı.

        test_exit_code:
            Pytest işleminin çıkış kodu.

        duration_seconds:
            Coverage işleminin çalışma süresi.

        file_coverage:
            Aynı test çalıştırmasına ait dosya geneli coverage sonucu.
    """

    source_file: Path
    test_file: Path
    function_name: str
    start_line: int
    end_line: int
    line_coverage_percent: float
    branch_coverage_percent: float
    covered_lines: tuple[int, ...]
    missing_lines: tuple[int, ...]
    covered_branch_count: int
    missing_branch_count: int
    test_exit_code: int
    duration_seconds: float
    file_coverage: CoverageResult

    def __post_init__(self) -> None:
        """Fonksiyon coverage alanlarının geçerliliğini doğrular."""
        self._validate_paths()
        self._validate_function_name()
        self._validate_line_range()

        CoverageResult._validate_percentage(
            field_name="Fonksiyon satır coverage yüzdesi",
            value=self.line_coverage_percent,
        )

        CoverageResult._validate_percentage(
            field_name="Fonksiyon branch coverage yüzdesi",
            value=self.branch_coverage_percent,
        )

        self._validate_line_collection(
            field_name="covered_lines",
            lines=self.covered_lines,
        )

        self._validate_line_collection(
            field_name="missing_lines",
            lines=self.missing_lines,
        )

        if set(self.covered_lines) & set(self.missing_lines):
            raise ValueError(
                "Çalıştırılan ve eksik satırlar kesişemez."
            )

        CoverageResult._validate_non_negative_integer(
            field_name="Fonksiyon çalıştırılan branch sayısı",
            value=self.covered_branch_count,
        )

        CoverageResult._validate_non_negative_integer(
            field_name="Fonksiyon eksik branch sayısı",
            value=self.missing_branch_count,
        )

        CoverageResult._validate_exit_code(
            self.test_exit_code,
        )

        CoverageResult._validate_duration(
            self.duration_seconds,
        )

        if not isinstance(
            self.file_coverage,
            CoverageResult,
        ):
            raise TypeError(
                "file_coverage bir CoverageResult "
                "örneği olmalıdır."
            )

        if self.source_file.resolve() != (
            self.file_coverage.source_file.resolve()
        ):
            raise ValueError(
                "Fonksiyon coverage ve dosya coverage "
                "kaynak dosyaları eşleşmelidir."
            )

        if self.test_file.resolve() != (
            self.file_coverage.test_file.resolve()
        ):
            raise ValueError(
                "Fonksiyon coverage ve dosya coverage "
                "test dosyaları eşleşmelidir."
            )

    @property
    def covered_line_count(self) -> int:
        """Fonksiyon içerisinde çalıştırılan satır sayısını döndürür."""
        return len(self.covered_lines)

    @property
    def missing_line_count(self) -> int:
        """Fonksiyon içerisinde eksik kalan satır sayısını döndürür."""
        return len(self.missing_lines)

    @property
    def total_line_count(self) -> int:
        """Fonksiyonun ölçülebilir toplam satır sayısını döndürür."""
        return (
            self.covered_line_count
            + self.missing_line_count
        )

    @property
    def total_branch_count(self) -> int:
        """Fonksiyonun toplam branch sayısını döndürür."""
        return (
            self.covered_branch_count
            + self.missing_branch_count
        )

    @property
    def success(self) -> bool:
        """Pytest işlemi başarılıysa True döndürür."""
        return self.test_exit_code == 0

    @property
    def has_full_line_coverage(self) -> bool:
        """Fonksiyon satır coverage değeri tam ise True döndürür."""
        return (
            self.line_coverage_percent == 100.0
            and self.missing_line_count == 0
        )

    @property
    def has_full_branch_coverage(self) -> bool:
        """Fonksiyon branch coverage değeri tam ise True döndürür."""
        return (
            self.branch_coverage_percent == 100.0
            and self.missing_branch_count == 0
        )

    @property
    def has_full_coverage(self) -> bool:
        """Fonksiyon satır ve branch coverage değerleri tam ise True."""
        return (
            self.has_full_line_coverage
            and self.has_full_branch_coverage
        )

    @property
    def file_line_coverage_percent(self) -> float:
        """Dosya genelindeki satır coverage yüzdesini döndürür."""
        return self.file_coverage.line_coverage_percent

    @property
    def file_branch_coverage_percent(self) -> float:
        """Dosya genelindeki branch coverage yüzdesini döndürür."""
        return self.file_coverage.branch_coverage_percent

    def _validate_paths(self) -> None:
        """Kaynak ve test dosyası alanlarını doğrular."""
        if not isinstance(self.source_file, Path):
            raise TypeError(
                "source_file bir Path örneği olmalıdır."
            )

        if not isinstance(self.test_file, Path):
            raise TypeError(
                "test_file bir Path örneği olmalıdır."
            )

    def _validate_function_name(self) -> None:
        """Fonksiyon adını doğrular."""
        if not isinstance(self.function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        if not self.function_name.strip():
            raise ValueError(
                "function_name boş olamaz."
            )

        if not self.function_name.isidentifier():
            raise ValueError(
                "function_name geçerli bir Python "
                "tanımlayıcısı olmalıdır."
            )

    def _validate_line_range(self) -> None:
        """Fonksiyon başlangıç ve bitiş satırlarını doğrular."""
        CoverageResult._validate_non_negative_integer(
            field_name="Fonksiyon başlangıç satırı",
            value=self.start_line,
        )

        CoverageResult._validate_non_negative_integer(
            field_name="Fonksiyon bitiş satırı",
            value=self.end_line,
        )

        if self.start_line < 1:
            raise ValueError(
                "Fonksiyon başlangıç satırı en az 1 olmalıdır."
            )

        if self.end_line < self.start_line:
            raise ValueError(
                "Fonksiyon bitiş satırı başlangıç "
                "satırından küçük olamaz."
            )

    def _validate_line_collection(
        self,
        *,
        field_name: str,
        lines: tuple[int, ...],
    ) -> None:
        """Fonksiyon satır numarası koleksiyonunu doğrular."""
        if not isinstance(lines, tuple):
            raise TypeError(
                f"{field_name} bir tuple olmalıdır."
            )

        if any(
            isinstance(line, bool)
            or not isinstance(line, int)
            for line in lines
        ):
            raise TypeError(
                f"{field_name} yalnızca tam sayı "
                "değerler içermelidir."
            )

        if any(
            line < self.start_line
            or line > self.end_line
            for line in lines
        ):
            raise ValueError(
                f"{field_name} yalnızca fonksiyon satır "
                "aralığındaki değerleri içermelidir."
            )

        if len(set(lines)) != len(lines):
            raise ValueError(
                f"{field_name} tekrar eden satır "
                "numarası içeremez."
            )

        if tuple(sorted(lines)) != lines:
            raise ValueError(
                f"{field_name} artan sırada olmalıdır."
            )