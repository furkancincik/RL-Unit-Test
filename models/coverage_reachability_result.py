from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models.coverage_result import FunctionCoverageResult


class LineReachabilityStatus(str, Enum):
    """Bir fonksiyon satırının coverage ve path analiz durumudur."""

    COVERED = "COVERED"
    FEASIBLE_UNCOVERED = "FEASIBLE_UNCOVERED"
    INFEASIBLE_ONLY = "INFEASIBLE_ONLY"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class LineReachabilityEvidence:
    """
    Tek bir çalıştırılabilir satır için sınıflandırma kanıtını taşır.

    Path indeksleri bir tabanlıdır ve ``PathFeasibilityAnalyzer``
    sonuçlarının giriş sırasına karşılık gelir. ``INFEASIBLE_ONLY``
    yalnızca sınırlandırılmış CFG path kümesi içindeki sonucu ifade eder;
    programın bütün olası çalışmaları için evrensel erişilemezlik kanıtı
    değildir.
    """

    line_number: int
    status: LineReachabilityStatus
    feasible_path_indices: tuple[int, ...] = ()
    infeasible_path_indices: tuple[int, ...] = ()
    unknown_path_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Satır ve kanıt alanlarının tutarlılığını doğrular."""
        if (
            isinstance(self.line_number, bool)
            or not isinstance(self.line_number, int)
        ):
            raise TypeError(
                "line_number bir integer olmalıdır."
            )

        if self.line_number < 1:
            raise ValueError(
                "line_number en az 1 olmalıdır."
            )

        if not isinstance(
            self.status,
            LineReachabilityStatus,
        ):
            raise TypeError(
                "status bir LineReachabilityStatus "
                "değeri olmalıdır."
            )

        collections = {
            "feasible_path_indices": self.feasible_path_indices,
            "infeasible_path_indices": self.infeasible_path_indices,
            "unknown_path_indices": self.unknown_path_indices,
        }

        for field_name, indices in collections.items():
            self._validate_path_indices(
                field_name=field_name,
                indices=indices,
            )

        feasible_indices = set(
            self.feasible_path_indices
        )
        infeasible_indices = set(
            self.infeasible_path_indices
        )
        unknown_indices = set(
            self.unknown_path_indices
        )

        if (
            feasible_indices & infeasible_indices
            or feasible_indices & unknown_indices
            or infeasible_indices & unknown_indices
        ):
            raise ValueError(
                "Aynı path indeksi birden fazla feasibility "
                "durumunda bulunamaz."
            )

        if (
            self.status
            == LineReachabilityStatus.FEASIBLE_UNCOVERED
            and not self.feasible_path_indices
        ):
            raise ValueError(
                "FEASIBLE_UNCOVERED satır en az bir FEASIBLE "
                "path kanıtı içermelidir."
            )

        if (
            self.status
            == LineReachabilityStatus.INFEASIBLE_ONLY
        ):
            if not self.infeasible_path_indices:
                raise ValueError(
                    "INFEASIBLE_ONLY satır en az bir INFEASIBLE "
                    "path kanıtı içermelidir."
                )

            if (
                self.feasible_path_indices
                or self.unknown_path_indices
            ):
                raise ValueError(
                    "INFEASIBLE_ONLY satır FEASIBLE veya UNKNOWN "
                    "path kanıtı içeremez."
                )

        if (
            self.status == LineReachabilityStatus.UNRESOLVED
            and self.feasible_path_indices
        ):
            raise ValueError(
                "UNRESOLVED satır FEASIBLE path kanıtı "
                "içeremez."
            )

    @property
    def path_indices(self) -> tuple[int, ...]:
        """Satırla ilişkili bütün path indekslerini sıralı döndürür."""
        return tuple(
            sorted(
                (
                    *self.feasible_path_indices,
                    *self.infeasible_path_indices,
                    *self.unknown_path_indices,
                )
            )
        )

    @staticmethod
    def _validate_path_indices(
        *,
        field_name: str,
        indices: tuple[int, ...],
    ) -> None:
        """Bir tabanlı path indeks koleksiyonunu doğrular."""
        if not isinstance(indices, tuple):
            raise TypeError(
                f"{field_name} bir tuple olmalıdır."
            )

        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            for index in indices
        ):
            raise TypeError(
                f"{field_name} yalnızca integer "
                "değerler içermelidir."
            )

        if any(index < 1 for index in indices):
            raise ValueError(
                f"{field_name} yalnızca pozitif "
                "değerler içermelidir."
            )

        if len(set(indices)) != len(indices):
            raise ValueError(
                f"{field_name} tekrar eden path "
                "indeksi içeremez."
            )

        if tuple(sorted(indices)) != indices:
            raise ValueError(
                f"{field_name} artan sırada olmalıdır."
            )


@dataclass(frozen=True, slots=True)
class FunctionCoverageReachabilityResult:
    """
    Fonksiyon coverage satırlarını bounded path analiziyle sınıflandırır.

    Bu model ham coverage sonucunu değiştirmez. Özellikle
    ``INFEASIBLE_ONLY`` satırlar, yalnızca ``max_visits_per_node`` ile
    üretilmiş path kümesi kapsamında yorumlanmalıdır.
    """

    coverage_result: FunctionCoverageResult
    line_evidence: tuple[LineReachabilityEvidence, ...]
    analyzed_path_count: int
    max_visits_per_node: int
    path_metadata_complete: bool

    def __post_init__(self) -> None:
        """Coverage ve sınıflandırma bilgilerinin tutarlılığını doğrular."""
        if not isinstance(
            self.coverage_result,
            FunctionCoverageResult,
        ):
            raise TypeError(
                "coverage_result bir FunctionCoverageResult "
                "örneği olmalıdır."
            )

        self._validate_positive_integer(
            field_name="analyzed_path_count",
            value=self.analyzed_path_count,
        )
        self._validate_positive_integer(
            field_name="max_visits_per_node",
            value=self.max_visits_per_node,
        )

        if not isinstance(
            self.path_metadata_complete,
            bool,
        ):
            raise TypeError(
                "path_metadata_complete bir bool olmalıdır."
            )

        if not isinstance(self.line_evidence, tuple):
            raise TypeError(
                "line_evidence bir tuple olmalıdır."
            )

        if any(
            not isinstance(
                evidence,
                LineReachabilityEvidence,
            )
            for evidence in self.line_evidence
        ):
            raise TypeError(
                "line_evidence yalnızca LineReachabilityEvidence "
                "nesneleri içermelidir."
            )

        line_numbers = tuple(
            evidence.line_number
            for evidence in self.line_evidence
        )

        if len(set(line_numbers)) != len(line_numbers):
            raise ValueError(
                "line_evidence tekrar eden satır içeremez."
            )

        if tuple(sorted(line_numbers)) != line_numbers:
            raise ValueError(
                "line_evidence satır numarasına göre "
                "artan sırada olmalıdır."
            )

        expected_lines = set(
            self.coverage_result.covered_lines
        ) | set(
            self.coverage_result.missing_lines
        )

        if set(line_numbers) != expected_lines:
            raise ValueError(
                "line_evidence, coverage sonucundaki bütün "
                "çalıştırılabilir satırları tam olarak içermelidir."
            )

        covered_lines = set(
            self.coverage_result.covered_lines
        )

        for evidence in self.line_evidence:
            if (
                evidence.line_number in covered_lines
                and evidence.status
                != LineReachabilityStatus.COVERED
            ):
                raise ValueError(
                    "Coverage tarafından çalıştırılmış satırlar "
                    "COVERED olarak sınıflandırılmalıdır."
                )

            if (
                evidence.line_number not in covered_lines
                and evidence.status
                == LineReachabilityStatus.COVERED
            ):
                raise ValueError(
                    "Coverage tarafından eksik bildirilen satırlar "
                    "COVERED olarak sınıflandırılamaz."
                )

            if (
                evidence.path_indices
                and evidence.path_indices[-1]
                > self.analyzed_path_count
            ):
                raise ValueError(
                    "Satır kanıtındaki path indeksi analiz edilen "
                    "path sayısını aşamaz."
                )

            if (
                not self.path_metadata_complete
                and evidence.status
                == LineReachabilityStatus.INFEASIBLE_ONLY
            ):
                raise ValueError(
                    "Path metadata eksikken INFEASIBLE_ONLY "
                    "sınıflandırması yapılamaz."
                )

    @property
    def covered_lines(self) -> tuple[int, ...]:
        """Gerçek coverage çalıştırmasında kapsanan satırları döndürür."""
        return self._lines_with_status(
            LineReachabilityStatus.COVERED
        )

    @property
    def feasible_uncovered_lines(self) -> tuple[int, ...]:
        """FEASIBLE path üzerinde bulunup kapsanmayan satırları döndürür."""
        return self._lines_with_status(
            LineReachabilityStatus.FEASIBLE_UNCOVERED
        )

    @property
    def infeasible_only_lines(self) -> tuple[int, ...]:
        """Yalnızca bounded INFEASIBLE path'lerde görülen satırları döndürür."""
        return self._lines_with_status(
            LineReachabilityStatus.INFEASIBLE_ONLY
        )

    @property
    def unresolved_lines(self) -> tuple[int, ...]:
        """Güvenle sınıflandırılamayan eksik satırları döndürür."""
        return self._lines_with_status(
            LineReachabilityStatus.UNRESOLVED
        )

    @property
    def covered_line_count(self) -> int:
        return len(self.covered_lines)

    @property
    def feasible_uncovered_line_count(self) -> int:
        return len(self.feasible_uncovered_lines)

    @property
    def infeasible_only_line_count(self) -> int:
        return len(self.infeasible_only_lines)

    @property
    def unresolved_line_count(self) -> int:
        return len(self.unresolved_lines)

    @property
    def has_unresolved_lines(self) -> bool:
        """En az bir satır çözümlenememişse True döndürür."""
        return bool(self.unresolved_lines)

    @property
    def classification_complete(self) -> bool:
        """Bounded analiz tüm satırları metadata ile sınıflandırdıysa True."""
        return (
            self.path_metadata_complete
            and not self.has_unresolved_lines
        )

    @property
    def observed_line_coverage_percent(self) -> float:
        """Ham gerçek test çalıştırmasının satır coverage yüzdesidir."""
        return self.coverage_result.line_coverage_percent

    def evidence_for_line(
        self,
        line_number: int,
    ) -> LineReachabilityEvidence:
        """Belirtilen çalıştırılabilir satırın kanıtını döndürür."""
        if (
            isinstance(line_number, bool)
            or not isinstance(line_number, int)
        ):
            raise TypeError(
                "line_number bir integer olmalıdır."
            )

        for evidence in self.line_evidence:
            if evidence.line_number == line_number:
                return evidence

        raise KeyError(
            "Coverage sonucunda satır bulunamadı: "
            f"{line_number}"
        )

    def _lines_with_status(
        self,
        status: LineReachabilityStatus,
    ) -> tuple[int, ...]:
        return tuple(
            evidence.line_number
            for evidence in self.line_evidence
            if evidence.status == status
        )

    @staticmethod
    def _validate_positive_integer(
        *,
        field_name: str,
        value: int,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} bir integer olmalıdır."
            )

        if value < 1:
            raise ValueError(
                f"{field_name} en az 1 olmalıdır."
            )
