from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.coverage_result import CoverageResult
from services.coverage_service import CoverageService
from services.generation_service import (
    GeneratedTestArtifact,
    GenerationService,
)


@dataclass(frozen=True, slots=True)
class CoverageArtifactResult:
    """
    Üretilen test dosyası ile coverage sonucunu ilişkilendirir.

    Attributes:
        artifact: Otomatik oluşturulan pytest dosyasının bilgileri.
        coverage: Kaynak dosya için elde edilen coverage sonucu.
    """

    artifact: GeneratedTestArtifact
    coverage: CoverageResult

    @property
    def success(self) -> bool:
        """Coverage altında çalıştırılan pytest işleminin sonucunu döndürür."""
        return self.coverage.success


@dataclass(frozen=True, slots=True)
class CoverageWorkflowSummary:
    """
    Otomatik test üretimi ve coverage ölçümünün toplu sonucudur.

    Attributes:
        results: Fonksiyon bazındaki üretim ve coverage sonuçları.
    """

    results: tuple[CoverageArtifactResult, ...]

    @property
    def measured_file_count(self) -> int:
        """Coverage ölçülen test dosyası sayısını döndürür."""
        return len(self.results)

    @property
    def successful_file_count(self) -> int:
        """Coverage altında başarıyla çalışan dosya sayısını döndürür."""
        return sum(
            result.success
            for result in self.results
        )

    @property
    def failed_file_count(self) -> int:
        """Coverage altında başarısız çalışan dosya sayısını döndürür."""
        return (
            self.measured_file_count
            - self.successful_file_count
        )

    @property
    def full_coverage_file_count(self) -> int:
        """Tam satır ve branch coverage sağlayan dosya sayısını döndürür."""
        return sum(
            result.coverage.has_full_coverage
            for result in self.results
        )

    @property
    def average_line_coverage_percent(self) -> float:
        """Dosyaların ortalama satır coverage yüzdesini döndürür."""
        if not self.results:
            return 0.0

        return round(
            sum(
                result.coverage.line_coverage_percent
                for result in self.results
            )
            / len(self.results),
            2,
        )

    @property
    def average_branch_coverage_percent(self) -> float:
        """Dosyaların ortalama branch coverage yüzdesini döndürür."""
        if not self.results:
            return 0.0

        return round(
            sum(
                result.coverage.branch_coverage_percent
                for result in self.results
            )
            / len(self.results),
            2,
        )

    @property
    def success(self) -> bool:
        """
        Bütün coverage işlemlerinin başarılı olup olmadığını döndürür.

        En az bir ölçüm sonucu bulunmalı ve tüm pytest işlemleri
        başarılı olmalıdır.
        """
        return bool(self.results) and all(
            result.success
            for result in self.results
        )


class CoverageWorkflowService:
    """Test üretme ve coverage ölçme işlemlerini uçtan uca yönetir."""

    def __init__(
        self,
        generation_service: GenerationService | None = None,
        coverage_service: CoverageService | None = None,
    ) -> None:
        """
        Coverage iş akışının bağımlılıklarını hazırlar.

        Args:
            generation_service: Pytest dosyalarını üreten servis.
            coverage_service: Coverage ölçüm servisi.
        """
        self._generation_service = (
            generation_service or GenerationService()
        )
        self._coverage_service = (
            coverage_service or CoverageService()
        )

    def generate_and_measure(
        self,
        source_file: str | Path,
        module_path: str,
        output_directory: str | Path,
        *,
        overwrite: bool = False,
        timeout_seconds: float = 30.0,
    ) -> CoverageWorkflowSummary:
        """
        Kaynak dosya için test üretir ve coverage ölçümü gerçekleştirir.

        Args:
            source_file: Analiz ve coverage uygulanacak kaynak dosya.
            module_path: Kaynak dosyanın Python import yolu.
            output_directory: Üretilen testlerin kaydedileceği klasör.
            overwrite: Mevcut test dosyalarının üzerine yazma izni.
            timeout_seconds: Her coverage işlemi için süre sınırı.

        Returns:
            Test dosyaları ile coverage sonuçlarını içeren özet.

        Raises:
            ValueError: Hiçbir test dosyası üretilemediğinde.
        """
        artifacts = self._generation_service.generate_for_file(
            source_file=source_file,
            module_path=module_path,
            output_directory=output_directory,
            overwrite=overwrite,
        )

        if not artifacts:
            raise ValueError(
                "Coverage ölçümü için test dosyası oluşturulamadı."
            )

        results: list[CoverageArtifactResult] = []

        for artifact in artifacts:
            coverage_result = self._coverage_service.measure(
                source_file=source_file,
                test_file=artifact.output_path,
                timeout_seconds=timeout_seconds,
            )

            results.append(
                CoverageArtifactResult(
                    artifact=artifact,
                    coverage=coverage_result,
                )
            )

        return CoverageWorkflowSummary(
            results=tuple(results),
        )