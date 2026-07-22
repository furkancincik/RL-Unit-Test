from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.execution_service import (
    ExecutionResult,
    ExecutionService,
)
from services.generation_service import (
    GeneratedTestArtifact,
    GenerationService,
)


@dataclass(frozen=True, slots=True)
class AutomatedTestResult:
    """
    Otomatik test Ã¼retme ve Ã§alÄ±ÅŸtÄ±rma iÅŸleminin sonucunu temsil eder.

    Attributes:
        artifact: Ãœretilen pytest dosyasÄ±na ait bilgiler.
        execution: Ãœretilen test dosyasÄ±nÄ±n Ã§alÄ±ÅŸtÄ±rma sonucu.
    """

    artifact: GeneratedTestArtifact
    execution: ExecutionResult

    @property
    def success(self) -> bool:
        """
        Otomatik test iÅŸleminin baÅŸarÄ±lÄ± olup olmadÄ±ÄŸÄ±nÄ± dÃ¶ndÃ¼rÃ¼r.

        Returns:
            Ãœretilen test dosyasÄ± baÅŸarÄ±yla Ã§alÄ±ÅŸtÄ±ysa True.
        """
        return self.execution.success


@dataclass(frozen=True, slots=True)
class AutomationSummary:
    """
    Otomatik test Ã¼retim sÃ¼recinin toplu sonucunu temsil eder.

    Attributes:
        results: Her fonksiyon iÃ§in Ã¼retim ve Ã§alÄ±ÅŸtÄ±rma sonuÃ§larÄ±.
    """

    results: tuple[AutomatedTestResult, ...]

    @property
    def generated_file_count(self) -> int:
        """OluÅŸturulan test dosyasÄ± sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return len(self.results)

    @property
    def successful_file_count(self) -> int:
        """BaÅŸarÄ±yla Ã§alÄ±ÅŸan test dosyasÄ± sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return sum(
            result.success
            for result in self.results
        )

    @property
    def failed_file_count(self) -> int:
        """BaÅŸarÄ±sÄ±z Ã§alÄ±ÅŸan test dosyasÄ± sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return (
            self.generated_file_count
            - self.successful_file_count
        )

    @property
    def total_generated_scenario_count(self) -> int:
        """Ãœretilen toplam test senaryosu sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return sum(
            result.artifact.scenario_count
            for result in self.results
        )

    @property
    def total_passed_test_count(self) -> int:
        """Ã‡alÄ±ÅŸtÄ±rma sonucundaki toplam baÅŸarÄ±lÄ± test sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return sum(
            result.execution.passed_count
            for result in self.results
        )

    @property
    def total_failed_test_count(self) -> int:
        """Ã‡alÄ±ÅŸtÄ±rma sonucundaki toplam baÅŸarÄ±sÄ±z test sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return sum(
            result.execution.failed_count
            for result in self.results
        )

    @property
    def success(self) -> bool:
        """
        BÃ¼tÃ¼n oluÅŸturulan test dosyalarÄ±nÄ±n baÅŸarÄ±lÄ± olup olmadÄ±ÄŸÄ±nÄ± dÃ¶ndÃ¼rÃ¼r.

        Returns:
            En az bir sonuÃ§ varsa ve bÃ¼tÃ¼n sonuÃ§lar baÅŸarÄ±lÄ±ysa True.
        """
        return bool(self.results) and all(
            result.success
            for result in self.results
        )


class AutomationService:
    """Test Ã¼retme ve Ã§alÄ±ÅŸtÄ±rma sÃ¼recini uÃ§tan uca yÃ¶netir."""

    def __init__(
        self,
        generation_service: GenerationService | None = None,
        execution_service: ExecutionService | None = None,
    ) -> None:
        """
        Otomasyon servisinin baÄŸÄ±mlÄ±lÄ±klarÄ±nÄ± hazÄ±rlar.

        Args:
            generation_service: Pytest dosyalarÄ±nÄ± Ã¼reten servis.
            execution_service: Pytest dosyalarÄ±nÄ± Ã§alÄ±ÅŸtÄ±ran servis.
        """
        self._generation_service = (
            generation_service or GenerationService()
        )
        self._execution_service = (
            execution_service or ExecutionService()
        )

    def generate_and_execute(
        self,
        source_file: str | Path,
        module_path: str,
        output_directory: str | Path,
        *,
        overwrite: bool = False,
        timeout_seconds: float = 30.0,
    ) -> AutomationSummary:
        """
        Kaynak dosya iÃ§in pytest dosyalarÄ± Ã¼retir ve Ã§alÄ±ÅŸtÄ±rÄ±r.

        Args:
            source_file: Analiz edilecek Python kaynak dosyasÄ±nÄ±n yolu.
            module_path: Kaynak dosyanÄ±n Python modÃ¼l yolu.
            output_directory: Test dosyalarÄ±nÄ±n oluÅŸturulacaÄŸÄ± klasÃ¶r.
            overwrite: Mevcut test dosyalarÄ±nÄ±n Ã¼zerine yazma izni.
            timeout_seconds: Her test dosyasÄ± iÃ§in azami Ã§alÄ±ÅŸma sÃ¼resi.

        Returns:
            Ãœretim ve Ã§alÄ±ÅŸtÄ±rma sonuÃ§larÄ±nÄ± iÃ§eren toplu sonuÃ§.

        Raises:
            ValueError: Ãœretim servisi herhangi bir test dosyasÄ±
                oluÅŸturmadÄ±ÄŸÄ±nda.
        """
        artifacts = self._generation_service.generate_for_file(
            source_file=source_file,
            module_path=module_path,
            output_directory=output_directory,
            overwrite=overwrite,
        )

        if not artifacts:
            raise ValueError(
                "Otomatik Ã§alÄ±ÅŸtÄ±rma iÃ§in test dosyasÄ± oluÅŸturulamadÄ±."
            )

        results: list[AutomatedTestResult] = []

        for artifact in artifacts:
            execution_result = self._execution_service.execute(
                test_file=artifact.output_path,
                timeout_seconds=timeout_seconds,
            )

            results.append(
                AutomatedTestResult(
                    artifact=artifact,
                    execution=execution_result,
                )
            )

        return AutomationSummary(
            results=tuple(results),
        )

