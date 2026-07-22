from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class ExecutionTimeoutError(TimeoutError):
    """Pytest Ã§alÄ±ÅŸtÄ±rma iÅŸlemi zaman aÅŸÄ±mÄ±na uÄŸradÄ±ÄŸÄ±nda oluÅŸur."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """
    Bir pytest Ã§alÄ±ÅŸtÄ±rma iÅŸleminin sonucunu temsil eder.

    Attributes:
        test_file: Ã‡alÄ±ÅŸtÄ±rÄ±lan pytest dosyasÄ±nÄ±n yolu.
        command: Pytest iÅŸlemi iÃ§in kullanÄ±lan komut.
        exit_code: Pytest iÅŸleminden dÃ¶nen Ã§Ä±kÄ±ÅŸ kodu.
        duration_seconds: Ä°ÅŸlemin saniye cinsinden Ã§alÄ±ÅŸma sÃ¼resi.
        stdout: Pytest standart Ã§Ä±ktÄ±sÄ±.
        stderr: Pytest hata Ã§Ä±ktÄ±sÄ±.
        passed_count: BaÅŸarÄ±lÄ± test sayÄ±sÄ±.
        failed_count: BaÅŸarÄ±sÄ±z test sayÄ±sÄ±.
        error_count: Test toplama veya Ã§alÄ±ÅŸtÄ±rma hata sayÄ±sÄ±.
        skipped_count: Atlanan test sayÄ±sÄ±.
    """

    test_file: Path
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int

    @property
    def success(self) -> bool:
        """
        Pytest iÅŸleminin baÅŸarÄ±lÄ± olup olmadÄ±ÄŸÄ±nÄ± dÃ¶ndÃ¼rÃ¼r.

        Returns:
            Ã‡Ä±kÄ±ÅŸ kodu sÄ±fÄ±rsa ve hata/baÅŸarÄ±sÄ±z test yoksa True.
        """
        return (
            self.exit_code == 0
            and self.failed_count == 0
            and self.error_count == 0
        )

    @property
    def total_test_count(self) -> int:
        """
        SonuÃ§ Ã¶zetindeki toplam test sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r.

        Returns:
            BaÅŸarÄ±lÄ±, baÅŸarÄ±sÄ±z, hatalÄ± ve atlanan testlerin toplamÄ±.
        """
        return (
            self.passed_count
            + self.failed_count
            + self.error_count
            + self.skipped_count
        )


class ExecutionService:
    """Pytest dosyalarÄ±nÄ± ayrÄ± bir iÅŸlemde gÃ¼venli biÃ§imde Ã§alÄ±ÅŸtÄ±rÄ±r."""

    _SUMMARY_PATTERN = re.compile(
        r"(?P<count>\d+)\s+"
        r"(?P<status>passed|failed|error|errors|skipped)"
    )

    def execute(
        self,
        test_file: str | Path,
        *,
        timeout_seconds: float = 30.0,
        extra_args: Sequence[str] = (),
    ) -> ExecutionResult:
        """
        Belirtilen pytest dosyasÄ±nÄ± Ã§alÄ±ÅŸtÄ±rÄ±r.

        Args:
            test_file: Ã‡alÄ±ÅŸtÄ±rÄ±lacak pytest dosyasÄ±nÄ±n yolu.
            timeout_seconds: Ä°ÅŸlem iÃ§in izin verilen azami sÃ¼re.
            extra_args: Pytest komutuna eklenecek isteÄŸe baÄŸlÄ± argÃ¼manlar.

        Returns:
            Pytest Ã§alÄ±ÅŸma sonucunu iÃ§eren ExecutionResult nesnesi.

        Raises:
            TypeError: Girdi tÃ¼rlerinden biri geÃ§ersiz olduÄŸunda.
            ValueError: Dosya yolu, sÃ¼re veya ek argÃ¼manlar geÃ§ersizse.
            FileNotFoundError: Test dosyasÄ± bulunamadÄ±ÄŸÄ±nda.
            ExecutionTimeoutError: Pytest iÅŸlemi zaman aÅŸÄ±mÄ±na
                uÄŸradÄ±ÄŸÄ±nda.
        """
        normalized_test_file = self._normalize_test_file(test_file)
        normalized_timeout = self._validate_timeout(timeout_seconds)
        normalized_extra_args = self._validate_extra_args(extra_args)

        command = (
            sys.executable,
            "-m",
            "pytest",
            str(normalized_test_file),
            "-q",
            "--color=no",
            *normalized_extra_args,
        )

        start_time = time.perf_counter()

        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=normalized_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            duration_seconds = time.perf_counter() - start_time

            raise ExecutionTimeoutError(
                "Pytest Ã§alÄ±ÅŸtÄ±rma iÅŸlemi zaman aÅŸÄ±mÄ±na uÄŸradÄ±. "
                f"Dosya: {normalized_test_file}, "
                f"sÃ¼re: {duration_seconds:.3f} saniye."
            ) from error

        duration_seconds = time.perf_counter() - start_time

        counts = self._parse_test_counts(
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
        )

        return ExecutionResult(
            test_file=normalized_test_file,
            command=command,
            exit_code=completed_process.returncode,
            duration_seconds=duration_seconds,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            passed_count=counts["passed"],
            failed_count=counts["failed"],
            error_count=counts["error"],
            skipped_count=counts["skipped"],
        )

    @staticmethod
    def _normalize_test_file(
        test_file: str | Path,
    ) -> Path:
        """
        Test dosyasÄ± yolunu doÄŸrular.

        Args:
            test_file: DoÄŸrulanacak test dosyasÄ± yolu.

        Returns:
            DoÄŸrulanmÄ±ÅŸ Path nesnesi.

        Raises:
            TypeError: Yol string veya Path deÄŸilse.
            ValueError: Yol boÅŸsa, dosya deÄŸilse veya uzantÄ±sÄ±
                ``.py`` deÄŸilse.
            FileNotFoundError: Dosya bulunamadÄ±ÄŸÄ±nda.
        """
        if not isinstance(test_file, (str, Path)):
            raise TypeError(
                "Test dosyasÄ± yolu string veya Path olmalÄ±dÄ±r."
            )

        if isinstance(test_file, str) and not test_file.strip():
            raise ValueError("Test dosyasÄ± yolu boÅŸ olamaz.")

        path = Path(test_file)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Test dosyasÄ±nÄ±n uzantÄ±sÄ± .py olmalÄ±dÄ±r."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Test dosyasÄ± bulunamadÄ±: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Test dosyasÄ± yolu bir dosya olmalÄ±dÄ±r: {path}"
            )

        return path

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> float:
        """
        Zaman aÅŸÄ±mÄ± deÄŸerini doÄŸrular.

        Args:
            timeout_seconds: DoÄŸrulanacak saniye deÄŸeri.

        Returns:
            DoÄŸrulanmÄ±ÅŸ zaman aÅŸÄ±mÄ± deÄŸeri.

        Raises:
            TypeError: DeÄŸer sayÄ±sal deÄŸilse.
            ValueError: DeÄŸer sÄ±fÄ±r veya negatifse.
        """
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds,
            (int, float),
        ):
            raise TypeError(
                "Zaman aÅŸÄ±mÄ± deÄŸeri sayÄ±sal olmalÄ±dÄ±r."
            )

        normalized_timeout = float(timeout_seconds)

        if normalized_timeout <= 0:
            raise ValueError(
                "Zaman aÅŸÄ±mÄ± deÄŸeri sÄ±fÄ±rdan bÃ¼yÃ¼k olmalÄ±dÄ±r."
            )

        return normalized_timeout

    @staticmethod
    def _validate_extra_args(
        extra_args: Sequence[str],
    ) -> tuple[str, ...]:
        """
        Ek pytest argÃ¼manlarÄ±nÄ± doÄŸrular.

        Args:
            extra_args: Pytest komutuna eklenecek argÃ¼manlar.

        Returns:
            DoÄŸrulanmÄ±ÅŸ argÃ¼man tuple'Ä±.

        Raises:
            TypeError: Koleksiyon veya eleman tÃ¼rÃ¼ geÃ§ersizse.
            ValueError: BoÅŸ bir argÃ¼man bulunduÄŸunda.
        """
        if isinstance(extra_args, str) or not isinstance(
            extra_args,
            Sequence,
        ):
            raise TypeError(
                "Ek pytest argÃ¼manlarÄ± bir string koleksiyonu olmalÄ±dÄ±r."
            )

        normalized_args: list[str] = []

        for argument in extra_args:
            if not isinstance(argument, str):
                raise TypeError(
                    "Her ek pytest argÃ¼manÄ± string olmalÄ±dÄ±r."
                )

            normalized_argument = argument.strip()

            if not normalized_argument:
                raise ValueError(
                    "Ek pytest argÃ¼manlarÄ± boÅŸ olamaz."
                )

            normalized_args.append(normalized_argument)

        return tuple(normalized_args)

    @classmethod
    def _parse_test_counts(
        cls,
        stdout: str,
        stderr: str,
    ) -> dict[str, int]:
        """
        Pytest Ã§Ä±ktÄ±sÄ±ndan test sayÄ±larÄ±nÄ± Ã§Ä±karÄ±r.

        Args:
            stdout: Pytest standart Ã§Ä±ktÄ±sÄ±.
            stderr: Pytest hata Ã§Ä±ktÄ±sÄ±.

        Returns:
            BaÅŸarÄ±lÄ±, baÅŸarÄ±sÄ±z, hatalÄ± ve atlanan test sayÄ±larÄ±.
        """
        counts = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
        }

        combined_output = f"{stdout}\n{stderr}"

        for match in cls._SUMMARY_PATTERN.finditer(combined_output):
            count = int(match.group("count"))
            status = match.group("status")

            normalized_status = (
                "error"
                if status in {"error", "errors"}
                else status
            )

            counts[normalized_status] = max(
                counts[normalized_status],
                count,
            )

        return counts

