from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from models.coverage_result import CoverageResult


class CoverageExecutionTimeoutError(TimeoutError):
    """Coverage ölçüm işlemi zaman aşımına uğradığında oluşur."""


class CoverageService:
    """Pytest çalıştırarak satır ve branch coverage değerlerini ölçer."""

    def measure(
        self,
        source_file: str | Path,
        test_file: str | Path,
        *,
        timeout_seconds: float = 30.0,
    ) -> CoverageResult:
        """
        Kaynak dosyanın pytest testiyle elde edilen coverage değerini ölçer.

        Args:
            source_file: Coverage ölçümü yapılacak Python kaynak dosyası.
            test_file: Çalıştırılacak pytest dosyası.
            timeout_seconds: Coverage işlemi için azami çalışma süresi.

        Returns:
            Satır ve branch kapsam bilgilerini içeren CoverageResult.

        Raises:
            TypeError: Yol veya zaman aşımı değerinin türü geçersizse.
            ValueError: Yol, dosya uzantısı veya zaman aşımı geçersizse.
            FileNotFoundError: Kaynak veya test dosyası bulunamazsa.
            CoverageExecutionTimeoutError: İşlem zaman aşımına uğrarsa.
            RuntimeError: Coverage raporu oluşturulamaz veya okunamazsa.
        """
        normalized_source_file = self._normalize_python_file(
            file_path=source_file,
            field_name="Kaynak dosya",
        )
        normalized_test_file = self._normalize_python_file(
            file_path=test_file,
            field_name="Test dosyası",
        )
        normalized_timeout = self._validate_timeout(
            timeout_seconds
        )

        start_time = time.perf_counter()

        with tempfile.TemporaryDirectory(
            prefix="rl_unit_test_coverage_"
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            coverage_data_file = (
                temporary_path / ".coverage"
            )
            coverage_json_file = (
                temporary_path / "coverage.json"
            )

            environment = os.environ.copy()
            environment["COVERAGE_FILE"] = str(
                coverage_data_file
            )

            run_command = self._create_run_command(
                source_file=normalized_source_file,
                test_file=normalized_test_file,
            )

            try:
                run_process = subprocess.run(
                    run_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=normalized_timeout,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                duration_seconds = (
                    time.perf_counter() - start_time
                )

                raise CoverageExecutionTimeoutError(
                    "Coverage ölçüm işlemi zaman aşımına uğradı. "
                    f"Süre: {duration_seconds:.3f} saniye."
                ) from error

            remaining_timeout = self._calculate_remaining_timeout(
                start_time=start_time,
                timeout_seconds=normalized_timeout,
            )

            report_command = self._create_report_command(
                source_file=normalized_source_file,
                output_file=coverage_json_file,
            )

            try:
                report_process = subprocess.run(
                    report_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=remaining_timeout,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                duration_seconds = (
                    time.perf_counter() - start_time
                )

                raise CoverageExecutionTimeoutError(
                    "Coverage raporu oluşturma işlemi "
                    "zaman aşımına uğradı. "
                    f"Süre: {duration_seconds:.3f} saniye."
                ) from error

            if report_process.returncode != 0:
                raise RuntimeError(
                    "Coverage JSON raporu oluşturulamadı. "
                    f"Hata: {report_process.stderr.strip()}"
                )

            report_data = self._read_report(
                coverage_json_file
            )

            file_summary = self._find_file_summary(
                report_data=report_data,
                source_file=normalized_source_file,
            )

        duration_seconds = time.perf_counter() - start_time

        return self._create_result(
            source_file=normalized_source_file,
            test_file=normalized_test_file,
            test_exit_code=run_process.returncode,
            duration_seconds=duration_seconds,
            file_summary=file_summary,
        )

    @staticmethod
    def _create_run_command(
        source_file: Path,
        test_file: Path,
    ) -> tuple[str, ...]:
        """Coverage altında pytest çalıştırma komutunu oluşturur."""
        return (
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            f"--include={source_file}",
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "--color=no",
        )

    @staticmethod
    def _create_report_command(
        source_file: Path,
        output_file: Path,
    ) -> tuple[str, ...]:
        """Coverage JSON raporu oluşturma komutunu hazırlar."""
        return (
            sys.executable,
            "-m",
            "coverage",
            "json",
            "-o",
            str(output_file),
            f"--include={source_file}",
        )

    @staticmethod
    def _normalize_python_file(
        file_path: str | Path,
        field_name: str,
    ) -> Path:
        """
        Python dosyası yolunu doğrular.

        Args:
            file_path: Doğrulanacak dosya yolu.
            field_name: Hata mesajında kullanılacak alan adı.

        Returns:
            Mutlak ve doğrulanmış Path nesnesi.
        """
        if not isinstance(file_path, (str, Path)):
            raise TypeError(
                f"{field_name} yolu string veya Path olmalıdır."
            )

        if isinstance(file_path, str) and not file_path.strip():
            raise ValueError(
                f"{field_name} yolu boş olamaz."
            )

        path = Path(file_path)

        if path.suffix.lower() != ".py":
            raise ValueError(
                f"{field_name} uzantısı .py olmalıdır."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"{field_name} bulunamadı: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{field_name} yolu bir dosya olmalıdır: {path}"
            )

        return path.resolve()

    @staticmethod
    def _validate_timeout(
        timeout_seconds: float,
    ) -> float:
        """Coverage zaman aşımı değerini doğrular."""
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds,
            (int, float),
        ):
            raise TypeError(
                "Coverage zaman aşımı sayısal olmalıdır."
            )

        normalized_timeout = float(timeout_seconds)

        if normalized_timeout <= 0:
            raise ValueError(
                "Coverage zaman aşımı sıfırdan büyük olmalıdır."
            )

        return normalized_timeout

    @staticmethod
    def _calculate_remaining_timeout(
        start_time: float,
        timeout_seconds: float,
    ) -> float:
        """Toplam süreden geriye kalan zaman aşımı değerini hesaplar."""
        elapsed_seconds = time.perf_counter() - start_time
        remaining_seconds = timeout_seconds - elapsed_seconds

        if remaining_seconds <= 0:
            raise CoverageExecutionTimeoutError(
                "Coverage işlemi zaman aşımına uğradı."
            )

        return remaining_seconds

    @staticmethod
    def _read_report(
        report_file: Path,
    ) -> dict[str, Any]:
        """Coverage JSON raporunu okur."""
        if not report_file.exists():
            raise RuntimeError(
                "Coverage JSON rapor dosyası bulunamadı."
            )

        try:
            return json.loads(
                report_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Coverage JSON raporu okunamadı."
            ) from error

    @staticmethod
    def _find_file_summary(
        report_data: dict[str, Any],
        source_file: Path,
    ) -> dict[str, Any]:
        """Raporda hedef kaynak dosyaya ait özet bilgiyi bulur."""
        files = report_data.get("files")

        if not isinstance(files, dict):
            raise RuntimeError(
                "Coverage raporunda dosya bilgileri bulunamadı."
            )

        resolved_source_file = source_file.resolve()

        for reported_path, file_data in files.items():
            candidate_path = Path(reported_path)

            if not candidate_path.is_absolute():
                candidate_path = (
                    Path.cwd() / candidate_path
                )

            if candidate_path.resolve() == resolved_source_file:
                summary = file_data.get("summary")

                if not isinstance(summary, dict):
                    raise RuntimeError(
                        "Coverage dosya özeti bulunamadı."
                    )

                return summary

        raise RuntimeError(
            "Coverage raporunda hedef kaynak dosya bulunamadı."
        )

    @classmethod
    def _create_result(
        cls,
        source_file: Path,
        test_file: Path,
        test_exit_code: int,
        duration_seconds: float,
        file_summary: dict[str, Any],
    ) -> CoverageResult:
        """Coverage JSON özetinden CoverageResult oluşturur."""
        total_line_count = cls._read_integer(
            file_summary,
            "num_statements",
        )
        covered_line_count = cls._read_integer(
            file_summary,
            "covered_lines",
        )
        missing_line_count = cls._read_integer(
            file_summary,
            "missing_lines",
        )

        total_branch_count = cls._read_integer(
            file_summary,
            "num_branches",
        )
        covered_branch_count = cls._read_integer(
            file_summary,
            "covered_branches",
        )
        missing_branch_count = cls._read_integer(
            file_summary,
            "missing_branches",
        )

        return CoverageResult(
            source_file=source_file,
            test_file=test_file,
            line_coverage_percent=cls._calculate_percentage(
                covered_count=covered_line_count,
                total_count=total_line_count,
            ),
            branch_coverage_percent=cls._calculate_percentage(
                covered_count=covered_branch_count,
                total_count=total_branch_count,
            ),
            covered_line_count=covered_line_count,
            missing_line_count=missing_line_count,
            total_line_count=total_line_count,
            covered_branch_count=covered_branch_count,
            missing_branch_count=missing_branch_count,
            total_branch_count=total_branch_count,
            test_exit_code=test_exit_code,
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def _read_integer(
        summary: dict[str, Any],
        field_name: str,
    ) -> int:
        """Coverage özetindeki zorunlu tam sayı alanını okur."""
        value = summary.get(field_name)

        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                "Coverage raporundaki alan geçersiz: "
                f"{field_name}"
            )

        return value

    @staticmethod
    def _calculate_percentage(
        covered_count: int,
        total_count: int,
    ) -> float:
        """
        Kapsanan ve toplam öğe sayısından yüzde hesaplar.

        Ölçülebilir öğe bulunmadığında ilgili coverage değeri yüzde
        100 kabul edilir.
        """
        if total_count == 0:
            return 100.0

        return round(
            covered_count / total_count * 100,
            2,
        )