from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)


class CoverageExecutionTimeoutError(TimeoutError):
    """Coverage ölçüm işlemi zaman aşımına uğradığında oluşur."""


class CoverageService:
    """
    Pytest çalıştırarak satır ve branch coverage değerlerini ölçer.

    Servis iki farklı ölçüm sunar:

    - measure:
        Kaynak dosyanın tamamına ait coverage sonucunu üretir.

    - measure_function:
        Dosya geneli coverage sonucunun yanında yalnızca hedef
        fonksiyonun satır aralığına ait coverage sonucunu üretir.
    """

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
            source_file:
                Coverage ölçümü yapılacak Python kaynak dosyası.

            test_file:
                Çalıştırılacak pytest dosyası.

            timeout_seconds:
                Coverage işlemi için azami çalışma süresi.

        Returns:
            Dosya geneline ait satır ve branch coverage bilgilerini
            içeren CoverageResult.

        Raises:
            TypeError:
                Yol veya zaman aşımı değerinin türü geçersizse.

            ValueError:
                Yol, dosya uzantısı veya zaman aşımı geçersizse.

            FileNotFoundError:
                Kaynak veya test dosyası bulunamazsa.

            CoverageExecutionTimeoutError:
                İşlem zaman aşımına uğrarsa.

            RuntimeError:
                Coverage raporu oluşturulamaz veya okunamazsa.
        """
        (
            normalized_source_file,
            normalized_test_file,
            test_exit_code,
            duration_seconds,
            file_data,
        ) = self._execute_measurement(
            source_file=source_file,
            test_file=test_file,
            timeout_seconds=timeout_seconds,
        )

        file_summary = self._read_file_summary(
            file_data=file_data,
        )

        return self._create_result(
            source_file=normalized_source_file,
            test_file=normalized_test_file,
            test_exit_code=test_exit_code,
            duration_seconds=duration_seconds,
            file_summary=file_summary,
        )

    def measure_function(
        self,
        source_file: str | Path,
        test_file: str | Path,
        function_name: str,
        start_line: int,
        end_line: int,
        *,
        timeout_seconds: float = 30.0,
    ) -> FunctionCoverageResult:
        """
        Belirli bir fonksiyon için coverage ölçümü gerçekleştirir.

        Coverage işlemi kaynak dosyanın tamamı üzerinde çalıştırılır.
        Ardından coverage JSON raporundaki satır ve branch ayrıntıları,
        hedef fonksiyonun başlangıç ve bitiş satırlarına göre filtrelenir.

        Args:
            source_file:
                Fonksiyonun bulunduğu Python kaynak dosyası.

            test_file:
                Çalıştırılacak pytest dosyası.

            function_name:
                Coverage sonucu oluşturulacak fonksiyonun adı.

            start_line:
                Fonksiyonun kaynak dosyadaki başlangıç satırı.

            end_line:
                Fonksiyonun kaynak dosyadaki bitiş satırı.

            timeout_seconds:
                Coverage işlemi için azami çalışma süresi.

        Returns:
            Hem hedef fonksiyona hem de dosyanın tamamına ait coverage
            bilgilerini içeren FunctionCoverageResult.

        Raises:
            TypeError:
                Parametre türlerinden biri geçersizse.

            ValueError:
                Fonksiyon adı veya satır aralığı geçersizse.

            FileNotFoundError:
                Kaynak veya test dosyası bulunamazsa.

            CoverageExecutionTimeoutError:
                Coverage işlemi zaman aşımına uğrarsa.

            RuntimeError:
                Coverage raporu okunamaz veya gerekli alanları
                içermiyorsa.
        """
        normalized_function_name = self._normalize_function_name(
            function_name
        )

        normalized_start_line, normalized_end_line = (
            self._validate_function_line_range(
                start_line=start_line,
                end_line=end_line,
            )
        )

        (
            normalized_source_file,
            normalized_test_file,
            test_exit_code,
            duration_seconds,
            file_data,
        ) = self._execute_measurement(
            source_file=source_file,
            test_file=test_file,
            timeout_seconds=timeout_seconds,
        )

        file_summary = self._read_file_summary(
            file_data=file_data,
        )

        file_coverage = self._create_result(
            source_file=normalized_source_file,
            test_file=normalized_test_file,
            test_exit_code=test_exit_code,
            duration_seconds=duration_seconds,
            file_summary=file_summary,
        )

        executed_lines = self._read_line_collection(
            file_data=file_data,
            field_name="executed_lines",
        )

        missing_lines = self._read_line_collection(
            file_data=file_data,
            field_name="missing_lines",
        )

        function_covered_lines = self._filter_lines_by_range(
            lines=executed_lines,
            start_line=normalized_start_line,
            end_line=normalized_end_line,
        )

        function_missing_lines = self._filter_lines_by_range(
            lines=missing_lines,
            start_line=normalized_start_line,
            end_line=normalized_end_line,
        )

        executed_branches = self._read_branch_collection(
            file_data=file_data,
            field_name="executed_branches",
        )

        missing_branches = self._read_branch_collection(
            file_data=file_data,
            field_name="missing_branches",
        )

        function_covered_branches = (
            self._filter_branches_by_source_range(
                branches=executed_branches,
                start_line=normalized_start_line,
                end_line=normalized_end_line,
            )
        )

        function_missing_branches = (
            self._filter_branches_by_source_range(
                branches=missing_branches,
                start_line=normalized_start_line,
                end_line=normalized_end_line,
            )
        )

        covered_line_count = len(
            function_covered_lines
        )

        missing_line_count = len(
            function_missing_lines
        )

        total_line_count = (
            covered_line_count
            + missing_line_count
        )

        covered_branch_count = len(
            function_covered_branches
        )

        missing_branch_count = len(
            function_missing_branches
        )

        total_branch_count = (
            covered_branch_count
            + missing_branch_count
        )

        return FunctionCoverageResult(
            source_file=normalized_source_file,
            test_file=normalized_test_file,
            function_name=normalized_function_name,
            start_line=normalized_start_line,
            end_line=normalized_end_line,
            line_coverage_percent=self._calculate_percentage(
                covered_count=covered_line_count,
                total_count=total_line_count,
            ),
            branch_coverage_percent=self._calculate_percentage(
                covered_count=covered_branch_count,
                total_count=total_branch_count,
            ),
            covered_lines=function_covered_lines,
            missing_lines=function_missing_lines,
            covered_branch_count=covered_branch_count,
            missing_branch_count=missing_branch_count,
            covered_branches=function_covered_branches,
            missing_branches=function_missing_branches,
            test_exit_code=test_exit_code,
            duration_seconds=duration_seconds,
            file_coverage=file_coverage,
        )

    def _execute_measurement(
        self,
        source_file: str | Path,
        test_file: str | Path,
        timeout_seconds: float,
    ) -> tuple[
        Path,
        Path,
        int,
        float,
        dict[str, Any],
    ]:
        """
        Coverage işlemini çalıştırır ve hedef dosyanın ham raporunu döndürür.

        Bu yardımcı metot hem dosya bazlı hem de fonksiyon bazlı ölçüm
        tarafından kullanılır. Böylece aynı ölçüm sırasında pytest ve
        coverage işlemleri yalnızca bir kez çalıştırılır.
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
        working_directory = Path.cwd().resolve()

        with tempfile.TemporaryDirectory(
            prefix="rl_unit_test_coverage_"
        ) as temporary_directory:
            temporary_path = Path(
                temporary_directory
            )

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
                    cwd=working_directory,
                )
            except subprocess.TimeoutExpired as error:
                duration_seconds = (
                    time.perf_counter()
                    - start_time
                )

                raise CoverageExecutionTimeoutError(
                    "Coverage ölçüm işlemi zaman aşımına uğradı. "
                    f"Süre: {duration_seconds:.3f} saniye."
                ) from error

            remaining_timeout = (
                self._calculate_remaining_timeout(
                    start_time=start_time,
                    timeout_seconds=normalized_timeout,
                )
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
                    cwd=working_directory,
                )
            except subprocess.TimeoutExpired as error:
                duration_seconds = (
                    time.perf_counter()
                    - start_time
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

            file_data = self._find_file_data(
                report_data=report_data,
                source_file=normalized_source_file,
            )

        duration_seconds = (
            time.perf_counter()
            - start_time
        )

        return (
            normalized_source_file,
            normalized_test_file,
            run_process.returncode,
            duration_seconds,
            file_data,
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
            f"--include={source_file.as_posix()}",
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
            f"--include={source_file.as_posix()}",
        )

    @staticmethod
    def _normalize_python_file(
        file_path: str | Path,
        field_name: str,
    ) -> Path:
        """
        Python dosyası yolunu doğrular.

        Args:
            file_path:
                Doğrulanacak dosya yolu.

            field_name:
                Hata mesajında kullanılacak alan adı.

        Returns:
            Mutlak ve doğrulanmış Path nesnesi.
        """
        if not isinstance(
            file_path,
            (
                str,
                Path,
            ),
        ):
            raise TypeError(
                f"{field_name} yolu string veya Path olmalıdır."
            )

        if (
            isinstance(file_path, str)
            and not file_path.strip()
        ):
            raise ValueError(
                f"{field_name} yolu boş olamaz."
            )

        path = Path(
            file_path
        )

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
    def _normalize_function_name(
        function_name: str,
    ) -> str:
        """Fonksiyon adını doğrular ve normalize eder."""
        if not isinstance(
            function_name,
            str,
        ):
            raise TypeError(
                "function_name string olmalıdır."
            )

        normalized_function_name = (
            function_name.strip()
        )

        if not normalized_function_name:
            raise ValueError(
                "function_name boş olamaz."
            )

        if not normalized_function_name.isidentifier():
            raise ValueError(
                "function_name geçerli bir Python "
                "tanımlayıcısı olmalıdır."
            )

        return normalized_function_name

    @staticmethod
    def _validate_function_line_range(
        start_line: int,
        end_line: int,
    ) -> tuple[int, int]:
        """Fonksiyon başlangıç ve bitiş satırlarını doğrular."""
        if (
            isinstance(start_line, bool)
            or not isinstance(start_line, int)
        ):
            raise TypeError(
                "start_line bir tam sayı olmalıdır."
            )

        if (
            isinstance(end_line, bool)
            or not isinstance(end_line, int)
        ):
            raise TypeError(
                "end_line bir tam sayı olmalıdır."
            )

        if start_line < 1:
            raise ValueError(
                "start_line 1 veya daha büyük olmalıdır."
            )

        if end_line < start_line:
            raise ValueError(
                "end_line start_line değerinden "
                "küçük olamaz."
            )

        return (
            start_line,
            end_line,
        )

    @staticmethod
    def _validate_timeout(
        timeout_seconds: float,
    ) -> float:
        """Coverage zaman aşımı değerini doğrular."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(
                timeout_seconds,
                (
                    int,
                    float,
                ),
            )
        ):
            raise TypeError(
                "Coverage zaman aşımı sayısal olmalıdır."
            )

        normalized_timeout = float(
            timeout_seconds
        )

        if not math.isfinite(
            normalized_timeout
        ):
            raise ValueError(
                "Coverage zaman aşımı sonlu olmalıdır."
            )

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
        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        remaining_seconds = (
            timeout_seconds
            - elapsed_seconds
        )

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
            report_data = json.loads(
                report_file.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Coverage JSON raporu okunamadı."
            ) from error

        if not isinstance(
            report_data,
            dict,
        ):
            raise RuntimeError(
                "Coverage JSON raporunun kök değeri "
                "bir nesne olmalıdır."
            )

        return report_data

    @staticmethod
    def _find_file_data(
        report_data: dict[str, Any],
        source_file: Path,
    ) -> dict[str, Any]:
        """Raporda hedef kaynak dosyaya ait tüm bilgiyi bulur."""
        files = report_data.get(
            "files"
        )

        if not isinstance(
            files,
            dict,
        ):
            raise RuntimeError(
                "Coverage raporunda dosya bilgileri bulunamadı."
            )

        resolved_source_file = (
            source_file.resolve()
        )

        for reported_path, file_data in files.items():
            if not isinstance(
                reported_path,
                str,
            ):
                continue

            if not isinstance(
                file_data,
                dict,
            ):
                continue

            candidate_path = Path(
                reported_path
            )

            if not candidate_path.is_absolute():
                candidate_path = (
                    Path.cwd()
                    / candidate_path
                )

            if (
                candidate_path.resolve()
                == resolved_source_file
            ):
                return file_data

        raise RuntimeError(
            "Coverage raporunda hedef kaynak dosya bulunamadı."
        )

    @staticmethod
    def _read_file_summary(
        file_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Dosya verisi içerisindeki coverage özetini döndürür."""
        summary = file_data.get(
            "summary"
        )

        if not isinstance(
            summary,
            dict,
        ):
            raise RuntimeError(
                "Coverage dosya özeti bulunamadı."
            )

        return summary

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
    def _read_line_collection(
        file_data: dict[str, Any],
        field_name: str,
    ) -> tuple[int, ...]:
        """Coverage raporundaki satır numarası listesini okur."""
        raw_lines = file_data.get(
            field_name
        )

        if not isinstance(
            raw_lines,
            list,
        ):
            raise RuntimeError(
                "Coverage raporundaki satır alanı geçersiz: "
                f"{field_name}"
            )

        normalized_lines: list[int] = []

        for raw_line in raw_lines:
            if (
                isinstance(raw_line, bool)
                or not isinstance(raw_line, int)
            ):
                raise RuntimeError(
                    "Coverage raporundaki satır numarası "
                    f"geçersiz: {field_name}"
                )

            if raw_line < 1:
                raise RuntimeError(
                    "Coverage raporundaki satır numarası "
                    "pozitif olmalıdır."
                )

            normalized_lines.append(
                raw_line
            )

        return tuple(
            sorted(
                set(normalized_lines)
            )
        )

    @staticmethod
    def _read_branch_collection(
        file_data: dict[str, Any],
        field_name: str,
    ) -> tuple[tuple[int, int], ...]:
        """
        Coverage raporundaki branch çiftlerini okur.

        Coverage.py branch verilerini kaynak ve hedef satır
        çiftleri şeklinde döndürür.
        """
        raw_branches = file_data.get(
            field_name,
            [],
        )

        if not isinstance(
            raw_branches,
            list,
        ):
            raise RuntimeError(
                "Coverage raporundaki branch alanı geçersiz: "
                f"{field_name}"
            )

        normalized_branches: list[
            tuple[int, int]
        ] = []

        for raw_branch in raw_branches:
            if (
                not isinstance(raw_branch, list)
                or len(raw_branch) != 2
            ):
                raise RuntimeError(
                    "Coverage raporundaki branch kaydı "
                    f"geçersiz: {field_name}"
                )

            source_line = raw_branch[0]
            target_line = raw_branch[1]

            if (
                isinstance(source_line, bool)
                or not isinstance(source_line, int)
                or isinstance(target_line, bool)
                or not isinstance(target_line, int)
            ):
                raise RuntimeError(
                    "Coverage raporundaki branch satırları "
                    f"geçersiz: {field_name}"
                )

            normalized_branches.append(
                (
                    source_line,
                    target_line,
                )
            )

        return tuple(
            sorted(
                set(normalized_branches)
            )
        )

    @staticmethod
    def _filter_lines_by_range(
        lines: tuple[int, ...],
        start_line: int,
        end_line: int,
    ) -> tuple[int, ...]:
        """Satırları hedef fonksiyon aralığına göre filtreler."""
        return tuple(
            line
            for line in lines
            if start_line <= line <= end_line
        )

    @staticmethod
    def _filter_branches_by_source_range(
        branches: tuple[tuple[int, int], ...],
        start_line: int,
        end_line: int,
    ) -> tuple[tuple[int, int], ...]:
        """
        Branchleri kaynak satırına göre fonksiyon aralığında filtreler.

        Branch hedefi fonksiyon dışındaki özel coverage.py çıkış
        değerlerinden biri olabilir. Bu nedenle kapsam belirlenirken
        branch'in kaynak satırı esas alınır.
        """
        return tuple(
            branch
            for branch in branches
            if start_line <= branch[0] <= end_line
        )

    @staticmethod
    def _read_integer(
        summary: dict[str, Any],
        field_name: str,
    ) -> int:
        """Coverage özetindeki zorunlu tam sayı alanını okur."""
        value = summary.get(
            field_name
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
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

        Ölçülebilir öğe bulunmadığında ilgili coverage değeri
        yüzde 100 kabul edilir.
        """
        if total_count == 0:
            return 100.0

        return round(
            covered_count
            / total_count
            * 100,
            2,
        )
