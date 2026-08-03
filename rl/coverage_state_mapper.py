from __future__ import annotations

from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from rl.coverage_state import CoverageState


CoverageInput = CoverageResult | FunctionCoverageResult


class CoverageStateMapper:
    """
    Coverage sonucunu RL tarafından kullanılabilecek
    CoverageState modeline dönüştürür.

    CoverageResult yalnızca eksik satır sayısını tuttuğundan,
    gerçek satır numaraları bilinmiyorsa temsili satırlar üretilir.

    FunctionCoverageResult kullanıldığında ise gerçek eksik
    satır numaraları doğrudan kullanılır.
    """

    @staticmethod
    def map(
        coverage_result: CoverageInput,
        executed_tests: int,
        missing_lines: tuple[int, ...] | None = None,
    ) -> CoverageState:
        """
        Coverage sonucundan CoverageState oluşturur.

        Args:
            coverage_result:
                Coverage ölçümünün sonucu.

            executed_tests:
                Şu ana kadar çalıştırılan test sayısı.

            missing_lines:
                Gerçek eksik satır numaraları.
                None verilirse;

                - CoverageResult için temsili satırlar oluşturulur.
                - FunctionCoverageResult için gerçek eksik satırlar
                  kullanılır.
        """
        CoverageStateMapper._validate_coverage_result(
            coverage_result
        )

        CoverageStateMapper._validate_executed_tests(
            executed_tests
        )

        normalized_missing_lines = (
            CoverageStateMapper._normalize_missing_lines(
                coverage_result=coverage_result,
                missing_lines=missing_lines,
            )
        )

        return CoverageState(
            coverage_percentage=(
                coverage_result.line_coverage_percent
            ),
            executed_tests=executed_tests,
            missing_lines=normalized_missing_lines,
            uncovered_branches=(
                coverage_result.missing_branch_count
            ),
        )

    @staticmethod
    def _normalize_missing_lines(
        coverage_result: CoverageInput,
        missing_lines: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """
        Eksik satır bilgisini doğrular veya oluşturur.
        """

        if missing_lines is None:

            if isinstance(
                coverage_result,
                FunctionCoverageResult,
            ):
                return coverage_result.missing_lines

            return tuple(
                range(
                    1,
                    coverage_result.missing_line_count + 1,
                )
            )

        if not isinstance(missing_lines, tuple):
            raise TypeError(
                "missing_lines bir tuple veya None olmalıdır."
            )

        if any(
            isinstance(line_number, bool)
            or not isinstance(line_number, int)
            for line_number in missing_lines
        ):
            raise TypeError(
                "missing_lines yalnızca tam sayı değerler "
                "içermelidir."
            )

        if any(
            line_number < 1
            for line_number in missing_lines
        ):
            raise ValueError(
                "Eksik satır numaraları 1 veya daha büyük "
                "olmalıdır."
            )

        if len(set(missing_lines)) != len(missing_lines):
            raise ValueError(
                "missing_lines tekrar eden satır numarası "
                "içeremez."
            )

        if (
            len(missing_lines)
            != coverage_result.missing_line_count
        ):
            raise ValueError(
                "Verilen eksik satır sayısı CoverageResult ile "
                "uyuşmuyor."
            )

        return tuple(sorted(missing_lines))

    @staticmethod
    def _validate_coverage_result(
        coverage_result: CoverageInput,
    ) -> None:
        if not isinstance(
            coverage_result,
            (
                CoverageResult,
                FunctionCoverageResult,
            ),
        ):
            raise TypeError(
                "coverage_result bir CoverageResult "
                "örneği olmalıdır."
            )

    @staticmethod
    def _validate_executed_tests(
        executed_tests: int,
    ) -> None:
        if (
            isinstance(executed_tests, bool)
            or not isinstance(executed_tests, int)
        ):
            raise TypeError(
                "executed_tests bir tam sayı olmalıdır."
            )

        if executed_tests < 0:
            raise ValueError(
                "executed_tests negatif olamaz."
            )