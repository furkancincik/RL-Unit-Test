from __future__ import annotations

from models.coverage_result import CoverageResult
from rl.coverage_state import CoverageState


class CoverageStateMapper:
    """Coverage sonucunu RL durum modeline dönüştürür."""

    @staticmethod
    def map(
        coverage_result: CoverageResult,
        executed_tests: int,
        missing_lines: tuple[int, ...] = (),
    ) -> CoverageState:
        """
        CoverageResult nesnesinden CoverageState oluşturur.

        CoverageResult yalnızca eksik satır sayısını tuttuğu için eksik satır
        numaraları ayrıca alınır. Satır numaraları henüz bilinmiyorsa boş tuple
        kullanılabilir.
        """

        return CoverageState(
            coverage_percentage=coverage_result.line_coverage_percent,
            executed_tests=executed_tests,
            missing_lines=missing_lines,
            uncovered_branches=coverage_result.missing_branch_count,
        )