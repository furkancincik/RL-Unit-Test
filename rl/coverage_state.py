from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CoverageState:
    """RL ajanının kullanacağı coverage durumunu temsil eder."""

    coverage_percentage: float
    executed_tests: int
    missing_lines: tuple[int, ...]
    uncovered_branches: int

    def __post_init__(self) -> None:
        """Durum verilerinin geçerli olup olmadığını doğrular."""

        if not 0.0 <= self.coverage_percentage <= 100.0:
            raise ValueError(
                "Coverage percentage must be between 0 and 100."
            )

        if self.executed_tests < 0 or self.uncovered_branches < 0:
            raise ValueError("Counts cannot be negative.")

        if any(line_number <= 0 for line_number in self.missing_lines):
            raise ValueError(
                "Missing line numbers must be positive."
            )

    @property
    def missing_line_count(self) -> int:
        """Coverage dışında kalan satır sayısını döndürür."""

        return len(self.missing_lines)

    @property
    def is_fully_covered(self) -> bool:
        """Kodun tamamen kapsanıp kapsanmadığını belirtir."""

        return (
            self.coverage_percentage == 100.0
            and not self.missing_lines
            and self.uncovered_branches == 0
        )