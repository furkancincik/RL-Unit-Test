from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CoverageStopReason(str, Enum):
    TARGET_REACHED = "TARGET_REACHED"
    CANDIDATES_EXHAUSTED = "CANDIDATES_EXHAUSTED"
    PROVEN_PLATEAU = "PROVEN_PLATEAU"
    TEST_LIMIT_REACHED = "TEST_LIMIT_REACHED"
    DEADLINE_REACHED = "DEADLINE_REACHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class CoverageProgressSnapshot:
    """Tek bir job run'ına ait immutable ve güvenli coverage ilerlemesi."""

    revision: int
    stage: str
    metric: str
    coverage_percent: float
    line_percent: float
    branch_percent: float | None
    covered_lines: int
    total_lines: int
    covered_branches: int
    total_branches: int
    candidate_count: int
    validated_count: int
    effective_test_count: int
    last_gain_percent: float
    last_new_line_count: int
    last_new_branch_count: int
    plateau_count: int
    stop_reason: CoverageStopReason | None

    def __post_init__(self) -> None:
        for name in (
            "revision", "covered_lines", "total_lines", "covered_branches",
            "total_branches", "candidate_count", "validated_count",
            "effective_test_count", "last_new_line_count",
            "last_new_branch_count", "plateau_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if self.revision == 0:
            raise ValueError("revision pozitif olmalıdır.")
        if self.stage != "COVERAGE_OPTIMIZATION":
            raise ValueError("stage COVERAGE_OPTIMIZATION olmalıdır.")
        if self.metric not in {"LINE", "COMBINED"}:
            raise ValueError("metric LINE veya COMBINED olmalıdır.")
        for name in ("coverage_percent", "line_percent", "last_gain_percent"):
            self._validate_percent(getattr(self, name), name)
        if self.branch_percent is not None:
            self._validate_percent(self.branch_percent, "branch_percent")
        if self.covered_lines > self.total_lines:
            raise ValueError("covered_lines total_lines değerini aşamaz.")
        if self.covered_branches > self.total_branches:
            raise ValueError("covered_branches total_branches değerini aşamaz.")
        if self.validated_count > self.candidate_count:
            raise ValueError("validated_count candidate_count değerini aşamaz.")
        if self.effective_test_count > self.validated_count:
            raise ValueError("effective_test_count validated_count değerini aşamaz.")
        if self.last_new_line_count > self.covered_lines:
            raise ValueError("last_new_line_count covered_lines değerini aşamaz.")
        if self.last_new_branch_count > self.covered_branches:
            raise ValueError(
                "last_new_branch_count covered_branches değerini aşamaz."
            )
        if self.plateau_count > self.validated_count:
            raise ValueError("plateau_count validated_count değerini aşamaz.")
        expected_gain = self._percentage(
            self.last_new_line_count + self.last_new_branch_count,
            self.total_lines + self.total_branches,
        )
        if not math.isclose(self.last_gain_percent, expected_gain, abs_tol=1e-9):
            raise ValueError(
                "last_gain_percent exact yeni coverage sayaçlarıyla eşleşmelidir."
            )
        expected_line = self._percentage(self.covered_lines, self.total_lines)
        if not math.isclose(self.line_percent, expected_line, abs_tol=1e-9):
            raise ValueError("line_percent exact line sayaçlarıyla eşleşmelidir.")
        if self.total_branches == 0:
            if self.metric != "LINE" or self.branch_percent is not None:
                raise ValueError("Branch yoksa LINE metriği kullanılmalıdır.")
            expected = expected_line
        else:
            if self.metric != "COMBINED" or self.branch_percent is None:
                raise ValueError("Branch varsa COMBINED metriği kullanılmalıdır.")
            expected_branch = self._percentage(
                self.covered_branches, self.total_branches
            )
            if not math.isclose(
                self.branch_percent, expected_branch, abs_tol=1e-9
            ):
                raise ValueError(
                    "branch_percent exact branch sayaçlarıyla eşleşmelidir."
                )
            expected = self._percentage(
                self.covered_lines + self.covered_branches,
                self.total_lines + self.total_branches,
            )
        if not math.isclose(self.coverage_percent, expected, abs_tol=1e-9):
            raise ValueError("coverage_percent exact combined formülle eşleşmelidir.")
        if self.stop_reason is not None and not isinstance(
            self.stop_reason, CoverageStopReason
        ):
            raise TypeError("stop_reason CoverageStopReason olmalıdır.")

    @staticmethod
    def _validate_percent(value: object, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            raise ValueError(f"{name} finite 0..100 aralığında olmalıdır.")

    @staticmethod
    def _percentage(covered: int, total: int) -> float:
        return 0.0 if total == 0 else covered / total * 100.0

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["stop_reason"] = (
            self.stop_reason.value if self.stop_reason is not None else None
        )
        return payload
